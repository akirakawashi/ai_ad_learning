"""Определение бренда через зрительно-языковую модель.

Обученный классификатор учится на макетах, а макеты меняются каждые пару месяцев.
VLM смотрит на логотип и надпись, то есть на то, что у бренда не меняется годами.
Плюс ей не нужна размеченная выборка — а её у нас и нет.

Модель поднимается отдельно, через `llama-server`, и отвечает по HTTP. Ответ
загнан в жёсткую JSON-схему: свободный текст от VLM нестабилен, а схема на уровне
декодера физически не даёт ответить чем-то посторонним.

Кроме самого бренда модель обязана указать, **на чём основан ответ**. Это и есть
защита от выдумывания: на проверке все ложные срабатывания приходились на кадры,
где названия бренда в прочитанном тексте не было — модель видела букву T на
розовом и достраивала Tele2. Основание ответа позволяет такие случаи отделить, не
теряя честные, где виден фирменный знак без надписи.
"""

from __future__ import annotations

import base64
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import requests

from adlearn.classification.config import TELECOM_BRANDS, VLM_UNCLEAR

ANSWERS = (*TELECOM_BRANDS, "other", VLM_UNCLEAR)

BASIS_NAME = "name_read"
BASIS_LOGO = "logo_only"
BASIS_COLORS = "colors_only"
BASES = (BASIS_NAME, BASIS_LOGO, BASIS_COLORS)

MAX_SIDE = 1400
"""Длинная сторона кадра при отправке.

Крупные картинки разворачиваются в тысячи токенов и переполняют контекст —
на проверке ровно так упали четыре кадра от 4.4 Мпикс и больше. Щиту с улицы
такое разрешение и не нужно: надпись читается и в полутора тысячах пикселей.
"""

NAME_FORMS: dict[str, tuple[str, ...]] = {
    "beeline": ("билайн", "beeline"),
    "megafon": ("мегафон", "megafon"),
    "mts": ("мтс", "mts"),
    "tele2": ("tele2", "теле2", "т2", "t2"),
}

BRAND_HINTS = """\
beeline (Билайн) — жёлтый и чёрный, круг в чёрно-жёлтую полоску, надпись «Билайн» или Beeline
megafon (МегаФон) — зелёный, иногда с фиолетовым и белым, надпись «МегаФон»
mts (МТС) — красный, часто красный квадрат с белыми буквами, надпись «МТС»
tele2 (Tele2 / t2) — надпись TELE2; в новой айдентике знак t2 — строчные латинские \
буквы t и 2, слитно, часто с розовым квадратом
other — любая другая реклама: магазины, банки, авто, аптеки, застройщики, афиши; \
а также пустой, заклеенный щит или щит «сдаётся в аренду»
unclear — разобрать невозможно"""

NOT_THESE = """\
Это НЕ наши бренды, хотя похожи:
- T-Mobile — розовый фон и большая буква T. Это другой оператор, отвечай other.
- Т-Банк, Тинькофф — жёлтый щит с чёрной буквой Т. Не Билайн.
- Сбер, СберМобайл, аптеки, ВкусВилл — зелёные. Не МегаФон.
- Магнит, Магнит Косметик, Альфа-Банк, Лукойл — красные или розовые. Не МТС и не Tele2.
- Мегамаркет — слово «МЕГА», но это маркетплейс. Не МегаФон.
- Яндекс Go, Яндекс Такси — чёрное на жёлтом. Не Билайн.
- Отдельная буква или пара букв (T, MS, BB, М) сама по себе ничего не значит."""

PROMPT = f"""На фотографии рекламный щит или вывеска. Определи, реклама какого \
оператора сотовой связи на нём размещена.

Возможные ответы:
{BRAND_HINTS}

{NOT_THESE}

Заполни поля так:
- visible_text — весь текст, который удалось прочитать на щите, дословно. Если \
текста нет, оставь пустым. Не пересказывай, а выписывай прочитанное.
- basis — на чём основан твой ответ:
  {BASIS_NAME} — прочитал название бренда в тексте;
  {BASIS_LOGO} — названия нет, но узнал фирменный знак; опиши его в evidence;
  {BASIS_COLORS} — ни названия, ни знака, только цвета и общий вид.
- evidence — что именно тебя убедило.

Если основание — только цвета, честно ставь {BASIS_COLORS} и отвечай {VLM_UNCLEAR} \
либо other. Не угадывай бренд по цвету."""

SCHEMA = {
    "type": "object",
    "properties": {
        "brand": {"type": "string", "enum": list(ANSWERS)},
        "basis": {"type": "string", "enum": list(BASES)},
        "visible_text": {"type": "string"},
        "evidence": {"type": "string"},
    },
    "required": ["brand", "basis", "visible_text", "evidence"],
    "additionalProperties": False,
}

VERDICT_CONFIRMED = "подтверждён"
VERDICT_REVIEW = "на проверку"
VERDICT_REJECTED = "отклонён"


@dataclass(frozen=True)
class VlmAnswer:
    path: Path
    brand: str
    basis: str
    evidence: str
    visible_text: str
    error: str = ""

    @property
    def name_in_text(self) -> bool:
        low = self.visible_text.lower()
        return any(form in low for form in NAME_FORMS.get(self.brand, ()))

    @property
    def verdict(self) -> str:
        """Насколько можно верить названному бренду.

        `other` и `unclear` проверять незачем — они ничего не утверждают.
        Заявленное «прочитал название», не подтверждённое текстом, — это модель
        противоречит сама себе, и такой ответ надёжнее считать догадкой.
        """

        if self.error or self.brand not in NAME_FORMS:
            return VERDICT_CONFIRMED
        if self.basis == BASIS_NAME and self.name_in_text:
            return VERDICT_CONFIRMED
        if self.basis == BASIS_LOGO or self.basis == BASIS_NAME:
            return VERDICT_REVIEW
        return VERDICT_REJECTED

    def decided(self, *, accept_logo: bool = False) -> str:
        """Итоговый ответ после проверки основания."""

        verdict = self.verdict
        if verdict == VERDICT_CONFIRMED:
            return self.brand
        if verdict == VERDICT_REVIEW and accept_logo:
            return self.brand
        return VLM_UNCLEAR


def encode(path: Path) -> str:
    """Кадр в data-URL, уменьшенный до `MAX_SIDE` по длинной стороне."""

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return f"data:image/jpeg;base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"
    height, width = image.shape[:2]
    scale = MAX_SIDE / max(height, width)
    if scale < 1.0:
        image = cv2.resize(
            image,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise RuntimeError(f"Не удалось пережать кадр: {path}")
    return f"data:image/jpeg;base64,{base64.b64encode(np.asarray(buffer)).decode('ascii')}"


def ask(
    path: Path,
    *,
    url: str,
    prompt: str = PROMPT,
    temperature: float = 0.0,
    timeout: float = 180.0,
) -> VlmAnswer:
    """Один кадр — один ответ по схеме."""

    body: dict[str, Any] = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": encode(path)}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "temperature": temperature,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "brand", "schema": SCHEMA, "strict": True},
        },
    }
    try:
        response = requests.post(
            f"{url.rstrip('/')}/v1/chat/completions", json=body, timeout=timeout
        )
        response.raise_for_status()
        parsed = json.loads(response.json()["choices"][0]["message"]["content"])
    except Exception as error:  # noqa: BLE001 — причина уходит в отчёт, прогон не рвём
        return VlmAnswer(
            path=path,
            brand=VLM_UNCLEAR,
            basis=BASIS_COLORS,
            evidence="",
            visible_text="",
            error=str(error),
        )
    return VlmAnswer(
        path=path,
        brand=parsed.get("brand", VLM_UNCLEAR),
        basis=parsed.get("basis", BASIS_COLORS),
        evidence=parsed.get("evidence", ""),
        visible_text=parsed.get("visible_text", ""),
    )


def ask_many(paths: Sequence[Path], *, url: str, **kwargs: object) -> list[VlmAnswer]:
    return [ask(path, url=url, **kwargs) for path in paths]  # type: ignore[arg-type]


def health(url: str, *, timeout: float = 5.0) -> bool:
    try:
        return requests.get(f"{url.rstrip('/')}/health", timeout=timeout).ok
    except requests.RequestException:
        return False
