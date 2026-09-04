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
import re
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
    "mts": ("мтс", "mts", "mtc"),
    "tele2": ("tele2", "теле2", "т2", "t2"),
    "volna": ("волна", "volna", "win mobile"),
    "miranda": ("миранда", "miranda"),
    "plus7": ("+7 телеком", "+7телеком", "7 телеком", "7телеком", "7telecom"),
}
"""Как название пишут на щите.

`mtc` — это МТС, прочитанный латиницей: буквы совпадают по начертанию, и модель
выписывает их то так, то так. Голый «+7» сюда не входит: так начинается любой
телефон.
"""

STANDALONE_WORDS: dict[str, tuple[str, ...]] = {
    "plus7": ("телеком",),
}
"""Слова, которые подтверждают бренд, только если стоят отдельно.

Знак «+7» — картинка, и в прочитанный текст модель его почти никогда не выписывает:
остаётся одно слово «Телеком». Само по себе оно бренд подтверждает, но лишь
целым словом — внутри «Ростелеком» на щите Миранды оно ничего не значит.
"""

PACKED_MIN_LENGTH = 3
"""Короче этого форма в тексте без пробелов не ищется.

Буквы нового знака МТС модель читает как «М Т С», поэтому текст сверяется ещё и
со схлопнутыми пробелами. Но «т2» в таком тексте нашлось бы внутри «кабинет 2»,
так что двухбуквенные формы ищутся только как есть.
"""

BRAND_HINTS = """\
beeline (Билайн) — жёлтый и чёрный, круг в чёрно-жёлтую полоску, надпись «Билайн» или \
Beeline; в новой айдентике рядом с полосатым шаром стоит буква «б.»
megafon (МегаФон) — главный цвет зелёный, он есть почти всегда; фиолетовый только вторым \
цветом рядом с зелёным, оранжевого не бывает; надпись «МегаФон»; знак — три круга в ряд: \
зелёный, зелёный с вырезом и фиолетовый
mts (МТС) — красный. Старый знак: белое яйцо на красном квадрате и надпись «МТС». \
Новый знак: красный квадрат с белыми буквами М, Т, С по углам. МТС Банк, МТС Shop \
и латинское MTS — тоже mts
tele2 (Tele2 / t2) — надпись TELE2; в новой айдентике знак t2 — строчные латинские \
буквы t и 2, слитно, часто с розовым квадратом
volna (Волна) — синий фон с розовым или малиновым; знак — белый кружок и розовая \
капля, вместе похожи на букву В; надпись «волна» строчными. Дочерний бренд WIN mobile \
(бирюзовый, знак-галочка W) считай той же volna
miranda (Миранда) — фиолетовый с оранжевым, никогда с зелёным; знак — лента из \
фиолетового и оранжевого треугольников, похожа на галочку или букву М; надпись \
«Миранда», рядом бывает «Как и должно быть», «от Ростелеком», «Федеральный оператор \
мобильной связи», сайт miranda-media.ru
plus7 (+7 Телеком) — синий, реже красный фон; белый знак «+7», где плюс похож на \
крест, рядом слово «Телеком»; сайт 7telecom.ru
other — любая другая реклама: магазины, банки, авто, аптеки, застройщики, афиши; \
а также пустой, заклеенный щит или щит «сдаётся в аренду»
unclear — разобрать невозможно"""

NOT_THESE = """\
Это НЕ наши бренды, хотя похожи:
- T-Mobile — розовый фон и одна большая буква T, иногда с квадратиками по бокам. Это \
другой оператор, отвечай other. У t2 буква t маленькая и к ней приклеена двойка.
- Т-Банк, Тинькофф — жёлтый щит с чёрной буквой Т. Не Билайн.
- Сбер, СберМобайл — зелёная галочка в круге, круг бывает с сине-жёлтым градиентом. \
Аптеки, ВкусВилл — тоже зелёные. Не МегаФон и не Волна.
- Магнит — красная буква M в рамке, Альфа-Банк — красная буква A с чертой под ней, \
Лукойл — красный. Ни один из них не МТС и не Tele2: у МТС буквы М, Т и С или яйцо.
- Мегамаркет — слово «МЕГА», но это маркетплейс. Не МегаФон.
- Яндекс Go, Яндекс Такси — чёрное на жёлтом, жёлтый прямоугольник с шашечками. Не Билайн.
- Ростелеком без слова «Миранда» — other, хотя цвета те же, фиолетовый и оранжевый.
- Фиолетовый с оранжевым и без зелёного — это Миранда, не МегаФон. Фиолетовый без \
зелёного МегаФоном не бывает.
- Слово «Телеком» рядом со знаком +7 — это +7 Телеком, а не Tele2. У Tele2 слова \
«Телеком» нет.
- Телефонный номер, начинающийся с +7, сам по себе ничего не значит. +7 Телеком — \
это только знак и слово «Телеком».
- Отдельная буква или пара букв (T, MS, BB, М, В) сама по себе ничего не значит."""

PROMPT = f"""На фотографии рекламный щит или вывеска. Определи, реклама какого \
оператора сотовой связи на нём размещена.

Возможные ответы:
{BRAND_HINTS}

{NOT_THESE}

Заполни поля так:
- visible_text — весь текст, который удалось прочитать на щите, дословно. Если \
текста нет, оставь пустым. Не пересказывай, а выписывай прочитанное. Если фирменный \
знак состоит из букв или цифр — t2, МТС, +7, б. — выпиши и их: это тоже текст.
- basis — на чём основан твой ответ:
  {BASIS_NAME} — прочитал название бренда в тексте; ставь только если название \
буквально есть в visible_text, слоган или цвета названием не считаются;
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
        packed = re.sub(r"\s+", "", low)
        forms = NAME_FORMS.get(self.brand, ())
        if any(form in low for form in forms):
            return True
        packed_forms = [re.sub(r"\s+", "", form) for form in forms]
        if any(form in packed for form in packed_forms if len(form) >= PACKED_MIN_LENGTH):
            return True
        return any(
            re.search(rf"(?<![a-zа-яё]){word}(?![a-zа-яё])", low)
            for word in STANDALONE_WORDS.get(self.brand, ())
        )

    @property
    def verdict(self) -> str:
        """Насколько можно верить названному бренду.

        `other` и `unclear` проверять незачем — они ничего не утверждают.
        Название в прочитанном тексте — довод сам по себе, каким бы модель ни
        назвала основание: новый знак МТС состоит из букв, и она честно пишет
        «узнала знак», выписав при этом «М Т С». Ответ без названия в тексте
        уходит на проверку — и заявленное «прочитал название», которого в тексте
        нет, тоже: это модель противоречит сама себе. Ответ по одним цветам не
        спасает даже название: на проверке такие ответы были догадками.
        """

        if self.error or self.brand not in NAME_FORMS:
            return VERDICT_CONFIRMED
        if self.basis == BASIS_COLORS:
            return VERDICT_REJECTED
        if self.name_in_text:
            return VERDICT_CONFIRMED
        return VERDICT_REVIEW

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


MAX_ANSWER_TOKENS = 400
"""Лимит длины ответа.

На отдельных кадрах модель зацикливается в поле `evidence` и пишет до конца
контекста, по минуте на кадр. Оборванный ответ не разбирается и уходит в сбой —
это честнее, чем ждать. То же число стоит в `VlmConfig` пайплайна.
"""


def ask(
    path: Path,
    *,
    url: str,
    prompt: str = PROMPT,
    temperature: float = 0.0,
    timeout: float = 180.0,
    max_tokens: int = MAX_ANSWER_TOKENS,
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
        "max_tokens": max_tokens,
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
