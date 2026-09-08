from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from adlearn.classification import vlm
from adlearn.classification.vlm import (
    BASIS_COLORS,
    BASIS_LOGO,
    BASIS_NAME,
    MAX_SIDE,
    VERDICT_CONFIRMED,
    VERDICT_REJECTED,
    VERDICT_REVIEW,
    VlmAnswer,
    encode,
)


def answer(**kwargs: object) -> VlmAnswer:
    base = {
        "path": Path("frame.jpg"),
        "brand": "tele2",
        "basis": BASIS_NAME,
        "evidence": "",
        "visible_text": "",
    }
    return VlmAnswer(**{**base, **kwargs})  # type: ignore[arg-type]


def test_name_read_and_confirmed_by_text_is_trusted() -> None:
    item = answer(visible_text="ЛОВИТ ВЕЗДЕ, ГДЕ ВАМ НУЖНО t2.ru")

    assert item.verdict == VERDICT_CONFIRMED
    assert item.decided() == "tele2"


def test_claimed_name_without_the_name_in_text_is_downgraded() -> None:
    """Модель противоречит сама себе — так выглядели все ложные срабатывания.

    «Прочитал название», а в тексте «Go» или «БИЗНЕС»: на самом деле бренд
    достроен по цвету и одной букве.
    """

    item = answer(brand="beeline", visible_text="Go")

    assert item.verdict == VERDICT_REVIEW
    assert item.decided() == "unclear"


def test_logo_without_a_name_waits_for_a_human() -> None:
    """Знак t2 без надписи — честный довод, но отличить его от чужой буквы T нельзя."""

    item = answer(basis=BASIS_LOGO, evidence="знак t2 в белом круге")

    assert item.verdict == VERDICT_REVIEW
    assert item.decided() == "unclear"
    assert item.decided(accept_logo=True) == "tele2"


def test_colors_alone_are_rejected() -> None:
    item = answer(brand="megafon", basis=BASIS_COLORS, evidence="зелёный фон")

    assert item.verdict == VERDICT_REJECTED
    assert item.decided(accept_logo=True) == "unclear"


def test_other_and_unclear_are_not_second_guessed() -> None:
    for brand in ("other", "unclear"):
        item = answer(brand=brand, basis=BASIS_COLORS)
        assert item.decided() == brand


def test_large_frames_are_shrunk_before_sending(tmp_path: Path) -> None:
    """Крупные кадры разворачивались в тысячи токенов и роняли запрос."""

    path = tmp_path / "big.jpg"
    cv2.imwrite(str(path), np.full((2700, 2700, 3), 200, dtype=np.uint8))

    import base64

    payload = encode(path).split(",", 1)[1]
    decoded = cv2.imdecode(np.frombuffer(base64.b64decode(payload), np.uint8), cv2.IMREAD_COLOR)

    assert decoded is not None
    assert max(decoded.shape[:2]) == MAX_SIDE


def test_logo_basis_with_the_name_in_text_is_trusted() -> None:
    """Новый знак МТС — это буквы: модель пишет «узнала знак», но название выписано."""

    item = answer(brand="mts", basis=BASIS_LOGO, visible_text="М Т С")

    assert item.verdict == VERDICT_CONFIRMED
    assert item.decided() == "mts"


def test_latin_lookalike_of_mts_counts() -> None:
    item = answer(brand="mts", visible_text="MTC")

    assert item.decided() == "mts"


def test_short_forms_are_not_searched_across_spaces() -> None:
    """«т2» нашлось бы внутри «кабинет 2», если схлопнуть пробелы."""

    item = answer(brand="tele2", visible_text="кабинет 2")

    assert item.verdict == VERDICT_REVIEW


def test_plus7_is_confirmed_by_the_word_telecom_alone() -> None:
    """Знак +7 — картинка, в тексте от него остаётся одно слово «Телеком»."""

    item = answer(brand="plus7", visible_text="Телеком\nПриходи со своим номером")

    assert item.decided() == "plus7"


def test_rostelecom_and_a_phone_number_do_not_confirm_plus7() -> None:
    for text in ("Миранда от Ростелеком", "+7 (990) 007-07-07"):
        item = answer(brand="plus7", visible_text=text)
        assert item.verdict == VERDICT_REVIEW, text


ANSWER_JSON = '{"brand":"tele2","basis":"name_read","evidence":"","visible_text":"t2"}'


class _Response:
    ok = True

    @staticmethod
    def raise_for_status() -> None:
        return None

    @staticmethod
    def json() -> dict[str, object]:
        return {"choices": [{"message": {"content": ANSWER_JSON}}]}


def _frame(tmp_path: Path) -> Path:
    path = tmp_path / "crop.jpg"
    cv2.imwrite(str(path), np.full((64, 64, 3), 200, dtype=np.uint8))
    return path


def test_own_server_gets_neither_model_name_nor_key(tmp_path: Path, monkeypatch) -> None:
    """llama-server отдаёт единственную загруженную модель и ключа не спрашивает."""

    sent: dict[str, object] = {}

    def post(url: str, **kwargs: object) -> _Response:
        sent.update(kwargs, url=url)
        return _Response()

    monkeypatch.setattr(vlm.requests, "post", post)
    vlm.ask(_frame(tmp_path), url="http://127.0.0.1:8080", model="", api_key="")

    assert "model" not in sent["json"]  # type: ignore[operator]
    assert sent["headers"] == {}


def test_shared_server_gets_model_name_and_key(tmp_path: Path, monkeypatch) -> None:
    """vLLM считает имя модели обязательным и закрыт ключом на прокси."""

    sent: dict[str, object] = {}

    def post(url: str, **kwargs: object) -> _Response:
        sent.update(kwargs, url=url)
        return _Response()

    monkeypatch.setattr(vlm.requests, "post", post)
    vlm.ask(_frame(tmp_path), url="http://10.0.0.1:8000", model="qwen", api_key="secret")

    assert sent["json"]["model"] == "qwen"  # type: ignore[index]
    assert sent["headers"] == {"Authorization": "Bearer secret"}


def test_health_probes_models_when_a_key_is_set(monkeypatch) -> None:
    """С ключом мало знать, что сервер жив: неверный ключ виден только на /v1/models.

    Иначе прогон стартует, а потом каждый кадр получает 401 и уходит в сбой.
    """

    seen: list[str] = []

    def get(url: str, **_: object) -> _Response:
        seen.append(url)
        return _Response()

    monkeypatch.setattr(vlm.requests, "get", get)

    assert vlm.health("http://srv:8000", api_key="secret")
    assert vlm.health("http://127.0.0.1:8080", api_key="")
    assert seen == ["http://srv:8000/v1/models", "http://127.0.0.1:8080/health"]
