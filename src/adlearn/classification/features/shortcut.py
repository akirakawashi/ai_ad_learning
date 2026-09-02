"""Признаки-подсказки: то, чем классы отличаются помимо содержания.

Это диагностика, а не полезный сигнал. Здесь собраны величины, которые не знают
ничего о бренде — размер кадра, резкость, соотношение сторон, контраст. Если
голова, обученная **только** на них, уверенно различает классы, значит классы
отличаются источником, а не содержанием.

Так бывает, когда позитивы набраны из интернета, а негативы нарезаны с улицы:
чистая картинка против смазанной. Модель находит эту разницу раньше, чем
логотипы, показывает прекрасную метрику на проверке и разваливается в бою.

Ставить эту арму в абляцию обязательно, пока набор собран из разных источников.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from adlearn.classification.features.base import register
from adlearn.classification.features.color import read_image

DIMS = [
    "log_width",
    "log_height",
    "aspect",
    "log_pixels",
    "sharpness",
]
"""Только то, как кадр получен и сохранён — ничего о его содержании.

Яркость, контраст и насыщенность сюда намеренно не входят, хотя и выдают
источник: они заодно несут и сам бренд (Билайн ярче Tele2). Диагностика,
подмешавшая содержание, показала бы утечку там, где её нет.
"""


def describe(path: Path) -> np.ndarray:
    image = read_image(path)
    if image is None:
        return np.zeros(len(DIMS), dtype=np.float32)
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)
    return np.asarray(
        [
            math.log10(width),
            math.log10(height),
            width / height,
            math.log10(width * height),
            math.log10(cv2.Laplacian(small, cv2.CV_64F).var() + 1e-6),
        ],
        dtype=np.float32,
    )


class ShortcutExtractor:
    """Размер, пропорции и резкость — всё, что выдаёт источник кадра."""

    name = "shortcut"
    version = "1"

    @property
    def dims(self) -> list[str]:
        return list(DIMS)

    def __call__(self, paths: Sequence[Path]) -> np.ndarray:
        return np.stack([describe(path) for path in paths])


register("shortcut")(ShortcutExtractor)
