"""Цветовые якоря и фирменные палитры.

Якоря заданы в sRGB для читаемости и переводятся в Lab при загрузке. Значения
брендов — не догма, а точка старта: настоящие оттенки на улице и в печати уезжают,
поэтому якоря стоит уточнять кластеризацией по обучающей части.
"""

from __future__ import annotations

import cv2
import numpy as np

ANCHORS: dict[str, tuple[int, int, int]] = {
    "black": (20, 20, 20),
    "dark_gray": (70, 70, 70),
    "gray": (128, 128, 128),
    "light_gray": (190, 190, 190),
    "white": (245, 245, 245),
    "yellow": (255, 204, 0),
    "orange": (240, 130, 20),
    "red": (200, 30, 30),
    "magenta": (230, 0, 126),
    "purple": (115, 25, 130),
    "blue": (20, 50, 140),
    "cyan": (0, 174, 239),
    "green": (0, 185, 86),
    "beige": (200, 170, 140),
}

ANCHOR_NAMES = tuple(ANCHORS)

KERNEL_SIGMA = 15.0
"""Ширина ядра мягкого отнесения в единицах ΔE76.

Шире — соседние оттенки перетекают друг в друга (беж начинает съедать жёлтый),
уже — признак становится хрупким к сдвигу баланса белого.
"""


def anchors_lab() -> np.ndarray:
    """Якоря в Lab: L в [0,100], a и b вокруг нуля."""

    bgr = np.array([[ANCHORS[name][::-1] for name in ANCHOR_NAMES]], dtype=np.uint8)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[0].astype(np.float32)
    lab[:, 0] *= 100.0 / 255.0
    lab[:, 1:] -= 128.0
    return lab


BRAND_PROTOTYPES: dict[str, dict[str, float]] = {
    "beeline": {"yellow": 0.5, "black": 0.5},
    "megafon": {"green": 0.6, "purple": 0.2, "white": 0.2},
    "tele2": {"black": 0.45, "white": 0.35, "magenta": 0.20},
}
"""Ожидаемое распределение фирменных цветов.

У Tele2 намеренно учтены обе эпохи бренда: чёрно-белый TELE2 и новый `t2` с
маджентой. В наборе встречаются обе, и палитра только из чёрного с белым половину
кадров описывает неверно.
"""

BRAND_REQUIRED: dict[str, tuple[str, ...]] = {
    "beeline": ("yellow", "black"),
    "megafon": ("green",),
    "tele2": ("black", "white"),
}
"""Цвета, которые должны присутствовать одновременно.

Отсюда берётся признак баланса: один чёрный без жёлтого — это ещё не Билайн.
"""

BRAND_PAIRS: dict[str, tuple[tuple[str, str], ...]] = {
    "beeline": (("yellow", "black"),),
    "megafon": (("green", "white"), ("green", "purple")),
    "tele2": (("black", "white"), ("black", "magenta")),
}
"""Сочетания, которые должны встречаться рядом друг с другом.

Логотип и есть высококонтрастное соседство фирменных цветов, поэтому соседство
разделяет бренды лучше, чем сами доли: чёрный есть у всех трёх.
"""
