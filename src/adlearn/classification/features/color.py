"""Цветовые признаки: доли, палитры брендов и соседство цветов.

Модуль намеренно ничего не решает про бренд. Он описывает кадр числами, а вывод
делает голова классификатора: `beeline_score` высокий у чёрно-жёлтой картинки —
это наблюдение, а не ответ. Чёрный высокий и у Tele2, и у Билайна, и так и должно
быть.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from adlearn.classification.features.base import register
from adlearn.classification.palettes import (
    ANCHOR_NAMES,
    BRAND_PAIRS,
    BRAND_PROTOTYPES,
    BRAND_REQUIRED,
    KERNEL_SIGMA,
    anchors_lab,
)

WORK_SIDE = 160
COARSE_GRID = 4
FINE_GRID = 8
ACHROMATIC_CHROMA = 12.0


def read_image(path: Path) -> np.ndarray | None:
    """Кадр в BGR. Прозрачный фон подкладывается белым, а не чёрным.

    Логотипы часто приходят PNG с альфой, и `IMREAD_COLOR` молча отдал бы то, что
    лежит под прозрачностью — обычно чёрный. Тогда логотип на «белом» фоне
    получил бы огромную долю чёрного, и вся палитра поехала бы.
    """

    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return None
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        rgb = image[:, :, :3].astype(np.float32)
        alpha = image[:, :, 3:4].astype(np.float32) / 255.0
        return (rgb * alpha + 255.0 * (1.0 - alpha)).astype(np.uint8)
    return image[:, :, :3]


def white_balance(image: np.ndarray, power: int = 6) -> np.ndarray:
    """Shades-of-Gray: снимает цветовую температуру освещения.

    Без этого шага тёплый вечерний свет красит белую панель в жёлтый, и Tele2
    начинает давать ложный сигнал Билайна. Это самая полезная правка из всех:
    разный баланс белого искажает цвет сильнее, чем размытие и JPEG вместе.
    """

    data = image.astype(np.float32) + 1.0
    illuminant = np.power(np.power(data, power).mean(axis=(0, 1)), 1.0 / power)
    scale = illuminant.mean() / illuminant
    return np.clip(data * scale, 0, 255).astype(np.uint8)


def stretch_lightness(lab: np.ndarray) -> np.ndarray:
    """Растягивает светлоту по перцентилям кадра.

    Абсолютная светлота ненадёжна: один и тот же щит снят и в тень, и против
    солнца. Надёжно относительное — «самое тёмное в этом кадре». Растяжка по
    2-му и 98-му перцентилю приводит затемнённые и пересвеченные кадры к общей
    шкале, сохраняя контраст внутри кадра.
    """

    lightness = lab[:, :, 0]
    low, high = np.percentile(lightness, (2.0, 98.0))
    if high - low < 1.0:
        return lab
    out = lab.copy()
    out[:, :, 0] = np.clip((lightness - low) / (high - low) * 90.0 + 5.0, 0.0, 100.0)
    return out


def to_lab(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[:, :, 0] *= 100.0 / 255.0
    lab[:, :, 1:] -= 128.0
    return lab


def prepare(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Готовит кадр к замеру цвета и отдаёт (Lab растянутый, Lab исходный)."""

    height, width = image.shape[:2]
    scale = WORK_SIDE / max(height, width)
    if scale < 1.0:
        image = cv2.resize(
            image,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    image = cv2.bilateralFilter(image, d=5, sigmaColor=40, sigmaSpace=5)
    balanced = white_balance(image)
    lab = to_lab(balanced)
    return stretch_lightness(lab), lab


def membership(lab: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    """Мягкое отнесение каждого пикселя к якорям палитры.

    Мягкое, а не пороговое: под размытием и сдвигом баланса белого доля цвета
    меняется плавно, а не перескакивает, когда пиксель пересёк границу диапазона.
    """

    flat = lab.reshape(-1, 3)
    distance = np.linalg.norm(flat[:, None, :] - anchors[None, :, :], axis=2)
    weight = np.exp(-((distance / KERNEL_SIGMA) ** 2))
    weight /= weight.sum(axis=1, keepdims=True) + 1e-9
    return weight.reshape(*lab.shape[:2], len(anchors))


def pool_grid(values: np.ndarray, grid: int) -> np.ndarray:
    """Средняя принадлежность по ячейкам сетки `grid × grid`."""

    height, width, channels = values.shape
    rows = np.arange(height) * grid // height
    cols = np.arange(width) * grid // width
    index = (rows[:, None] * grid + cols[None, :]).ravel()
    totals = np.zeros((grid * grid, channels), dtype=np.float32)
    counts = np.zeros(grid * grid, dtype=np.float32)
    np.add.at(totals, index, values.reshape(-1, channels))
    np.add.at(counts, index, 1.0)
    return totals / np.maximum(counts, 1.0)[:, None]


def adjacency_share(cells: np.ndarray, grid: int, first: int, second: int) -> float:
    """Доля соседних ячеек, где встретилась эта пара цветов.

    Это и есть «фирменные цвета рядом друг с другом». Логотип — контрастное
    соседство, и оно разделяет бренды лучше, чем сами доли: чёрного много у всех.
    """

    dominant = cells.argmax(axis=1).reshape(grid, grid)
    hits = 0
    total = 0
    for axis in (0, 1):
        left = dominant if axis == 0 else dominant.T
        for row in left:
            for a, b in zip(row[:-1], row[1:], strict=True):
                total += 1
                if {int(a), int(b)} == {first, second}:
                    hits += 1
    return hits / total if total else 0.0


def colorfulness(image: np.ndarray) -> float:
    """Метрика Хаслера — Зюсструнка: насколько кадр вообще цветной."""

    blue, green, red = (channel.astype(np.float32) for channel in cv2.split(image))
    rg = red - green
    yb = 0.5 * (red + green) - blue
    root = np.sqrt(rg.std() ** 2 + yb.std() ** 2)
    mean = np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    return float((root + 0.3 * mean) / 100.0)


def feature_names() -> list[str]:
    names = [f"share_{name}" for name in ANCHOR_NAMES]
    names += [f"peak_{name}" for name in ANCHOR_NAMES]
    names += [
        "achromatic",
        "lightness_median",
        "lightness_spread",
        "chroma",
        "colorfulness",
        "palette_entropy",
    ]
    for brand in BRAND_PROTOTYPES:
        names += [
            f"{brand}_coverage",
            f"{brand}_coverage_peak",
            f"{brand}_balance",
            f"{brand}_similarity",
            f"{brand}_score",
        ]
    for brand in BRAND_PAIRS:
        names += [f"{brand}_adjacency_coarse", f"{brand}_adjacency_fine"]
    return names


def describe(path: Path, anchors: np.ndarray) -> np.ndarray:
    """Все цветовые признаки одного кадра."""

    image = read_image(path)
    if image is None:
        return np.zeros(len(feature_names()), dtype=np.float32)

    stretched, original = prepare(image)
    weights = membership(stretched, anchors)
    shares = weights.reshape(-1, weights.shape[2]).mean(axis=0)

    coarse = pool_grid(weights, COARSE_GRID)
    fine = pool_grid(weights, FINE_GRID)
    peaks = coarse.max(axis=0)

    chroma = np.linalg.norm(original[:, :, 1:], axis=2)
    lightness = original[:, :, 0]
    entropy = float(-(shares * np.log(shares + 1e-9)).sum() / np.log(len(shares)))

    values: list[float] = []
    values += shares.tolist()
    values += peaks.tolist()
    values += [
        float((chroma < ACHROMATIC_CHROMA).mean()),
        float(np.median(lightness) / 100.0),
        float(lightness.std() / 100.0),
        float(chroma.mean() / 100.0),
        colorfulness(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)),
        entropy,
    ]

    index = {name: position for position, name in enumerate(ANCHOR_NAMES)}
    for brand, prototype in BRAND_PROTOTYPES.items():
        columns = [index[name] for name in prototype]
        coverage = float(shares[columns].sum())
        coverage_peak = float(coarse[:, columns].sum(axis=1).max())
        required = [index[name] for name in BRAND_REQUIRED[brand]]
        # Геометрическое среднее требует одновременного присутствия: один чёрный
        # без жёлтого обнуляет балл Билайна, как и должно быть.
        balance = float(np.exp(np.log(peaks[required] + 1e-6).mean()))
        target = np.zeros(len(ANCHOR_NAMES), dtype=np.float32)
        for name, weight in prototype.items():
            target[index[name]] = weight
        similarity = float(
            shares @ target / (np.linalg.norm(shares) * np.linalg.norm(target) + 1e-9)
        )
        score = float(np.cbrt(max(coverage_peak, 0.0) * max(balance, 0.0) * max(similarity, 0.0)))
        values += [coverage, coverage_peak, balance, similarity, score]

    for pairs in BRAND_PAIRS.values():
        for cells, grid in ((coarse, COARSE_GRID), (fine, FINE_GRID)):
            values.append(
                max(
                    adjacency_share(cells, grid, index[first], index[second])
                    for first, second in pairs
                )
            )

    return np.asarray(values, dtype=np.float32)


class ColorExtractor:
    """Цветовые признаки: доли, пики по сетке, палитры брендов, соседство."""

    name = "color"
    version = "1"

    def __init__(self) -> None:
        self._anchors = anchors_lab()
        self._dims = feature_names()

    @property
    def dims(self) -> list[str]:
        return self._dims

    def __call__(self, paths: Sequence[Path]) -> np.ndarray:
        return np.stack([describe(path, self._anchors) for path in paths])


register("color")(ColorExtractor)
