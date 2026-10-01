"""Абляция: помогают ли цветовые признаки, и помогает ли именно знание брендов."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold

from adlearn.classification.config import BRANDS
from adlearn.classification.head import BlockScaler, make_head

ARMS: dict[str, tuple[str, ...]] = {
    "визуал": ("visual",),
    "цвет": ("color",),
    "визуал + цвет": ("visual", "color"),
}

SHORTCUT_ARM = "ПОДСКАЗКА (размер/резкость)"
"""Диагностическая арма — см. `features.shortcut`.

Она не про бренды: если голова на одних только размерах и резкости различает
классы, значит классы отличаются источником кадров, а не содержанием, и остальные
цифры завышены на ту же величину.
"""

GENERIC_PREFIXES = (
    "share_",
    "peak_",
    "achromatic",
    "lightness_",
    "chroma",
    "colorfulness",
    "palette_entropy",
)
"""Цветовые признаки, не знающие ничего про бренды.

Контрольная арма на них отвечает на настоящий вопрос эксперимента. Если общая
гистограмма даёт тот же прирост, что и фирменные палитры, значит знание брендов
не добавило ничего — хватило просто цвета.
"""


@dataclass(frozen=True)
class ArmResult:
    name: str
    macro_f1: list[float]
    predictions: np.ndarray
    truth: np.ndarray

    @property
    def mean(self) -> float:
        """Возвращает средний macro F1 по всем фолдам и повторам.

        Returns:
            Среднее значение macro F1.
        """

        return float(np.mean(self.macro_f1))

    @property
    def spread(self) -> float:
        """Возвращает разброс macro F1 между фолдами и повторами.

        Returns:
            Стандартное отклонение macro F1.
        """

        return float(np.std(self.macro_f1))

    @property
    def matrix(self) -> np.ndarray:
        """Строит матрицу ошибок по предсказаниям первой итерации.

        Returns:
            Матрица ошибок со строками истинных классов и столбцами ответов.
        """

        return confusion_matrix(self.truth, self.predictions, labels=range(len(BRANDS)))


def generic_columns(dims: list[str]) -> list[int]:
    """Находит общие цветовые признаки, которые не используют палитры брендов.

    Args:
        dims: Имена измерений цветового блока.

    Returns:
        Индексы общих цветовых признаков.
    """

    return [i for i, name in enumerate(dims) if name.startswith(GENERIC_PREFIXES)]


def run_arm(
    *,
    name: str,
    blocks: dict[str, np.ndarray],
    y: np.ndarray,
    groups: np.ndarray,
    folds: int,
    repeats: int,
    color_weight: float,
    regularization: float,
    seed: int,
) -> ArmResult:
    """Прогон одной армы по повторной групповой кросс-валидации.

    Групповой фолд обязателен: почти-дубликаты из интернета иначе разъедутся
    между обучением и тестом, и завышены окажутся все армы сразу — сравнивать
    станет нечего.

    Args:
        name: Название армы для отчёта.
        blocks: Блоки признаков выбранной армы.
        y: Индексы правильных классов.
        groups: Группы почти одинаковых кадров.
        folds: Число фолдов групповой кросс-валидации.
        repeats: Число повторов разбиения.
        color_weight: Вес нормированного цветового блока.
        regularization: Параметр `C` логистической регрессии.
        seed: Начальное значение генератора разбиений.

    Returns:
        Метрики армы и ответы первой итерации.
    """

    scores: list[float] = []
    predictions = np.zeros_like(y)
    for repeat in range(repeats):
        splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed + repeat)
        for train_index, test_index in splitter.split(np.zeros(len(y)), y, groups):
            scaler = BlockScaler(weights={"color": color_weight})
            train_blocks = {k: v[train_index] for k, v in blocks.items()}
            test_blocks = {k: v[test_index] for k, v in blocks.items()}
            model = make_head(regularization=regularization, seed=seed)
            model.fit(scaler.fit_transform(train_blocks), y[train_index])
            guess = model.predict(scaler.transform(test_blocks))
            scores.append(float(f1_score(y[test_index], guess, average="macro")))
            if repeat == 0:
                predictions[test_index] = guess
    return ArmResult(name=name, macro_f1=scores, predictions=predictions, truth=y)


def build_blocks(
    features: dict[str, np.ndarray],
    dims: dict[str, list[str]],
    arm: tuple[str, ...],
    *,
    generic_only: bool = False,
) -> dict[str, np.ndarray]:
    """Отбирает блоки признаков для одной армы.

    Args:
        features: Все рассчитанные матрицы признаков.
        dims: Имена измерений каждого блока.
        arm: Имена блоков, входящих в арму.
        generic_only: Оставить ли в цветовом блоке только общие признаки.

    Returns:
        Выбранные матрицы признаков по именам блоков.
    """

    blocks: dict[str, np.ndarray] = {}
    for name in arm:
        values = features[name]
        if name == "color" and generic_only:
            values = values[:, generic_columns(dims[name])]
        blocks[name] = values
    return blocks
