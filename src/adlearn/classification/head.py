"""Голова классификатора и склейка блоков признаков.

Блоки склеиваются не встык. Визуальный эмбеддинг — это 2048 чисел своего
масштаба, цветовой блок — полсотни в долях единицы. Если подать их как есть,
эмбеддинг задавит цвет одной только размерностью, и арма «визуал + цвет» не
обгонит «только визуал» даже там, где цвет помогает. Поэтому каждый блок
приводится к своему масштабу отдельно, а на цветовой вешается вес.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


@dataclass
class BlockScaler:
    """Стандартизует и нормирует каждый блок признаков по отдельности."""

    weights: dict[str, float] = field(default_factory=dict)
    _scalers: dict[str, StandardScaler] = field(default_factory=dict, init=False)

    def fit(self, blocks: dict[str, np.ndarray]) -> BlockScaler:
        for name, values in blocks.items():
            self._scalers[name] = StandardScaler().fit(values)
        return self

    def transform(self, blocks: dict[str, np.ndarray]) -> np.ndarray:
        parts = []
        for name, values in blocks.items():
            scaled = self._scalers[name].transform(values)
            norm = np.linalg.norm(scaled, axis=1, keepdims=True)
            parts.append(scaled / (norm + 1e-9) * self.weights.get(name, 1.0))
        return np.concatenate(parts, axis=1)

    def fit_transform(self, blocks: dict[str, np.ndarray]) -> np.ndarray:
        return self.fit(blocks).transform(blocks)


def make_head(*, regularization: float = 1.0, seed: int = 0) -> LogisticRegression:
    """Многоклассовая логистическая регрессия.

    Не нейросеть: на полутора тысячах кадров у неё меньше гиперпараметров и
    меньше разброса, и сравнение арм получается про признаки, а не про удачу
    при обучении. `balanced` выравнивает вклад классов, чтобы macro F1 не
    перекосило в сторону самого крупного.
    """

    return LogisticRegression(
        C=regularization,
        max_iter=3000,
        class_weight="balanced",
        random_state=seed,
    )
