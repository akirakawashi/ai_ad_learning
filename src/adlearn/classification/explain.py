"""Что модель увидела на кадре: вероятности, карта внимания, разбор цвета.

Карта внимания получается почти даром. Голова у нас линейная и стоит поверх
признаков, усреднённых по кадру, — а это ровно та схема, для которой придуман
Class Activation Mapping. Вклад участка кадра в оценку класса считается теми же
весами головы, только без усреднения по пространству.

Поэтому карта показывает не «примерно куда», а буквально те места, из-за которых
модель поставила такую оценку.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50

from adlearn.classification.features.color import read_image
from adlearn.classification.features.visual import MEAN, SIDE, STD, letterbox

VISUAL_BLOCK = "visual"


@dataclass(frozen=True)
class Explanation:
    """Разбор одного кадра."""

    path: Path
    frame: np.ndarray
    """Кадр в том виде, в каком его получила сеть: вписан в квадрат, RGB."""

    probabilities: np.ndarray
    heatmaps: dict[str, np.ndarray]
    """Карта внимания на класс, в тех же координатах, что и `frame`."""

    color: dict[str, float]

    @property
    def answer(self) -> tuple[str, float]:
        best = int(np.argmax(self.probabilities))
        return self.brands[best], float(self.probabilities[best])

    brands: tuple[str, ...] = ()


class Backbone:
    """Тот же ResNet50, но отдаёт признаки до усреднения — сеткой 7×7."""

    def __init__(self, device: str = "cuda") -> None:
        self.device = device if torch.cuda.is_available() else "cpu"
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.stem = nn.Sequential(*list(model.children())[:-2]).eval().to(self.device)

    def prepare(self, path: Path) -> tuple[np.ndarray, torch.Tensor]:
        image = read_image(path)
        if image is None:
            image = np.full((SIDE, SIDE, 3), 255, dtype=np.uint8)
        boxed = letterbox(image, SIDE)
        rgb = boxed[:, :, ::-1].copy()
        tensor = torch.from_numpy(rgb).float().div(255.0).permute(2, 0, 1)[None]
        mean = torch.tensor(MEAN).view(1, 3, 1, 1)
        std = torch.tensor(STD).view(1, 3, 1, 1)
        return rgb, (tensor - mean) / std

    @torch.inference_mode()
    def spatial(self, path: Path) -> tuple[np.ndarray, np.ndarray]:
        """Кадр и его признаки формой (2048, 7, 7)."""

        rgb, tensor = self.prepare(path)
        features = self.stem(tensor.to(self.device))[0].float().cpu().numpy()
        return rgb, features


def visual_weights(bundle: dict[str, Any], class_index: int) -> np.ndarray:
    """Веса головы по визуальному блоку, приведённые к исходным признакам.

    Голова обучалась на стандартизованных значениях, поэтому вес каждого признака
    делится на его разброс — иначе карта отразила бы масштаб, а не важность.
    """

    order = list(bundle["scaler"]._scalers)
    start = sum(len(bundle["dims"][name]) for name in order[: order.index(VISUAL_BLOCK)])
    size = len(bundle["dims"][VISUAL_BLOCK])
    weights = bundle["model"].coef_[class_index][start : start + size]
    return weights / bundle["scaler"]._scalers[VISUAL_BLOCK].scale_


def activation_map(features: np.ndarray, weights: np.ndarray, side: int = SIDE) -> np.ndarray:
    """Карта вклада участков кадра в оценку класса, приведённая к [0, 1].

    Отрицательный вклад срезается: интересно, что модель считает доводом *за*
    класс, а «здесь ничего похожего» видно и так.
    """

    raw = np.tensordot(weights, features, axes=([0], [0]))
    raw = np.maximum(raw, 0.0)
    peak = raw.max()
    if peak > 0:
        raw = raw / peak
    # Кубическая интерполяция вылетает за границы диапазона, поэтому подрезаем
    # после растяжения, а не до: иначе наложение уходит в отрицательный вес.
    smooth = cv2.resize(raw.astype(np.float32), (side, side), interpolation=cv2.INTER_CUBIC)
    return np.clip(smooth, 0.0, 1.0)


def overlay(frame: np.ndarray, heatmap: np.ndarray, strength: float = 0.55) -> np.ndarray:
    """Кадр с наложенной картой внимания."""

    colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    weight = (heatmap[:, :, None] * strength).astype(np.float32)
    return (frame * (1 - weight) + colored * weight).astype(np.uint8)


def explain(
    paths: Sequence[Path],
    bundle: dict[str, Any],
    *,
    device: str = "cuda",
    backbone: Backbone | None = None,
) -> list[Explanation]:
    """Считает вероятности, карты внимания и цветовые признаки для кадров."""

    from adlearn.classification.features import ColorExtractor, VisualExtractor

    engine = backbone or Backbone(device=device)
    brands = tuple(bundle["brands"])
    color_dims = bundle["dims"]["color"]

    visual = VisualExtractor(device=device)(paths)
    color = ColorExtractor()(paths)
    probabilities = bundle["model"].predict_proba(
        bundle["scaler"].transform({"visual": visual, "color": color})
    )

    results: list[Explanation] = []
    for index, path in enumerate(paths):
        frame, features = engine.spatial(path)
        heatmaps = {
            brand: activation_map(features, visual_weights(bundle, position))
            for position, brand in enumerate(brands)
        }
        results.append(
            Explanation(
                path=path,
                frame=frame,
                probabilities=probabilities[index],
                heatmaps=heatmaps,
                color=dict(zip(color_dims, color[index].tolist(), strict=True)),
                brands=brands,
            )
        )
    return results
