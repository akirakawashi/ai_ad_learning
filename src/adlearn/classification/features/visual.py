"""Визуальный эмбеддинг: замороженный энкодер, обученный не нами.

Сеть уже обучена на миллионах картинок и наших меток не видит — она просто
описывает кадр столбиком чисел. Мы её не дообучаем: 25 миллионов параметров на
1700 фотографий модель не выучит, а запомнит.

Учится дальше только лёгкая голова поверх этих чисел. Заодно эмбеддинги считаются
один раз и кладутся в кэш, поэтому армы абляции обучаются за секунды и можно
позволить себе повторную кросс-валидацию вместо одной случайной цифры.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50

from adlearn.classification.features.base import register
from adlearn.classification.features.color import read_image

SIDE = 224
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def letterbox(image: np.ndarray, side: int) -> np.ndarray:
    """Вписывает кадр в квадрат с белыми полями, не растягивая.

    Растяжение под квадрат ломает пропорции логотипа, а он у брендов узнаваем
    именно формой. Поля белые, потому что чёрные читались бы как фирменный цвет.
    """

    import cv2

    height, width = image.shape[:2]
    scale = side / max(height, width)
    resized = cv2.resize(
        image,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    canvas = np.full((side, side, 3), 255, dtype=np.uint8)
    top = (side - resized.shape[0]) // 2
    left = (side - resized.shape[1]) // 2
    canvas[top : top + resized.shape[0], left : left + resized.shape[1]] = resized
    return canvas


class VisualExtractor:
    """Признаки предпоследнего слоя ResNet50 — 2048 чисел на кадр."""

    name = "visual"
    version = "resnet50-1"

    def __init__(self, device: str = "cuda", batch_size: int = 32) -> None:
        self.device = device if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size
        self._model: nn.Module | None = None

    @property
    def dims(self) -> list[str]:
        return [f"visual_{index:04d}" for index in range(2048)]

    def _load(self) -> nn.Module:
        if self._model is None:
            model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            model.fc = nn.Identity()
            self._model = model.eval().to(self.device)
        return self._model

    def _batch(self, paths: Sequence[Path]) -> torch.Tensor:
        frames = []
        for path in paths:
            image = read_image(path)
            if image is None:
                image = np.full((SIDE, SIDE, 3), 255, dtype=np.uint8)
            frames.append(letterbox(image, SIDE)[:, :, ::-1])
        batch = torch.from_numpy(np.ascontiguousarray(np.stack(frames))).float() / 255.0
        batch = batch.permute(0, 3, 1, 2)
        mean = torch.tensor(MEAN).view(1, 3, 1, 1)
        std = torch.tensor(STD).view(1, 3, 1, 1)
        return (batch - mean) / std

    @torch.inference_mode()
    def __call__(self, paths: Sequence[Path]) -> np.ndarray:
        model = self._load()
        chunks = []
        for start in range(0, len(paths), self.batch_size):
            batch = self._batch(paths[start : start + self.batch_size]).to(self.device)
            chunks.append(model(batch).float().cpu().numpy())
        return np.concatenate(chunks) if chunks else np.zeros((0, 2048), dtype=np.float32)


register("visual")(VisualExtractor)
