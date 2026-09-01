from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from adlearn.classification.features.color import (
    ColorExtractor,
    feature_names,
    read_image,
    white_balance,
)


def write(path: Path, image: np.ndarray) -> Path:
    cv2.imwrite(str(path), image)
    return path


def flat(color: tuple[int, int, int], side: int = 64) -> np.ndarray:
    return np.full((side, side, 3), color, dtype=np.uint8)


def test_vector_length_matches_declared_dims(tmp_path: Path) -> None:
    path = write(tmp_path / "a.jpg", flat((0, 200, 255)))
    assert ColorExtractor()([path]).shape == (1, len(feature_names()))


def test_transparent_logo_lands_on_white_not_black(tmp_path: Path) -> None:
    """PNG с альфой иначе отдал бы чёрный фон и перекосил всю палитру."""

    image = np.zeros((32, 32, 4), dtype=np.uint8)
    image[:, :, 3] = 0
    path = tmp_path / "logo.png"
    cv2.imwrite(str(path), image)

    restored = read_image(path)
    assert restored is not None
    assert restored.mean() > 250


def test_white_balance_removes_a_colour_cast() -> None:
    """Тёплый свет не должен превращать белую панель в жёлтую."""

    warm = flat((150, 220, 255))
    balanced = white_balance(warm).astype(float)

    assert balanced.std(axis=(0, 1)).mean() < 1.0
    assert balanced[:, :, 0].mean() - balanced[:, :, 2].mean() > -40


def test_yellow_and_black_score_higher_for_beeline_than_for_megafon(tmp_path: Path) -> None:
    """Цветовой модуль не решает за классификатор, но сигнал давать обязан."""

    image = flat((255, 255, 255))
    image[:32] = (0, 204, 255)
    image[32:] = (20, 20, 20)
    path = write(tmp_path / "b.jpg", image)

    names = feature_names()
    values = ColorExtractor()([path])[0]
    beeline = values[names.index("beeline_score")]
    megafon = values[names.index("megafon_score")]

    assert beeline > megafon
