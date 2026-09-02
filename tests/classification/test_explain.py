from __future__ import annotations

import numpy as np

from adlearn.classification.explain import activation_map, overlay


def test_map_is_bounded_after_stretching() -> None:
    """Кубическая интерполяция вылетает за диапазон — наложение бы поехало."""

    rng = np.random.default_rng(0)
    features = rng.normal(size=(8, 7, 7)).astype(np.float32)
    weights = rng.normal(size=8).astype(np.float32)

    heatmap = activation_map(features, weights, side=64)

    assert heatmap.shape == (64, 64)
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0


def test_overlay_keeps_the_frame_where_attention_is_zero() -> None:
    frame = np.full((32, 32, 3), 120, dtype=np.uint8)

    untouched = overlay(frame, np.zeros((32, 32), dtype=np.float32))

    assert np.array_equal(untouched, frame)
