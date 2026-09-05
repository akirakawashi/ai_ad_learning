"""Фотографии без рекламы: раскладка с пустой разметкой и удвоение копиями."""

from __future__ import annotations

import random
from pathlib import Path

from PIL import Image

from adlearn.detection import negatives


def test_every_photo_gets_a_twin_and_empty_labels(tmp_path: Path):
    source = tmp_path / "src"
    source.mkdir()
    for name in ("a.jpg", "b.png", "skip.jpg"):
        Image.new("RGB", (120, 80), "gray").save(source / name)
    config = negatives.NegativesConfig(source=source, output=tmp_path / "out", skip=("skip.jpg",))

    counts = negatives.prepare(config)

    assert counts == (2, 2, 1)
    names = sorted(path.name for path in config.images_dir.iterdir())
    assert names == [
        "truck_00001.jpg",
        "truck_00001_aug.jpg",
        "truck_00002.jpg",
        "truck_00002_aug.jpg",
    ]
    assert all(not path.read_text() for path in config.labels_dir.iterdir())
    assert len(list(config.labels_dir.iterdir())) == 4


def test_augment_keeps_most_of_the_frame(tmp_path: Path):
    config = negatives.NegativesConfig(source=tmp_path, output=tmp_path)
    image = Image.new("RGB", (200, 100), "white")

    copy = negatives.augment(image, config=config, rng=random.Random(1))

    assert copy.width >= 200 * config.crop_min_share - 1
    assert copy.height >= 100 * config.crop_min_share - 1
