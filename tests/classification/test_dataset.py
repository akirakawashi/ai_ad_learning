from __future__ import annotations

from pathlib import Path

import numpy as np

from adlearn.classification.dataset import collect, duplicate_groups, labels


def make_raw(tmp_path: Path, counts: dict[str, int]) -> Path:
    raw = tmp_path / "raw"
    for brand, count in counts.items():
        (raw / brand).mkdir(parents=True)
        for index in range(count):
            (raw / brand / f"{index:03d}.jpg").write_bytes(b"")
    return raw


def test_labels_follow_the_fixed_brand_order(tmp_path: Path) -> None:
    raw = make_raw(tmp_path, {"beeline": 2, "megafon": 1, "tele2": 1})
    samples = collect(raw)

    assert [item.brand for item in samples] == ["beeline", "beeline", "megafon", "tele2"]
    assert labels(samples).tolist() == [0, 0, 1, 2]


def test_near_duplicates_land_in_one_group() -> None:
    """Одна картинка в двух пережатиях не должна попасть в обучение и в тест сразу."""

    base = np.random.default_rng(0).normal(size=(3, 16)).astype(np.float32)
    embeddings = np.vstack([base, base[0:1] + 1e-4])

    groups = duplicate_groups(embeddings, threshold=0.97)

    assert groups[0] == groups[3]
    assert len({int(groups[1]), int(groups[2]), int(groups[0])}) == 3
