"""Проверки собранного набора: пары, разметка, пересечения и читаемость кадров."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2

SPLITS = ("train", "val", "test")


@dataclass
class Findings:
    """Что нашлось. Пустой список означает, что проверка прошла."""

    missing_label: list[str] = field(default_factory=list)
    missing_image: list[str] = field(default_factory=list)
    broken_link: list[str] = field(default_factory=list)
    unreadable: list[str] = field(default_factory=list)
    bad_line: list[str] = field(default_factory=list)
    out_of_range: list[str] = field(default_factory=list)
    degenerate: list[str] = field(default_factory=list)
    name_in_two_splits: list[str] = field(default_factory=list)
    same_photo_in_two_splits: list[str] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not any(vars(self).values())


def image_paths(root: Path, split: str) -> list[Path]:
    directory = root / "images" / split
    return sorted(directory.iterdir()) if directory.is_dir() else []


def check_label(text: str, *, source: str, findings: Findings) -> None:
    """Разметка YOLO: класс и четыре доли кадра, все внутри границ."""

    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            findings.bad_line.append(f"{source}:{number}")
            continue
        try:
            class_id = int(parts[0])
            values = [float(value) for value in parts[1:]]
        except ValueError:
            findings.bad_line.append(f"{source}:{number}")
            continue
        if class_id != 0:
            findings.bad_line.append(f"{source}:{number} класс {class_id}")
        if any(value < 0.0 or value > 1.0 for value in values):
            findings.out_of_range.append(f"{source}:{number}")
        width, height = values[2], values[3]
        if width <= 0.0 or height <= 0.0 or width * height < 1e-6:
            findings.degenerate.append(f"{source}:{number}")


def inspect(root: Path, *, read_images: bool = True) -> tuple[Findings, dict[str, int]]:
    findings = Findings()
    digests: dict[str, list[str]] = defaultdict(list)
    names: dict[str, list[str]] = defaultdict(list)
    boxes_per_split: dict[str, int] = {}

    for split in SPLITS:
        boxes = 0
        for image in image_paths(root, split):
            source = f"{split}/{image.name}"
            names[image.name].append(split)
            if not image.exists():
                findings.broken_link.append(source)
                continue

            label = root / "labels" / split / f"{image.stem}.txt"
            if not label.exists():
                findings.missing_label.append(source)
            else:
                text = label.read_text(encoding="utf-8")
                boxes += sum(1 for line in text.splitlines() if line.strip())
                check_label(text, source=source, findings=findings)

            payload = image.read_bytes()
            digests[hashlib.sha1(payload).hexdigest()].append(source)
            if read_images and cv2.imread(str(image)) is None:
                findings.unreadable.append(source)

        for label in (root / "labels" / split).iterdir():
            stem = label.stem
            if not any(
                (root / "images" / split / f"{stem}{suffix}").exists()
                for suffix in (".jpg", ".jpeg", ".png", ".webp", ".JPG", ".PNG")
            ):
                findings.missing_image.append(f"{split}/{label.name}")
        boxes_per_split[split] = boxes

    findings.name_in_two_splits = [
        name for name, where in names.items() if len(set(where)) > 1
    ]
    findings.same_photo_in_two_splits = [
        ", ".join(sorted(where))
        for where in digests.values()
        if len({item.split("/")[0] for item in where}) > 1
    ]
    return findings, boxes_per_split
