"""Разметка поверх кадров: одиночные картинки и контактные листы."""

from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np

from prelabel_tool.config import IMAGE_SUFFIXES

BOX_COLOR = (0, 0, 255)
GRID_COLOR = (255, 255, 255)


def read_boxes(label: Path) -> list[tuple[float, float, float, float]]:
    if not label.exists():
        return []
    boxes = []
    for line in label.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        _, center_x, center_y, width, height = (float(value) for value in parts)
        boxes.append((center_x, center_y, width, height))
    return boxes


def draw(image: np.ndarray, boxes: list[tuple[float, float, float, float]]) -> np.ndarray:
    """Рисует рамки в долях кадра. Толщина линии зависит от размера кадра."""

    height, width = image.shape[:2]
    thickness = max(2, round(min(width, height) / 250))
    for center_x, center_y, box_width, box_height in boxes:
        x1 = round((center_x - box_width / 2) * width)
        y1 = round((center_y - box_height / 2) * height)
        x2 = round((center_x + box_width / 2) * width)
        y2 = round((center_y + box_height / 2) * height)
        cv2.rectangle(image, (x1, y1), (x2, y2), BOX_COLOR, thickness)
    return image


def tile(image: np.ndarray, side: int) -> np.ndarray:
    """Вписывает кадр в квадрат, не растягивая: поля добеливаются."""

    height, width = image.shape[:2]
    scale = side / max(height, width)
    resized = cv2.resize(image, (max(1, round(width * scale)), max(1, round(height * scale))))
    canvas = np.full((side, side, 3), 255, dtype=np.uint8)
    top = (side - resized.shape[0]) // 2
    left = (side - resized.shape[1]) // 2
    canvas[top : top + resized.shape[0], left : left + resized.shape[1]] = resized
    return canvas


def build_sheets(
    *,
    images_dir: Path,
    labels_dir: Path,
    output: Path,
    limit: int,
    columns: int = 4,
    rows: int = 3,
    side: int = 420,
    seed: int = 0,
    only_empty: bool = False,
    save_frames: bool = False,
) -> tuple[int, int]:
    """Собирает контактные листы: кадр с рамками, по `columns × rows` на лист.

    Смотреть семь тысяч кадров по одному невозможно, а лист из дюжины
    просматривается за секунды — и промах видно сразу.
    """

    sheets_dir = output / "sheets"
    frames_dir = output / "frames"
    sheets_dir.mkdir(parents=True, exist_ok=True)
    if save_frames:
        frames_dir.mkdir(parents=True, exist_ok=True)
    images = sorted(
        path for path in images_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
    )
    if only_empty:
        images = [path for path in images if not read_boxes(labels_dir / f"{path.stem}.txt")]
    random.Random(seed).shuffle(images)
    images = images[:limit]

    tiles: list[np.ndarray] = []
    for path in images:
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        marked = draw(frame, read_boxes(labels_dir / f"{path.stem}.txt"))
        if save_frames:
            cv2.imwrite(str(frames_dir / f"{path.stem}.jpg"), marked)
        tiles.append(tile(marked, side))

    per_sheet = columns * rows
    sheets = 0
    for start in range(0, len(tiles), per_sheet):
        chunk = tiles[start : start + per_sheet]
        while len(chunk) < per_sheet:
            chunk.append(np.full((side, side, 3), 255, dtype=np.uint8))
        grid = np.vstack(
            [np.hstack(chunk[row * columns : (row + 1) * columns]) for row in range(rows)]
        )
        sheets += 1
        cv2.imwrite(str(sheets_dir / f"sheet_{sheets:02d}.jpg"), grid)
    return len(tiles), sheets
