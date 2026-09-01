"""Проверка собранного набора перед обучением."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from training.checks import SPLITS, image_paths, inspect

logger = logging.getLogger("check")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check the built dataset.")
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--skip-decode", action="store_true")
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    findings, boxes = inspect(args.dataset, read_images=not args.skip_decode)

    for split in SPLITS:
        frames = len(image_paths(args.dataset, split))
        per_frame = boxes[split] / frames if frames else 0.0
        logger.info(
            "%s: кадров %s, рамок %s, в среднем %.2f на кадр",
            split,
            frames,
            boxes[split],
            per_frame,
        )

    problems = {
        "нет разметки": findings.missing_label,
        "нет кадра": findings.missing_image,
        "битая ссылка": findings.broken_link,
        "кадр не читается": findings.unreadable,
        "строка разметки испорчена": findings.bad_line,
        "координаты вне кадра": findings.out_of_range,
        "вырожденная рамка": findings.degenerate,
        "имя в двух частях": findings.name_in_two_splits,
        "одно фото в двух частях": findings.same_photo_in_two_splits,
    }
    for title, items in problems.items():
        if items:
            logger.info("%s: %s — %s", title, len(items), ", ".join(items[:5]))

    logger.info("итог: %s", "чисто" if findings.clean else "есть замечания")
    return 0 if findings.clean else 1


if __name__ == "__main__":
    raise SystemExit(main())
