"""Сборка набора для дообучения: ручная разметка плюс уверенная псевдоразметка."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from prelabel_tool.config import PrelabelConfig
from training.dataset import TEST_SHARE, VALIDATION_SHARE, build

logger = logging.getLogger("dataset")


def parse_args() -> argparse.Namespace:
    defaults = PrelabelConfig()
    parser = argparse.ArgumentParser(description="Build the fine-tuning dataset.")
    parser.add_argument("--export", required=True, type=Path)
    parser.add_argument("--prelabel", type=Path, default=defaults.output)
    parser.add_argument("--output", type=Path, default=Path("dataset"))
    parser.add_argument("--class-name", type=str, default=defaults.class_name)
    parser.add_argument("--val-share", type=float, default=VALIDATION_SHARE)
    parser.add_argument("--test-share", type=float, default=TEST_SHARE)
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    counts = build(
        export_archive=args.export,
        prelabel_dir=args.prelabel,
        output=args.output,
        class_name=args.class_name,
        validation_share=args.val_share,
        test_share=args.test_share,
    )
    for title, part in (
        ("обучение", counts.train),
        ("проверка", counts.validation),
        ("тест", counts.test),
        ("тест, только ручная разметка", counts.test_handmade),
    ):
        logger.info(
            "%s: %s кадров, из них без рамок %s (%.1f%%)",
            title,
            part.frames,
            part.empty,
            part.empty_share * 100,
        )
    logger.info("описание набора: %s", args.output / "data.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
