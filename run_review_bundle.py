"""Сборка пачек для ручной правки в CVAT."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from prelabel_tool.bundle import CHUNK_SIZE, build
from prelabel_tool.config import PrelabelConfig

logger = logging.getLogger("bundle")


def parse_args() -> argparse.Namespace:
    defaults = PrelabelConfig()
    parser = argparse.ArgumentParser(description="Pack review chunks for CVAT.")
    parser.add_argument("--prelabel", type=Path, default=defaults.output)
    parser.add_argument("--output", type=Path, default=defaults.output / "review")
    parser.add_argument("--chunk", type=int, default=CHUNK_SIZE)
    parser.add_argument("--class-name", type=str, default=defaults.class_name)
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    parts = build(
        report_path=args.prelabel / "report.csv",
        images_dir=args.prelabel / "images",
        labels_dir=args.prelabel / "labels",
        output=args.output,
        class_name=args.class_name,
        chunk_size=args.chunk,
    )
    for images_archive, annotations_archive, count in parts:
        size_mb = images_archive.stat().st_size / 1e6
        logger.info(
            "%s — кадров %s, %.0f МБ, разметка %s",
            images_archive.name,
            count,
            size_mb,
            annotations_archive.name,
        )
    logger.info("пачек %s, всё в %s", len(parts), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
