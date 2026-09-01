"""Просмотр разметки: контактные листы с рамками поверх кадров."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from training.preview import build_sheets

logger = logging.getLogger("preview")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw labels over frames as contact sheets.")
    parser.add_argument("--images", type=Path, default=Path("dataset/images/train"))
    parser.add_argument("--labels", type=Path, default=Path("dataset/labels/train"))
    parser.add_argument("--output", type=Path, default=Path("preview"))
    parser.add_argument("--limit", type=int, default=120)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--rows", type=int, default=3)
    parser.add_argument("--side", type=int, default=420)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--only-empty", action="store_true")
    parser.add_argument("--frames", action="store_true")
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    output = args.output / args.images.name
    frames, sheets = build_sheets(
        images_dir=args.images,
        labels_dir=args.labels,
        output=output,
        limit=args.limit,
        columns=args.columns,
        rows=args.rows,
        side=args.side,
        seed=args.seed,
        only_empty=args.only_empty,
        save_frames=args.frames,
    )
    logger.info("кадров %s, листов %s, всё в %s", frames, sheets, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
