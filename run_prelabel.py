"""Псевдоразметка щитов: фотографии на входе, разметка и отчёт на выходе."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from prelabel_tool.config import PrelabelConfig
from prelabel_tool.report import STATUS_EMPTY, STATUS_OK, STATUS_WEAK, summary
from prelabel_tool.runner import run

logger = logging.getLogger("prelabel")


def parse_args() -> argparse.Namespace:
    defaults = PrelabelConfig()
    parser = argparse.ArgumentParser(description="Pre-label billboards with the detector.")
    parser.add_argument("--source", type=Path, default=defaults.source)
    parser.add_argument("--output", type=Path, default=defaults.output)
    parser.add_argument("--weights", type=Path, default=defaults.weights)
    parser.add_argument("--conf", type=float, default=defaults.confidence_min)
    parser.add_argument("--weak-conf", type=float, default=defaults.confidence_weak)
    parser.add_argument("--imgsz", type=int, default=defaults.image_size)
    parser.add_argument("--batch", type=int, default=defaults.batch_size)
    parser.add_argument("--device", type=str, default=defaults.device)
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args = parse_args()
    config = PrelabelConfig(
        weights=args.weights,
        source=args.source,
        output=args.output,
        image_size=args.imgsz,
        confidence_min=args.conf,
        confidence_weak=args.weak_conf,
        batch_size=args.batch,
        device=args.device,
    )
    reports = run(config)
    counts = summary(reports)
    boxes = sum(item.boxes for item in reports)
    logger.info(
        "фотографий %s, рамок %s | пусто %s, слабых %s, уверенных %s",
        len(reports),
        boxes,
        counts[STATUS_EMPTY],
        counts[STATUS_WEAK],
        counts[STATUS_OK],
    )
    logger.info("разметка: %s", config.output / "labels")
    logger.info("порядок проверки: %s", config.output / "review.txt")
    logger.info("импорт в CVAT: %s", config.output / "cvat_annotations.zip")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
