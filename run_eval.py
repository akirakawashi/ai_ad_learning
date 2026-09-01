"""Сравнение весов на проверочной части набора."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from training.train import TrainConfig, evaluate

logger = logging.getLogger("eval")


def parse_args() -> argparse.Namespace:
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(description="Compare detector weights on the val split.")
    parser.add_argument("--weights", type=Path, nargs="+", required=True)
    parser.add_argument("--data", type=Path, default=defaults.data)
    parser.add_argument("--imgsz", type=int, default=defaults.image_size)
    parser.add_argument("--device", type=str, default=defaults.device)
    parser.add_argument("--split", type=str, default="test_handmade", choices=("test", "test_handmade", "val"))
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    for weights in args.weights:
        scores = evaluate(
            weights=weights,
            data=args.data,
            image_size=args.imgsz,
            device=args.device,
            split=args.split,
        )
        logger.info(
            "%s: mAP50 %.3f, mAP50-95 %.3f, точность %.3f, полнота %.3f",
            weights,
            scores["mAP50"],
            scores["mAP50-95"],
            scores["precision"],
            scores["recall"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
