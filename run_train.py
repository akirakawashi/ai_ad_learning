"""Обучение детектора щитов на объединённом наборе."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from training.train import TrainConfig, train

logger = logging.getLogger("train")


def parse_args() -> argparse.Namespace:
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(description="Train the billboard detector.")
    parser.add_argument("--data", type=Path, default=defaults.data)
    parser.add_argument("--weights", type=str, default=defaults.weights)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--imgsz", type=int, default=defaults.image_size)
    parser.add_argument("--batch", type=int, default=defaults.batch)
    parser.add_argument("--patience", type=int, default=defaults.patience)
    parser.add_argument("--device", type=str, default=defaults.device)
    parser.add_argument("--name", type=str, default=defaults.name)
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    best = train(
        TrainConfig(
            data=args.data,
            weights=args.weights,
            epochs=args.epochs,
            image_size=args.imgsz,
            batch=args.batch,
            patience=args.patience,
            device=args.device,
            name=args.name,
        )
    )
    logger.info("лучшие веса: %s", best)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
