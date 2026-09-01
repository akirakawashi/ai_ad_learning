"""Команды классификации: `adlearn cls ...`."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import numpy as np

from adlearn.classification import dataset
from adlearn.classification.ablation import ARMS, build_blocks, run_arm
from adlearn.classification.config import BRANDS, UNSURE, ClassificationConfig
from adlearn.classification.features import VisualExtractor, get
from adlearn.classification.features.cache import compute
from adlearn.classification.head import BlockScaler, make_head
from adlearn.classification.report import print_matrix, print_scores
from adlearn.core.cli import Subparsers, command
from adlearn.core.images import find_images

logger = logging.getLogger("cls")
DEFAULTS = ClassificationConfig()


def register(tasks: Subparsers) -> None:
    parser = tasks.add_parser(
        "cls",
        help="классификация брендов",
        description="Признаки, обучение головы, абляция и предсказание бренда.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    _add_features(commands)
    _add_train(commands)
    _add_ablate(commands)
    _add_predict(commands)


def _load_features(
    args: argparse.Namespace, samples: list[dataset.Sample]
) -> tuple[dict[str, np.ndarray], dict[str, list[str]]]:
    paths = [item.path for item in samples]
    ids = [item.id for item in samples]
    features: dict[str, np.ndarray] = {}
    dims: dict[str, list[str]] = {}
    for name in ("visual", "color"):
        extractor = (
            VisualExtractor(device=args.device, batch_size=args.batch)
            if name == "visual"
            else get(name)
        )
        logger.info("признаки «%s» …", name)
        features[name] = compute(
            extractor=extractor,
            paths=paths,
            ids=ids,
            directory=args.features,
            refresh=getattr(args, "refresh", False),
        )
        dims[name] = extractor.dims
        logger.info("  готово: %s × %s", *features[name].shape)
    return features, dims


# --- признаки -------------------------------------------------------------


def _add_features(commands: Subparsers) -> None:
    parser = command(
        commands,
        "features",
        help="посчитать эмбеддинги и цветовые признаки, положить в кэш",
        handler=_features,
    )
    parser.add_argument("--raw", type=Path, default=DEFAULTS.raw)
    parser.add_argument("--features", type=Path, default=DEFAULTS.features)
    parser.add_argument("--device", type=str, default=DEFAULTS.device)
    parser.add_argument("--batch", type=int, default=DEFAULTS.batch_size)
    parser.add_argument("--refresh", action="store_true", help="пересчитать, игнорируя кэш")


def _features(args: argparse.Namespace) -> int:
    samples = dataset.collect(args.raw)
    logger.info("кадров %s", len(samples))
    for brand in BRANDS:
        logger.info("  %-10s %s", brand, sum(1 for item in samples if item.brand == brand))
    _load_features(args, samples)
    logger.info("кэш: %s", args.features)
    return 0


# --- обучение головы ------------------------------------------------------


def _add_train(commands: Subparsers) -> None:
    parser = command(
        commands,
        "train",
        help="обучить голову на всём наборе и сохранить модель",
        handler=_train,
    )
    parser.add_argument("--raw", type=Path, default=DEFAULTS.raw)
    parser.add_argument("--features", type=Path, default=DEFAULTS.features)
    parser.add_argument("--model", type=Path, default=DEFAULTS.model)
    parser.add_argument("--device", type=str, default=DEFAULTS.device)
    parser.add_argument("--batch", type=int, default=DEFAULTS.batch_size)
    parser.add_argument("--color-weight", type=float, default=1.0)
    parser.add_argument("-C", "--regularization", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=DEFAULTS.seed)


def _train(args: argparse.Namespace) -> int:
    samples = dataset.collect(args.raw)
    features, dims = _load_features(args, samples)
    y = dataset.labels(samples)
    blocks = {"visual": features["visual"], "color": features["color"]}
    scaler = BlockScaler(weights={"color": args.color_weight})
    model = make_head(regularization=args.regularization, seed=args.seed)
    model.fit(scaler.fit_transform(blocks), y)
    args.model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "scaler": scaler,
            "model": model,
            "brands": BRANDS,
            "dims": dims,
            "color_weight": args.color_weight,
        },
        args.model,
    )
    logger.info("модель сохранена: %s", args.model)
    logger.info(
        "качество на обучении (не метрика!): %.3f", model.score(scaler.transform(blocks), y)
    )
    logger.info("честные цифры даёт: adlearn cls ablate")
    return 0


# --- абляция --------------------------------------------------------------


def _add_ablate(commands: Subparsers) -> None:
    parser = command(
        commands,
        "ablate",
        help="сравнить армы: визуал / цвет / вместе + контроли",
        handler=_ablate,
    )
    parser.add_argument("--raw", type=Path, default=DEFAULTS.raw)
    parser.add_argument("--features", type=Path, default=DEFAULTS.features)
    parser.add_argument("--device", type=str, default=DEFAULTS.device)
    parser.add_argument("--batch", type=int, default=DEFAULTS.batch_size)
    parser.add_argument("--folds", type=int, default=DEFAULTS.folds)
    parser.add_argument("--repeats", type=int, default=DEFAULTS.repeats)
    parser.add_argument("--color-weight", type=float, default=1.0)
    parser.add_argument("-C", "--regularization", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=DEFAULTS.seed)
    parser.add_argument("--similarity", type=float, default=DEFAULTS.near_duplicate_similarity)


def _ablate(args: argparse.Namespace) -> int:
    samples = dataset.collect(args.raw)
    features, dims = _load_features(args, samples)
    y = dataset.labels(samples)

    groups = dataset.duplicate_groups(features["visual"], threshold=args.similarity)
    unique = len(set(groups.tolist()))
    logger.info(
        "кадров %s, групп после дедупа %s (склеено %s)", len(samples), unique, len(samples) - unique
    )

    plan = [(name, arm, False) for name, arm in ARMS.items()]
    plan += [
        ("цвет (без брендов)", ("color",), True),
        ("визуал + цвет (без брендов)", ("visual", "color"), True),
    ]

    results = []
    for name, arm, generic in plan:
        logger.info("арма «%s» …", name)
        results.append(
            run_arm(
                name=name,
                blocks=build_blocks(features, dims, arm, generic_only=generic),
                y=y,
                groups=groups,
                folds=args.folds,
                repeats=args.repeats,
                color_weight=args.color_weight,
                regularization=args.regularization,
                seed=args.seed,
            )
        )

    logger.info("")
    print_scores(results)
    for item in results:
        if item.name in ("визуал", "визуал + цвет"):
            print_matrix(item)
    return 0


# --- предсказание ---------------------------------------------------------


def _add_predict(commands: Subparsers) -> None:
    parser = command(
        commands,
        "predict",
        help="определить бренд на папке с фотографиями",
        handler=_predict,
    )
    parser.add_argument("--source", required=True, type=Path, help="папка с фотографиями")
    parser.add_argument("--model", type=Path, default=DEFAULTS.model)
    parser.add_argument("--output", type=Path, default=None, help="куда положить CSV")
    parser.add_argument("--device", type=str, default=DEFAULTS.device)
    parser.add_argument("--batch", type=int, default=DEFAULTS.batch_size)
    parser.add_argument("--min-confidence", type=float, default=DEFAULTS.confidence_min)


def _predict(args: argparse.Namespace) -> int:
    if not args.model.exists():
        raise FileNotFoundError(f"Нет модели {args.model}. Сначала: adlearn cls train")
    bundle = joblib.load(args.model)
    paths = find_images(args.source)
    logger.info("кадров %s", len(paths))

    visual = VisualExtractor(device=args.device, batch_size=args.batch)
    blocks = {"visual": visual(paths), "color": get("color")(paths)}
    probabilities = bundle["model"].predict_proba(bundle["scaler"].transform(blocks))

    brands = list(bundle["brands"])
    rows = []
    unsure = 0
    for path, row in zip(paths, probabilities, strict=True):
        best = int(np.argmax(row))
        confidence = float(row[best])
        answer = brands[best] if confidence >= args.min_confidence else UNSURE
        unsure += answer == UNSURE
        rows.append((path.name, answer, confidence, row))
        logger.info(
            "%-40s %-12s %.2f   [%s]",
            path.name[:40],
            answer,
            confidence,
            " ".join(f"{b} {p:.2f}" for b, p in zip(brands, row, strict=True)),
        )

    logger.info("\nвсего %s, «не уверен» %s (порог %.2f)", len(rows), unsure, args.min_confidence)

    output = args.output or args.source / "predictions.csv"
    with output.open("w", encoding="utf-8", newline="") as handle:
        import csv

        writer = csv.writer(handle)
        writer.writerow(["file", "brand", "confidence", *brands])
        for name, answer, confidence, row in rows:
            writer.writerow([name, answer, f"{confidence:.4f}", *[f"{p:.4f}" for p in row]])
    logger.info("таблица: %s", output)
    return 0
