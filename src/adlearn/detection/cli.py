"""Команды детекции: `adlearn detect ...`."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from adlearn import paths
from adlearn.core.cli import Subparsers, command
from adlearn.detection import bundle, checks, dataset, prelabel, preview, train
from adlearn.detection.config import (
    CLASS_NAME,
    DatasetConfig,
    PrelabelConfig,
    TrainConfig,
)
from adlearn.detection.report import STATUS_EMPTY, STATUS_OK, STATUS_WEAK, summary

logger = logging.getLogger("detect")

TASK = paths.DETECTION


def register(tasks: Subparsers) -> None:
    parser = tasks.add_parser(
        "detect",
        help="детектор рекламных щитов",
        description="Псевдоразметка, сборка набора, проверки и обучение детектора.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    _add_prelabel(commands)
    _add_bundle(commands)
    _add_build(commands)
    _add_check(commands)
    _add_preview(commands)
    _add_train(commands)
    _add_eval(commands)


# --- псевдоразметка -------------------------------------------------------


def _add_prelabel(commands: Subparsers) -> None:
    defaults = PrelabelConfig()
    parser = command(
        commands,
        "prelabel",
        help="прогнать фотографии через детектор: разметка, отчёт, архив для CVAT",
        handler=_prelabel,
    )
    parser.add_argument("--source", type=Path, default=defaults.source)
    parser.add_argument("--output", type=Path, default=defaults.output)
    parser.add_argument("--weights", type=Path, default=defaults.weights)
    parser.add_argument("--conf", type=float, default=defaults.confidence_min)
    parser.add_argument("--weak-conf", type=float, default=defaults.confidence_weak)
    parser.add_argument("--imgsz", type=int, default=defaults.image_size)
    parser.add_argument("--batch", type=int, default=defaults.batch_size)
    parser.add_argument("--device", type=str, default=defaults.device)


def _prelabel(args: argparse.Namespace) -> int:
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
    reports = prelabel.run(config)
    counts = summary(reports)
    logger.info(
        "фотографий %s, рамок %s | пусто %s, слабых %s, уверенных %s",
        len(reports),
        sum(item.boxes for item in reports),
        counts[STATUS_EMPTY],
        counts[STATUS_WEAK],
        counts[STATUS_OK],
    )
    logger.info("разметка: %s", config.labels_dir)
    logger.info("порядок проверки: %s", config.review_path)
    logger.info("импорт в CVAT: %s", config.cvat_archive)
    return 0


# --- пачки для ручной правки ---------------------------------------------


def _add_bundle(commands: Subparsers) -> None:
    defaults = PrelabelConfig()
    parser = command(
        commands,
        "bundle",
        help="собрать спорные кадры в пачки под задачи CVAT",
        handler=_bundle,
    )
    parser.add_argument("--prelabel", type=Path, default=defaults.output)
    parser.add_argument("--output", type=Path, default=TASK.export)
    parser.add_argument("--chunk", type=int, default=bundle.CHUNK_SIZE)
    parser.add_argument("--class-name", type=str, default=CLASS_NAME)


def _bundle(args: argparse.Namespace) -> int:
    parts = bundle.build(
        report_path=args.prelabel / "report.csv",
        images_dir=args.prelabel / "images",
        labels_dir=args.prelabel / "labels",
        output=args.output,
        class_name=args.class_name,
        chunk_size=args.chunk,
    )
    for images_archive, annotations_archive, count in parts:
        logger.info(
            "%s — кадров %s, %.0f МБ, разметка %s",
            images_archive.name,
            count,
            images_archive.stat().st_size / 1e6,
            annotations_archive.name,
        )
    logger.info("пачек %s, всё в %s", len(parts), args.output)
    return 0


# --- сборка набора --------------------------------------------------------


def _add_build(commands: Subparsers) -> None:
    defaults = DatasetConfig(export_archive=Path())
    parser = command(
        commands,
        "build",
        help="собрать набор из ручной разметки и уверенной псевдоразметки",
        handler=_build,
    )
    parser.add_argument("--export", required=True, type=Path, help="выгрузка из CVAT")
    parser.add_argument("--prelabel", type=Path, default=defaults.prelabel)
    parser.add_argument("--output", type=Path, default=defaults.output)
    parser.add_argument("--class-name", type=str, default=defaults.class_name)
    parser.add_argument("--val-share", type=float, default=defaults.validation_share)
    parser.add_argument("--test-share", type=float, default=defaults.test_share)
    parser.add_argument("--seed", type=int, default=defaults.seed)


def _build(args: argparse.Namespace) -> int:
    counts = dataset.build(
        DatasetConfig(
            export_archive=args.export,
            prelabel=args.prelabel,
            output=args.output,
            class_name=args.class_name,
            validation_share=args.val_share,
            test_share=args.test_share,
            seed=args.seed,
        )
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


# --- проверка набора ------------------------------------------------------


def _add_check(commands: Subparsers) -> None:
    parser = command(
        commands,
        "check",
        help="проверить набор перед обучением",
        handler=_check,
    )
    parser.add_argument("--dataset", type=Path, default=TASK.dataset)
    parser.add_argument(
        "--skip-decode",
        action="store_true",
        help="не открывать каждый кадр — быстрее, но нечитаемые не найдутся",
    )


def _check(args: argparse.Namespace) -> int:
    findings, boxes = checks.inspect(args.dataset, read_images=not args.skip_decode)

    for split in dataset.SPLITS:
        frames = len(checks.image_paths(args.dataset, split))
        logger.info(
            "%s: кадров %s, рамок %s, в среднем %.2f на кадр",
            split,
            frames,
            boxes[split],
            boxes[split] / frames if frames else 0.0,
        )

    for field_name, title in checks.TITLES.items():
        items = getattr(findings, field_name)
        if items:
            logger.info("%s: %s — %s", title, len(items), ", ".join(items[:5]))

    logger.info("итог: %s", "чисто" if findings.clean else "есть замечания")
    return 0 if findings.clean else 1


# --- просмотр разметки ----------------------------------------------------


def _add_preview(commands: Subparsers) -> None:
    parser = command(
        commands,
        "preview",
        help="контактные листы: разметка поверх кадров",
        handler=_preview,
    )
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--dataset", type=Path, default=TASK.dataset)
    parser.add_argument("--output", type=Path, default=TASK.preview)
    parser.add_argument("--limit", type=int, default=120)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--rows", type=int, default=3)
    parser.add_argument("--side", type=int, default=420)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--only-empty", action="store_true", help="только кадры без рамок")
    parser.add_argument("--frames", action="store_true", help="сохранить и отдельные кадры")


def _preview(args: argparse.Namespace) -> int:
    output = args.output / args.split
    frames, sheets = preview.build_sheets(
        images_dir=args.dataset / "images" / args.split,
        labels_dir=args.dataset / "labels" / args.split,
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


# --- обучение и оценка ----------------------------------------------------


def _add_train(commands: Subparsers) -> None:
    defaults = TrainConfig()
    parser = command(commands, "train", help="обучить детектор", handler=_train)
    parser.add_argument("--data", type=Path, default=defaults.data)
    parser.add_argument("--weights", type=str, default=defaults.weights)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--imgsz", type=int, default=defaults.image_size)
    parser.add_argument("--batch", type=int, default=defaults.batch)
    parser.add_argument("--patience", type=int, default=defaults.patience)
    parser.add_argument("--device", type=str, default=defaults.device)
    parser.add_argument("--name", type=str, default=defaults.name)


def _train(args: argparse.Namespace) -> int:
    best = train.train(
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


def _add_eval(commands: Subparsers) -> None:
    defaults = TrainConfig()
    parser = command(
        commands,
        "eval",
        help="сравнить веса на отложенной части",
        handler=_eval,
    )
    parser.add_argument("--weights", type=Path, nargs="+", required=True)
    parser.add_argument("--data", type=Path, default=defaults.data)
    parser.add_argument("--imgsz", type=int, default=defaults.image_size)
    parser.add_argument("--device", type=str, default=defaults.device)
    parser.add_argument(
        "--split",
        type=str,
        default=dataset.HANDMADE_SPLIT,
        choices=("test", dataset.HANDMADE_SPLIT, "val"),
    )


def _eval(args: argparse.Namespace) -> int:
    for weights in args.weights:
        scores = train.evaluate(
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
