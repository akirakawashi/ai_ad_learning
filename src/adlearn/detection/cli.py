"""Команды детекции: `adlearn detect ...`."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from adlearn import paths
from adlearn.core.cli import Subparsers, command
from adlearn.detection import bundle, checks, dataset, negatives, prelabel, preview, review, train
from adlearn.detection.config import (
    CLASS_NAME,
    DatasetConfig,
    PrelabelConfig,
    ReviewConfig,
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
    _add_review(commands)
    _add_negatives(commands)
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


# --- проверка псевдоразметки ---------------------------------------------


def _add_review(commands: Subparsers) -> None:
    defaults = ReviewConfig()
    parser = command(
        commands,
        "review",
        help="проверить псевдоразметку: рамки через VLM, спорное — на листы",
        handler=_review,
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=("scan", "judge", "sheets", "apply", "cvat", "import", "all"),
        help="scan — прогон детектора, judge — ответы VLM, sheets — листы, apply — чистая разметка",
    )
    parser.add_argument("--source", type=Path, default=defaults.source)
    parser.add_argument("--output", type=Path, default=defaults.output)
    parser.add_argument("--weights", type=Path, default=defaults.weights)
    parser.add_argument("--conf", type=float, default=defaults.confidence_min)
    parser.add_argument("--imgsz", type=int, default=defaults.image_size)
    parser.add_argument("--batch", type=int, default=defaults.batch_size)
    parser.add_argument("--device", type=str, default=defaults.device)
    parser.add_argument("--vlm-url", type=str, default=defaults.vlm_url)
    parser.add_argument("--crop-side", type=int, default=defaults.crop_max_side)
    parser.add_argument("--export", type=Path, help="выгрузка из CVAT для стадии import")
    parser.add_argument(
        "--empty-stride",
        type=int,
        default=defaults.empty_frame_stride,
        help="каждый N-й кадр без находок становится негативом; 0 — отдать человеку",
    )


def _review(args: argparse.Namespace) -> int:
    config = ReviewConfig(
        weights=args.weights,
        source=args.source,
        output=args.output,
        image_size=args.imgsz,
        confidence_min=args.conf,
        batch_size=args.batch,
        device=args.device,
        crop_max_side=args.crop_side,
        vlm_url=args.vlm_url,
        empty_frame_stride=args.empty_stride,
    )
    if args.stage in ("scan", "all"):
        boxes = review.scan(config)
        logger.info("рамки: %s", config.boxes_path)
    else:
        boxes = review.read_boxes_csv(config.boxes_path)

    if args.stage in ("judge", "all"):
        asked = review.judge(config)
        logger.info("новых ответов %s, всё в %s", asked, config.answers_path)

    if args.stage in ("sheets", "all"):
        answers = {item.box_id: item for item in review.read_answers(config.answers_path)}
        by_category: dict[str, list[review.Box]] = {}
        for box in boxes:
            answer = answers.get(box.box_id)
            if answer is None:
                continue
            by_category.setdefault(answer.category, []).append(box)
        sheets = 0
        for category, group in sorted(by_category.items()):
            group.sort(key=lambda box: box.confidence)
            sheets += review.crop_sheets(boxes=group, config=config, name=category)
        empty = review.frames_without_boxes(source=config.source, boxes=boxes)
        sheets += review.frame_sheets(
            files=empty, boxes_by_file={}, config=config, name="empty_frames"
        )
        logger.info("листов %s, всё в %s", sheets, config.sheets_dir)

    if args.stage == "import":
        if args.export is None:
            raise SystemExit("Стадии import нужен --export с архивом из CVAT.")
        imported, imported_boxes, imported_empty = review.import_export(config, archive=args.export)
        logger.info(
            "из CVAT: кадров %s, рамок %s, без рамок %s", imported, imported_boxes, imported_empty
        )
        logger.info("осталось в очереди: %s", len(config.cvat_list_path.read_text().split()))
        return 0

    if args.stage == "cvat":
        images_archive, annotations_archive, count = review.pack_disputed(config)
        logger.info(
            "кадров %s, картинки %s (%.0f МБ), разметка %s",
            count,
            images_archive,
            images_archive.stat().st_size / 1e6,
            annotations_archive,
        )
        return 0

    if args.stage == "apply":
        applied = review.apply_verdicts(config)
        logger.info(
            "кадров: чисто %s, поправлено %s, негативов %s, в CVAT %s, без рамок %s",
            applied.clean,
            applied.fixed,
            applied.negative,
            applied.disputed,
            applied.no_boxes,
        )
        logger.info("рамок было %s, стало %s", applied.boxes_before, applied.boxes_after)
        logger.info("разметка: %s", config.labels_dir)
        logger.info("список для CVAT: %s", config.cvat_list_path)
        return 0

    counts: dict[str, int] = {}
    for item in review.read_answers(config.answers_path):
        counts[item.category] = counts.get(item.category, 0) + 1
    for category, total in sorted(counts.items(), key=lambda pair: -pair[1]):
        logger.info("%s: %s", category, total)
    return 0


# --- отрицательные примеры со стороны -------------------------------------


def _add_negatives(commands: Subparsers) -> None:
    parser = command(
        commands,
        "negatives",
        help="фотографии без рекламы: разложить с пустой разметкой и удвоить копиями",
        handler=_negatives,
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=TASK.root / "negatives")
    parser.add_argument("--group", type=str, default="truck")
    parser.add_argument(
        "--skip", type=str, nargs="*", default=(), help="имена файлов, которые не брать"
    )
    parser.add_argument("--seed", type=int, default=0)


def _negatives(args: argparse.Namespace) -> int:
    originals, augmented, skipped = negatives.prepare(
        negatives.NegativesConfig(
            source=args.source,
            output=args.output,
            group=args.group,
            skip=tuple(args.skip),
            seed=args.seed,
        )
    )
    logger.info("оригиналов %s, копий %s, пропущено %s", originals, augmented, skipped)
    logger.info("кадры: %s", args.output / "images")
    return 0
