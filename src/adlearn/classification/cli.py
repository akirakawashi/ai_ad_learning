"""Команды классификации: `adlearn cls ...`."""

from __future__ import annotations

import argparse
import csv
import logging
import random
import shutil
import time
from collections import Counter
from pathlib import Path

import joblib
import numpy as np

from adlearn import paths
from adlearn.classification import dataset, vlm, vlm_report
from adlearn.classification.ablation import ARMS, SHORTCUT_ARM, build_blocks, run_arm
from adlearn.classification.config import (
    BRANDS,
    HARD_NEGATIVES,
    STREET_SOURCE,
    TELECOM_BRANDS,
    UNSURE,
    ClassificationConfig,
)
from adlearn.classification.features import VisualExtractor, get
from adlearn.classification.features.cache import compute
from adlearn.classification.head import BlockScaler, make_head
from adlearn.classification.report import print_matrix, print_scores
from adlearn.core.cli import Subparsers, command
from adlearn.core.images import find_images, reset_dir

logger = logging.getLogger("cls")
DEFAULTS = ClassificationConfig()
TASK = paths.CLASSIFICATION


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
    _add_vlm(commands)
    _add_compare(commands)
    _add_probe(commands)
    _add_blind(commands)


def _load_features(
    args: argparse.Namespace, samples: list[dataset.Sample]
) -> tuple[dict[str, np.ndarray], dict[str, list[str]]]:
    paths = [item.path for item in samples]
    ids = [item.id for item in samples]
    features: dict[str, np.ndarray] = {}
    dims: dict[str, list[str]] = {}
    for name in ("visual", "color", "shortcut"):
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
    # shortcut в модель не идёт: это диагностика, а не признак бренда
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
        (SHORTCUT_ARM, ("shortcut",), False),
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


# --- VLM ------------------------------------------------------------------


def _add_vlm(commands: Subparsers) -> None:
    parser = command(
        commands,
        "vlm",
        help="определить бренд зрительно-языковой моделью (нужен llama-server)",
        handler=_vlm,
    )
    parser.add_argument("--source", required=True, type=Path, help="папка с кадрами")
    parser.add_argument("--url", type=str, default="http://127.0.0.1:8080")
    parser.add_argument("--output", type=Path, default=None, help="куда положить CSV")
    parser.add_argument(
        "--labels",
        type=Path,
        default=None,
        help="CSV с правильными ответами: колонки file,brand",
    )
    parser.add_argument("--limit", type=int, default=0, help="взять только первые N кадров")
    parser.add_argument(
        "--accept-logo",
        action="store_true",
        help="засчитывать ответ по одному фирменному знаку, без прочитанного названия",
    )


def _read_labels(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8") as handle:
        return {row["file"]: row["brand"] for row in csv.DictReader(handle)}


def _vlm(args: argparse.Namespace) -> int:
    if not vlm.health(args.url):
        raise ConnectionError(f"llama-server не отвечает на {args.url}. Запусти его и повтори.")

    paths = find_images(args.source, recursive=True)
    if args.limit:
        paths = paths[: args.limit]
    truth = _read_labels(args.labels) if args.labels else {}
    logger.info("кадров %s, модель на %s", len(paths), args.url)

    output = args.output or args.source / "vlm_predictions.csv"
    output.parent.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    answers = []
    hits = raw_hits = graded = failed = 0
    # Пишем построчно: прогон идёт десятки минут, и падение на середине не должно
    # стоить всей работы.
    handle = output.open("w", encoding="utf-8", newline="")
    writer = csv.writer(handle)
    writer.writerow(
        [
            "file",
            "brand",
            "raw_brand",
            "basis",
            "verdict",
            "truth",
            "visible_text",
            "evidence",
            "error",
        ]
    )
    for index, path in enumerate(paths, start=1):
        answer = vlm.ask(path, url=args.url)
        answers.append(answer)
        failed += bool(answer.error)

        decided = answer.decided(accept_logo=args.accept_logo)
        expected = truth.get(path.name) or truth.get(path.stem)
        mark = ""
        if expected:
            graded += 1
            ok = decided == expected
            hits += ok
            raw_hits += answer.brand == expected
            mark = "  ✓" if ok else f"  ✗ правда {expected}"

        shown = "ОШИБКА" if answer.error else decided
        tail = "" if decided == answer.brand else f" (сырое: {answer.brand})"
        logger.info(
            "%4d/%d  %-30s %-9s%s%s  | %s",
            index,
            len(paths),
            path.name[:30],
            shown,
            mark,
            tail,
            (answer.error or answer.visible_text or answer.evidence)[:44],
        )
        writer.writerow(
            [
                path.name,
                decided,
                answer.brand,
                answer.basis,
                answer.verdict,
                expected or "",
                answer.visible_text,
                answer.evidence,
                answer.error,
            ]
        )
        handle.flush()
    handle.close()

    elapsed = time.monotonic() - started
    logger.info(
        "\nготово за %.0f с, %.1f с на кадр%s",
        elapsed,
        elapsed / max(1, len(paths)),
        f", сбоев {failed}" if failed else "",
    )
    counts = Counter(item.decided(accept_logo=args.accept_logo) for item in answers)
    for name, number in counts.most_common():
        logger.info("  %-10s %4d  %5.0f%%", name, number, 100 * number / len(answers))

    logger.info("\nоснование ответа:")
    for name, number in Counter(item.basis for item in answers).most_common():
        logger.info("  %-14s %4d", name, number)

    if graded:
        logger.info(
            "\nверно %s/%s = %.0f%%   (без проверки основания: %s = %.0f%%)",
            hits,
            graded,
            100 * hits / graded,
            raw_hits,
            100 * raw_hits / graded,
        )
        _score_brands(answers, truth, accept_logo=args.accept_logo)
        logger.info("")
        vlm_report.print_summary(vlm_report.read_outcomes(output), targets=TELECOM_BRANDS)

    logger.info("таблица: %s", output)
    return 0


# --- сравнение прогонов ---------------------------------------------------


def _add_compare(commands: Subparsers) -> None:
    parser = command(
        commands,
        "compare",
        help="сравнить два прогона VLM по одной выборке: что исправилось, что сломалось",
        handler=_compare,
    )
    parser.add_argument("before", type=Path, help="CSV прошлого прогона")
    parser.add_argument("after", type=Path, help="CSV нового прогона")


def _compare(args: argparse.Namespace) -> int:
    vlm_report.print_comparison(
        vlm_report.read_outcomes(args.before),
        vlm_report.read_outcomes(args.after),
        targets=TELECOM_BRANDS,
    )
    return 0


# --- выборка для проверки --------------------------------------------------


def _add_probe(commands: Subparsers) -> None:
    parser = command(
        commands,
        "probe",
        help="собрать выборку для проверки: двойники, улица, телеком",
        handler=_probe,
    )
    parser.add_argument("--raw", type=Path, default=DEFAULTS.raw)
    parser.add_argument("--output", type=Path, default=TASK.root / "probe" / "large")
    parser.add_argument("--per-brand", type=int, default=60, help="кадров на телеком-бренд")
    parser.add_argument(
        "--per-hard", type=int, default=0, help="кадров с каждого двойника, 0 — все"
    )
    parser.add_argument("--street", type=int, default=150, help="кадров с улицы")
    parser.add_argument("--random-other", type=int, default=150, help="случайное «другое»")
    parser.add_argument("--seed", type=int, default=0)


def _probe(args: argparse.Namespace) -> int:
    """Собирает выборку ссылками и пишет к ней правильные ответы.

    Случайная выборка из `other` почти не содержит двойников — Сбер и Магнит в ней
    единицы, и ошибка на них потеряется среди лёгких кадров. Поэтому двойники
    берутся целиком, а остальное досыпается для фона.
    """

    samples = dataset.collect(args.raw, brands=(*TELECOM_BRANDS, "other"))
    rng = random.Random(args.seed)
    by_source: dict[str, list[dataset.Sample]] = {}
    for item in samples:
        by_source.setdefault(item.source or item.brand, []).append(item)

    def take(pool: list[dataset.Sample], count: int) -> list[dataset.Sample]:
        return rng.sample(pool, min(count, len(pool)))

    chosen: list[dataset.Sample] = []
    for name in HARD_NEGATIVES:
        pool = by_source.get(name, [])
        chosen += take(pool, args.per_hard) if args.per_hard else pool
    picked_hard = len(chosen)

    chosen += take(by_source.get(STREET_SOURCE, []), args.street)
    rest = [
        item
        for item in samples
        if item.brand == "other"
        and item.source not in HARD_NEGATIVES
        and item.source != STREET_SOURCE
    ]
    chosen += take(rest, args.random_other)
    for brand in TELECOM_BRANDS:
        chosen += take(by_source.get(brand, []), args.per_brand)

    output = reset_dir(args.output)
    rows = []
    for item in chosen:
        source = (item.source or item.brand).replace("/", "-").replace(" ", "_")
        name = f"{source}__{item.path.name}"
        # Копия, а не ссылка: выборку удобно открыть глазами, в том числе из Windows.
        shutil.copy2(item.path, output / name)
        rows.append((name, item.brand, item.source or item.brand))

    labels = output / "labels.csv"
    with labels.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["file", "brand", "source"])
        writer.writerows(rows)

    logger.info("кадров %s, всё в %s", len(rows), output)
    logger.info("  двойники (все)          %s", picked_hard)
    logger.info(
        "  улица                   %s", min(args.street, len(by_source.get(STREET_SOURCE, [])))
    )
    logger.info("  прочее «другое»         %s", min(args.random_other, len(rest)))
    logger.info("  телеком                 %s", sum(1 for r in rows if r[1] != "other"))
    logger.info("ответы: %s", labels)
    logger.info("примерное время прогона: %.0f мин при 2.8 с на кадр", len(rows) * 2.8 / 60)
    return 0


# --- обезличивание -------------------------------------------------------


def _add_blind(commands: Subparsers) -> None:
    parser = command(
        commands,
        "blind",
        help="обезличить выборку: имена файлов заменить номерами, ключ в отдельный CSV",
        handler=_blind,
    )
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)


def _blind(args: argparse.Namespace) -> int:
    """Копия выборки с обезличенными именами.

    В запрос к модели имя файла и так не попадает — уходят только байты картинки
    и постоянный промпт. Но обезличенная копия снимает вопрос целиком: её можно
    без опаски скормить чему угодно, включая веб-интерфейсы и чужие инструменты,
    где имя файла может уехать вместе с картинкой.

    Порядок перемешивается, чтобы номер не выдавал исходную сортировку по классам.
    """

    paths = find_images(args.source, recursive=True)
    rng = random.Random(args.seed)
    shuffled = paths[:]
    rng.shuffle(shuffled)

    keys: dict[str, dict[str, dict[str, str]]] = {}
    for table in args.source.glob("*.csv"):
        with table.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                keys.setdefault(row["file"], {})[table.stem] = row

    output = reset_dir(args.output)
    rows = []
    for number, path in enumerate(shuffled, start=1):
        name = f"img_{number:04d}{path.suffix.lower()}"
        shutil.copy2(path, output / name)
        known = keys.get(path.name, {})
        best: dict[str, str] = next(iter(known.values()), {})
        rows.append(
            [
                name,
                best.get("brand", ""),
                best.get("source", ""),
                best.get("quality", ""),
                path.name,
            ]
        )

    key_path = args.output.parent / f"{args.output.name}_key.csv"
    with key_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["file", "brand", "source", "quality", "original"])
        writer.writerows(rows)

    logger.info("обезличено кадров %s → %s", len(rows), output)
    logger.info("ключ (не клади рядом с картинками): %s", key_path)
    return 0


def _score_brands(
    answers: list[vlm.VlmAnswer],
    truth: dict[str, str],
    *,
    accept_logo: bool,
) -> None:
    """Точность и полнота по телеком-брендам.

    Для мониторинга наружки важнее точность: выдуманная кампания в отчёте дороже
    пропущенной. Поэтому цифры печатаются отдельно, а не одной общей долей верных.
    """

    targets = set(vlm.NAME_FORMS)
    called = correct = present = found = 0
    for item in answers:
        expected = truth.get(item.path.name) or truth.get(item.path.stem)
        if not expected:
            continue
        decided = item.decided(accept_logo=accept_logo)
        called += decided in targets
        correct += decided in targets and decided == expected
        present += expected in targets
        found += expected in targets and decided == expected
    if called:
        logger.info("точность: %s/%s = %.0f%%", correct, called, 100 * correct / called)
    if present:
        logger.info("полнота:  %s/%s = %.0f%%", found, present, 100 * found / present)
