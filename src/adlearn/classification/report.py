"""Печать результатов абляции."""

from __future__ import annotations

import logging

import numpy as np

from adlearn.classification.ablation import SHORTCUT_ARM, ArmResult
from adlearn.classification.config import BRANDS

logger = logging.getLogger("cls")


def print_scores(results: list[ArmResult], *, baseline: str = "визуал") -> None:
    reference = next((item for item in results if item.name == baseline), None)
    logger.info("%-28s %-16s %s", "арма", "macro F1", "дельта к базе")
    for item in results:
        delta = ""
        if reference is not None and item is not reference:
            paired = np.array(item.macro_f1) - np.array(reference.macro_f1)
            sign = "+" if paired.mean() >= 0 else ""
            delta = f"{sign}{paired.mean():.3f} ± {paired.std():.3f}"
        logger.info("%-28s %.3f ± %.3f   %s", item.name, item.mean, item.spread, delta)
    warn = next((i for i in results if i.name == SHORTCUT_ARM), None)
    if warn is not None and warn.mean > 0.5:
        logger.info(
            "\nВНИМАНИЕ: одни размеры и резкость дают %.3f. Классы различаются\n"
            "источником кадров, а не содержанием — остальные цифры завышены.",
            warn.mean,
        )


def print_matrix(result: ArmResult) -> None:
    matrix = result.matrix
    logger.info("\nconfusion matrix — %s (строки: правда, столбцы: ответ)", result.name)
    logger.info("%-12s%s", "", "".join(f"{name:>10}" for name in BRANDS))
    for name, row in zip(BRANDS, matrix, strict=True):
        total = row.sum()
        cells = "".join(f"{value:>10}" for value in row)
        logger.info(
            "%-12s%s   (%d, %.0f%% верно)",
            name,
            cells,
            total,
            100.0 * row[list(BRANDS).index(name)] / total if total else 0.0,
        )
