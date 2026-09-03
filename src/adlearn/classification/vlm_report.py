"""Итоги прогона VLM и сравнение двух прогонов по одной выборке.

Прогоны идут кругами: поправили промпт или проверку, прогнали ту же выборку,
посмотрели, что сдвинулось. Общая доля верных для этого плохо годится: плюс три
кадра по Миранде и минус три ложных МТС дают ту же цифру, а это разные события.
Поэтому счёт ведётся по брендам и отдельно по ложным срабатываниям — выдуманная
кампания в отчёте дороже пропущенной.
"""

from __future__ import annotations

import csv
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("cls")

OTHER = "other"


@dataclass(frozen=True)
class Outcome:
    file: str
    truth: str
    decided: str


@dataclass(frozen=True)
class BrandScore:
    hits: int
    total: int
    false_positives: int
    """Сколько раз этот бренд назван на кадре, где правда другая."""

    @property
    def share(self) -> float:
        return self.hits / self.total if self.total else 0.0


def read_outcomes(path: Path) -> list[Outcome]:
    """Строки прогона `adlearn cls vlm` с правильными ответами. Кадры без правды пропускаются."""

    with path.open(encoding="utf-8") as handle:
        return [
            Outcome(file=row["file"], truth=row["truth"], decided=row["brand"])
            for row in csv.DictReader(handle)
            if row.get("truth")
        ]


def summarize(outcomes: Iterable[Outcome], *, targets: Sequence[str]) -> dict[str, BrandScore]:
    """Счёт по каждому бренду и по `other`.

    Для `other` ложных срабатываний не бывает: назвать чужой щит чужим — не ошибка,
    а `unclear` вместо `other` — промах, но не выдумка.
    """

    hits: dict[str, int] = dict.fromkeys((*targets, OTHER), 0)
    totals: dict[str, int] = dict.fromkeys((*targets, OTHER), 0)
    false_positives: dict[str, int] = dict.fromkeys((*targets, OTHER), 0)
    for item in outcomes:
        totals[item.truth] = totals.get(item.truth, 0) + 1
        if item.decided == item.truth:
            hits[item.truth] = hits.get(item.truth, 0) + 1
        elif item.decided in targets:
            false_positives[item.decided] += 1
    return {
        name: BrandScore(hits=hits[name], total=totals[name], false_positives=false_positives[name])
        for name in (*targets, OTHER)
    }


def print_summary(outcomes: Sequence[Outcome], *, targets: Sequence[str]) -> None:
    scores = summarize(outcomes, targets=targets)
    correct = sum(item.hits for item in scores.values())
    invented = sum(item.false_positives for item in scores.values())
    logger.info("верно %s/%s, ложных срабатываний %s", correct, len(outcomes), invented)
    logger.info("%-10s %-12s %s", "бренд", "верно", "ложных")
    for name, score in scores.items():
        logger.info(
            "%-10s %3d/%-3d %4.0f%%  %s",
            name,
            score.hits,
            score.total,
            100 * score.share,
            score.false_positives,
        )


def print_comparison(
    before: Sequence[Outcome],
    after: Sequence[Outcome],
    *,
    targets: Sequence[str],
) -> None:
    """Два прогона рядом и список кадров, где решение поменялось."""

    earlier = {item.file: item for item in before}
    later = {item.file: item for item in after}
    common = sorted(set(earlier) & set(later))
    if len(common) != len(earlier) or len(common) != len(later):
        logger.info(
            "выборки различаются: общих кадров %s, было %s, стало %s — сравниваю только общие",
            len(common),
            len(earlier),
            len(later),
        )
    first = summarize((earlier[name] for name in common), targets=targets)
    second = summarize((later[name] for name in common), targets=targets)

    logger.info(
        "%-10s %-12s %-12s %-8s %s", "бренд", "было", "стало", "сдвиг", "ложных было → стало"
    )
    for name in (*targets, OTHER):
        one, two = first[name], second[name]
        logger.info(
            "%-10s %3d/%-3d      %3d/%-3d      %+4d     %s → %s",
            name,
            one.hits,
            one.total,
            two.hits,
            two.total,
            two.hits - one.hits,
            one.false_positives,
            two.false_positives,
        )
    was = sum(item.hits for item in first.values())
    now = sum(item.hits for item in second.values())
    invented_was = sum(item.false_positives for item in first.values())
    invented_now = sum(item.false_positives for item in second.values())
    logger.info(
        "итого верно %s → %s из %s, ложных %s → %s",
        was,
        now,
        len(common),
        invented_was,
        invented_now,
    )

    changed = [
        (earlier[name], later[name])
        for name in common
        if earlier[name].decided != later[name].decided
    ]
    if not changed:
        logger.info("решения не изменились ни на одном кадре")
        return
    logger.info("\nизменившиеся решения (%s):", len(changed))
    for old, new in changed:
        if new.decided == new.truth:
            mark = "исправлено"
        elif old.decided == old.truth:
            mark = "СЛОМАНО"
        else:
            mark = "по-прежнему мимо"
        logger.info(
            "  %-16s %-42s %-8s → %-8s правда %s",
            mark,
            old.file[:42],
            old.decided,
            new.decided,
            old.truth,
        )
