"""Группы съёмки и деление набора на обучение, проверку и тест."""

from __future__ import annotations

import random
import re
from collections import defaultdict
from collections.abc import Callable, Hashable, Sequence
from pathlib import Path

VALIDATION_SHARE = 0.15
TEST_SHARE = 0.15
SPLIT_SEED = 0

HASH_NAME = re.compile(r"^[0-9a-f]{12,}$")


def group_of(name: str) -> str:
    """Откуда кадр: `video_hard_000187.jpg` → `video_hard`, `3f9c1a2b8e04.jpg` → `raw`.

    Группа — это съёмка одного рода: кадры с регистратора, студийные фотографии,
    выгрузка из интернета. Деление на части идёт внутри групп, чтобы рабочий
    домен попал во все три, а не только в обучение.
    """

    stem = Path(name).stem
    if HASH_NAME.match(stem.lower()):
        return "raw"
    prefix = re.sub(r"[0-9].*$", "", stem)
    return prefix.strip("_") or "numbers"


def stratified_split[T](
    items: Sequence[T],
    *,
    stratum: Callable[[T], Hashable],
    order: Callable[[T], str],
    validation_share: float = VALIDATION_SHARE,
    test_share: float = TEST_SHARE,
    seed: int = SPLIT_SEED,
) -> tuple[list[T], list[T], list[T]]:
    """Делит набор на обучение, проверку и отложенный тест.

    Проверка ведёт обучение: по ней выбирается лучшая эпоха и срабатывает ранняя
    остановка. Тест не участвует ни в том, ни в другом — он и отвечает, стала ли
    модель лучше.

    Делится внутри страт, которые задаёт вызывающий: обычно это пара «группа
    съёмки и содержимое кадра». Первое разводит по всем частям кадры с
    регистратора и снимки из интернета, второе не даёт целому классу кадров
    осесть в одной части и перекосить сразу и обучение, и метрику.

    Страта меньше трёх кадров уходит в обучение целиком: тест из одного кадра
    ничего не измеряет, а обучение теряет и этот кадр.

    `order` даёт устойчивый ключ сортировки, чтобы при одном `seed` деление
    повторялось от запуска к запуску независимо от порядка файлов на диске.
    """

    by_stratum: dict[Hashable, list[T]] = defaultdict(list)
    for item in items:
        by_stratum[stratum(item)].append(item)

    rng = random.Random(seed)
    train: list[T] = []
    validation: list[T] = []
    test: list[T] = []
    for key in sorted(by_stratum, key=repr):
        members = sorted(by_stratum[key], key=order)
        rng.shuffle(members)
        if len(members) < 3:
            train.extend(members)
            continue
        validation_size = max(1, round(len(members) * validation_share))
        test_size = max(1, round(len(members) * test_share))
        while validation_size + test_size >= len(members):
            if test_size >= validation_size:
                test_size -= 1
            else:
                validation_size -= 1
        validation.extend(members[:validation_size])
        test.extend(members[validation_size : validation_size + test_size])
        train.extend(members[validation_size + test_size :])
    return train, validation, test
