from __future__ import annotations

import logging

from adlearn.classification.vlm_report import Outcome, print_comparison, summarize

TARGETS = ("mts", "tele2")


def test_false_positive_is_charged_to_the_brand_that_was_named() -> None:
    scores = summarize(
        [
            Outcome("a.jpg", truth="mts", decided="mts"),
            Outcome("b.jpg", truth="other", decided="mts"),
            Outcome("c.jpg", truth="tele2", decided="unclear"),
            Outcome("d.jpg", truth="other", decided="unclear"),
        ],
        targets=TARGETS,
    )

    assert scores["mts"].hits == 1
    assert scores["mts"].false_positives == 1
    assert scores["tele2"].hits == 0
    assert scores["tele2"].false_positives == 0
    assert scores["other"].hits == 0
    assert scores["other"].total == 2


def test_unclear_on_a_foreign_billboard_is_a_miss_but_not_an_invention() -> None:
    scores = summarize([Outcome("a.jpg", truth="other", decided="unclear")], targets=TARGETS)

    assert scores["other"].hits == 0
    assert sum(item.false_positives for item in scores.values()) == 0


def test_comparison_names_what_was_fixed_and_what_broke(caplog) -> None:
    before = [
        Outcome("fixed.jpg", truth="mts", decided="unclear"),
        Outcome("broken.jpg", truth="tele2", decided="tele2"),
        Outcome("same.jpg", truth="other", decided="other"),
    ]
    after = [
        Outcome("fixed.jpg", truth="mts", decided="mts"),
        Outcome("broken.jpg", truth="tele2", decided="other"),
        Outcome("same.jpg", truth="other", decided="other"),
    ]

    with caplog.at_level(logging.INFO, logger="cls"):
        print_comparison(before, after, targets=TARGETS)

    text = caplog.text
    assert "исправлено" in text and "fixed.jpg" in text
    assert "СЛОМАНО" in text and "broken.jpg" in text
    assert "same.jpg" not in text
