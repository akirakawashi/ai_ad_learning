from __future__ import annotations

from adlearn.core.grouping import group_of, stratified_split


def test_group_is_read_from_the_name() -> None:
    assert group_of("video_hard_000187.jpg") == "video_hard"
    assert group_of("billboard_ad_052.jpg") == "billboard_ad"
    assert group_of("3f9c1a2b8e04.jpg") == "raw"
    assert group_of("351.png") == "numbers"


def split(names: list[str], **kwargs: object) -> tuple[list[str], list[str], list[str]]:
    return stratified_split(
        names,
        stratum=group_of,
        order=str,
        **kwargs,  # type: ignore[arg-type]
    )


def test_every_group_reaches_every_part() -> None:
    names = [f"photo_{i:03d}.jpg" for i in range(20)]
    names += [f"video_hard_{i:03d}.jpg" for i in range(20)]

    for part in split(names):
        assert {group_of(name) for name in part} == {"photo", "video_hard"}


def test_split_repeats_itself() -> None:
    names = [f"photo_{i:03d}.jpg" for i in range(50)]

    assert split(names, seed=7) == split(names, seed=7)


def test_a_tiny_stratum_stays_in_training() -> None:
    """Пара кадров одного рода целиком идёт в обучение.

    Тест из одного кадра ничего не измеряет, а обучение теряет и этот кадр.
    """

    train, validation, test = split([f"rare_{i:03d}.jpg" for i in range(2)])

    assert len(train) == 2
    assert validation == []
    assert test == []
