from __future__ import annotations

from pathlib import Path

from adlearn.core.images import find_images, link, reset_dir


def test_only_images_are_found(tmp_path: Path) -> None:
    for name in ("a.jpg", "b.PNG", "c.webp", "notes.txt"):
        (tmp_path / name).write_bytes(b"")

    assert [path.name for path in find_images(tmp_path)] == ["a.jpg", "b.PNG", "c.webp"]


def test_link_is_relative_so_the_tree_can_move(tmp_path: Path) -> None:
    """Абсолютная ссылка привязала бы набор к одному пути на одной машине."""

    source = tmp_path / "raw" / "frame.jpg"
    source.parent.mkdir()
    source.write_bytes(b"frame")

    link(source=source, destination=tmp_path / "dataset" / "images" / "train" / "frame.jpg")

    made = tmp_path / "dataset" / "images" / "train" / "frame.jpg"
    assert not Path(made.readlink()).is_absolute()
    assert made.read_bytes() == b"frame"

    (tmp_path / "dataset").rename(tmp_path / "moved")
    assert (tmp_path / "moved" / "images" / "train" / "frame.jpg").read_bytes() == b"frame"


def test_link_replaces_what_was_there(tmp_path: Path) -> None:
    old = tmp_path / "old.jpg"
    new = tmp_path / "new.jpg"
    old.write_bytes(b"old")
    new.write_bytes(b"new")
    destination = tmp_path / "out" / "frame.jpg"

    link(source=old, destination=destination)
    link(source=new, destination=destination)

    assert destination.read_bytes() == b"new"


def test_reset_dir_starts_from_empty(tmp_path: Path) -> None:
    directory = tmp_path / "split"
    directory.mkdir()
    (directory / "stale.jpg").write_bytes(b"")

    assert list(reset_dir(directory).iterdir()) == []
