"""Проверка псевдоразметки: вердикты, применение, слияние прогонов, импорт из CVAT."""

from __future__ import annotations

import csv
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from adlearn.detection import review
from adlearn.detection.config import ReviewConfig


def frame(directory: Path, name: str) -> Path:
    path = directory / name
    Image.new("RGB", (64, 48), "white").save(path)
    return path


def box(box_id: int, file: str, index: int = 0, *, confidence: float = 0.9) -> review.Box:
    return review.Box(
        box_id=box_id,
        file=file,
        index=index,
        center_x=0.5,
        center_y=0.5,
        width=0.2,
        height=0.2,
        confidence=confidence,
    )


def answer(box_id: int, category: str) -> review.Answer:
    return review.Answer(
        box_id=box_id, crop=f"{box_id}.jpg", category=category, reason="", seconds=1.0
    )


def write_answers(path: Path, answers: list[review.Answer]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(review.ANSWER_FIELDS))
        writer.writeheader()
        for item in answers:
            writer.writerow(
                {
                    "box_id": item.box_id,
                    "crop": item.crop,
                    "category": item.category,
                    "reason": item.reason,
                    "seconds": item.seconds,
                }
            )


@pytest.fixture
def config(tmp_path: Path) -> ReviewConfig:
    source = tmp_path / "raw"
    source.mkdir()
    for name in ("a.jpg", "b.jpg", "c.jpg", "d.jpg", "e.jpg"):
        frame(source, name)
    return ReviewConfig(source=source, output=tmp_path / "review")


class TestDecide:
    def test_human_word_beats_the_model(self):
        verdicts = {1: review.VERDICT_KEEP}
        fate = review.decide(box=box(1, "a.jpg"), answer=answer(1, "road_sign"), verdicts=verdicts)

        assert fate == review.VERDICT_KEEP

    def test_model_category_decides_when_nobody_looked(self):
        assert (
            review.decide(box=box(1, "a.jpg"), answer=answer(1, "vehicle_ad"), verdicts={})
            == review.VERDICT_KEEP
        )
        assert (
            review.decide(box=box(1, "a.jpg"), answer=answer(1, "building"), verdicts={})
            == review.VERDICT_DROP
        )

    def test_no_answer_is_disputed(self):
        assert (
            review.decide(box=box(1, "a.jpg"), answer=None, verdicts={}) == review.VERDICT_DISPUTED
        )


class TestApplyVerdicts:
    def test_frames_get_their_fate(self, config: ReviewConfig):
        """a — чистый, b — поправлен, c — негатив, d — спорный, e — без рамок."""

        config.output.mkdir()
        review.write_boxes(
            boxes=[
                box(1, "a.jpg"),
                box(2, "b.jpg", 0),
                box(3, "b.jpg", 1),
                box(4, "c.jpg"),
                box(5, "d.jpg"),
            ],
            path=config.boxes_path,
        )
        write_answers(
            config.answers_path,
            [
                answer(1, "ad_surface"),
                answer(2, "ad_surface"),
                answer(3, "road_sign"),
                answer(4, "building"),
                answer(5, "building"),
            ],
        )
        config.verdicts_path.write_text(
            "box_id,verdict,note\n5,спорно,лайтбокс\n", encoding="utf-8"
        )

        counts = review.apply_verdicts(config)

        assert (counts.clean, counts.fixed, counts.negative, counts.disputed, counts.no_boxes) == (
            1,
            1,
            1,
            1,
            1,
        )
        assert (config.labels_dir / "c.txt").read_text() == ""
        assert len((config.labels_dir / "b.txt").read_text().splitlines()) == 1
        assert not (config.labels_dir / "d.txt").exists()
        assert config.cvat_list_path.read_text().split() == ["d.jpg", "e.jpg"]

    def test_stride_turns_empty_frames_into_negatives(self, config: ReviewConfig):
        config = ReviewConfig(source=config.source, output=config.output, empty_frame_stride=2)
        config.output.mkdir()
        review.write_boxes(boxes=[box(1, "a.jpg")], path=config.boxes_path)
        write_answers(config.answers_path, [answer(1, "ad_surface")])

        counts = review.apply_verdicts(config)

        assert counts.negative == 2
        assert counts.no_boxes == 2
        assert config.cvat_list_path.read_text() == ""


class TestMergeScans:
    def test_only_new_boxes_are_added(self, tmp_path: Path):
        primary = ReviewConfig(source=tmp_path / "raw", output=tmp_path / "one")
        secondary = ReviewConfig(source=tmp_path / "raw", output=tmp_path / "two")
        for item in (primary, secondary):
            item.crops_dir.mkdir(parents=True)
        review.write_boxes(boxes=[box(1, "a.jpg")], path=primary.boxes_path)
        twin = box(1, "a.jpg")
        other = review.Box(
            box_id=2,
            file="a.jpg",
            index=1,
            center_x=0.1,
            center_y=0.1,
            width=0.1,
            height=0.1,
            confidence=0.5,
        )
        review.write_boxes(boxes=[twin, other], path=secondary.boxes_path)
        (secondary.crops_dir / other.crop).write_bytes(b"jpeg")

        added = review.merge_scans(primary=primary, secondary=secondary)

        assert added == 1
        merged = review.read_boxes_csv(primary.boxes_path)
        assert [item.box_id for item in merged] == [1, 2]
        assert (primary.crops_dir / merged[1].crop).exists()


class TestImportExport:
    def test_cvat_export_replaces_labels_and_shortens_the_queue(self, config: ReviewConfig):
        config.labels_dir.mkdir(parents=True)
        config.cvat_list_path.write_text("d.jpg\ne.jpg\n", encoding="utf-8")
        with config.frames_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(review.FRAME_FIELDS))
            writer.writeheader()
            writer.writerow(
                {
                    "file": "d.jpg",
                    "status": review.FRAME_DISPUTED,
                    "boxes_before": 1,
                    "boxes_after": 0,
                    "note": "",
                }
            )
        archive = config.output / "export.zip"
        with zipfile.ZipFile(archive, "w") as bundle:
            bundle.writestr("obj.names", "ad_object\n")
            bundle.writestr("obj_train_data/d.txt", "0 0.5 0.5 0.2 0.2\n")

        frames, boxes, empty = review.import_export(config, archive=archive)

        assert (frames, boxes, empty) == (1, 1, 0)
        assert (config.labels_dir / "d.txt").read_text().startswith("0 0.5")
        assert config.cvat_list_path.read_text().split() == ["e.jpg"]
        rows = list(csv.DictReader(config.frames_path.open(encoding="utf-8")))
        assert rows[0]["status"] == review.FRAME_FROM_CVAT
