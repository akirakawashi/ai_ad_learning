from __future__ import annotations

from pathlib import Path

from adlearn.detection.bundle import chunks, files_to_review


def test_review_bundle_takes_only_what_needs_a_human(tmp_path: Path) -> None:
    """Уверенные кадры в пачки не попадают, порядок — как в списке проверки."""

    report = tmp_path / "report.csv"
    report.write_text(
        "file,width,height,boxes,min_confidence,max_confidence,status\n"
        "sure.jpg,10,10,1,0.9000,0.9000,ок\n"
        "shaky.jpg,10,10,1,0.4500,0.9000,слабая\n"
        "blank.jpg,10,10,0,0.0000,0.0000,пусто\n"
        "shakier.jpg,10,10,1,0.2600,0.9000,слабая\n",
        encoding="utf-8",
    )

    names = files_to_review(report)

    assert names == ["blank.jpg", "shakier.jpg", "shaky.jpg"]
    assert chunks(names, 2) == [["blank.jpg", "shakier.jpg"], ["shaky.jpg"]]
