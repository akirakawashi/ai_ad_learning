"""Настройки детекции. Все числа живут здесь, как в соседнем ai_ad_ml."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from adlearn import paths

CLASS_NAME = "ad_object"

TASK = paths.DETECTION


@dataclass(frozen=True)
class PrelabelConfig:
    """Прогон папки фотографий через детектор щитов.

    `image_size` и `iou` повторяют боевые значения из `DetectionConfig`
    пайплайна: разметка должна получиться такой же, какую увидит рабочая модель,
    иначе правится не то.

    А вот `confidence_min` намеренно ниже боевого порога 0.50. Стереть лишнюю
    рамку в разметчике — одно движение, нарисовать пропущенную — десять. Всё, что
    слабее `confidence_weak`, помечается в отчёте как требующее взгляда: сам
    YOLO-формат уверенность не хранит, и после импорта в CVAT рамка на 0.26
    выглядит ровно так же, как рамка на 0.95.
    """

    weights: Path = TASK.weights
    source: Path = paths.RAW
    output: Path = TASK.prelabel
    class_name: str = CLASS_NAME
    image_size: int = 960
    confidence_min: float = 0.25
    confidence_weak: float = 0.50
    iou: float = 0.50
    batch_size: int = 16
    device: str | None = "0"

    @property
    def labels_dir(self) -> Path:
        return self.output / "labels"

    @property
    def images_dir(self) -> Path:
        return self.output / "images"

    @property
    def report_path(self) -> Path:
        return self.output / "report.csv"

    @property
    def review_path(self) -> Path:
        return self.output / "review.txt"

    @property
    def cvat_archive(self) -> Path:
        return self.output / "cvat_annotations.zip"


@dataclass(frozen=True)
class DatasetConfig:
    """Сборка набора из ручной разметки и уверенной псевдоразметки."""

    export_archive: Path
    prelabel: Path = TASK.prelabel
    output: Path = TASK.dataset
    class_name: str = CLASS_NAME
    validation_share: float = 0.15
    test_share: float = 0.15
    seed: int = 0


@dataclass(frozen=True)
class TrainConfig:
    """Настройки обучения.

    `weights` по умолчанию — предобученная `yolo11m`, а не текущая рабочая
    модель. Дообучать модель на разметке, которую она же и поставила, почти
    бессмысленно: она подтвердит собственные ответы и закрепит собственные
    промахи. Новый набор приносит другие ракурсы и форматы щитов, и им нужен вес,
    а не поправка к старым весам. Продолжить с боевых весов можно, передав их
    явно, — тогда обучение станет коротким дообучением.

    `image_size` повторяет боевое значение: модель, обученная в другом
    разрешении, в пайплайне поведёт себя иначе.
    """

    data: Path = field(default_factory=lambda: TASK.dataset / "data.yaml")
    weights: str = str(paths.PRETRAINED / "yolo11m.pt")
    epochs: int = 120
    image_size: int = 960
    batch: int = 10
    patience: int = 20
    learning_rate: float = 0.001
    device: str = "0"
    project: Path = paths.RUNS
    name: str = "ad_object_v2"
    seed: int = 0


@dataclass(frozen=True)
class ReviewConfig:
    """Проверка псевдоразметки перед сборкой набора.

    `weights` — та же модель, чью разметку проверяем: первая, размеченная руками
    владельца. `confidence_min` ниже боевого 0.50, потому что проверка тем и
    занимается, что отделяет годное от мусора, и слабые рамки ей нужны.

    `crop_max_side` держит вырезку мелкой намеренно. Щит на вырезке занимает
    почти весь кадр, и полтысячи пикселей ему хватает; крупная картинка
    разворачивается в тысячи токенов и замедляет ответ вчетверо.

    `empty_frame_stride` решает судьбу кадров, где детектор не нашёл ничего.
    Ноль означает «отдать человеку»: на стоковых фото такие кадры почти всегда
    прячут пропущенный щит. На записи с регистратора наоборот, пустой кадр
    честно пуст, и каждый `stride`-й из них становится негативом. Не все подряд,
    потому что таких кадров больше, чем кадров со щитами, а негативов в наборе
    должно быть около десятой части.
    """

    weights: Path = TASK.weights
    source: Path = paths.RAW
    output: Path = TASK.root / "review"
    image_size: int = 960
    confidence_min: float = 0.25
    iou: float = 0.50
    batch_size: int = 16
    device: str | None = "0"
    crop_margin: float = 0.06
    crop_max_side: int = 640
    vlm_url: str = "http://127.0.0.1:8080"
    vlm_timeout_sec: float = 120.0
    vlm_max_tokens: int = 120
    sheet_columns: int = 8
    sheet_rows: int = 6
    sheet_tile: int = 210
    frame_columns: int = 4
    frame_rows: int = 3
    frame_side: int = 420
    empty_frame_stride: int = 0

    @property
    def crops_dir(self) -> Path:
        return self.output / "crops"

    @property
    def sheets_dir(self) -> Path:
        return self.output / "sheets"

    @property
    def boxes_path(self) -> Path:
        return self.output / "boxes.csv"

    @property
    def answers_path(self) -> Path:
        return self.output / "answers.csv"

    @property
    def frames_path(self) -> Path:
        return self.output / "frames.csv"

    @property
    def verdicts_path(self) -> Path:
        return self.output / "verdicts.csv"

    @property
    def labels_dir(self) -> Path:
        return self.output / "labels"

    @property
    def cvat_list_path(self) -> Path:
        return self.output / "for_cvat.txt"


ARCHIVE = Path("/home/shiawase/ic8_ai/other/ml_archive/data/detection/yolo")
"""Ручная разметка владельца, на которой обучалась первая модель."""

DASHCAM_NEGATIVE_STRIDE = 5
"""Какую долю пустых кадров с регистратора брать негативами.

Кадров, где детектор не нашёл ничего, на записи больше, чем кадров со щитами.
Взять их все значит перекосить набор: негативов станет пятая часть, и модель
начнёт молчать там, где щит есть. Каждый пятый даёт около 15% негативов, а
кадры, где детектор реально ошибался, входят все до одного.
"""


@dataclass(frozen=True)
class BuildConfig:
    """Состав набора v3: четыре источника и отложенная съёмка.

    Фотографии фур приходят без разметки: рекламы на них нет вовсе, и файл
    разметки создаётся пустым. Записи `VideoProject` в делении не участвуют,
    они лежат отдельной частью и служат честной проверкой.
    """

    output: Path = TASK.dataset
    class_name: str = CLASS_NAME
    validation_share: float = 0.15
    test_share: float = 0.15
    seed: int = 0
    negative_stride: int = DASHCAM_NEGATIVE_STRIDE
