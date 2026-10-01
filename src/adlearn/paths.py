"""Где что лежит на диске.

Один модуль на весь проект: пути не разъезжаются по конфигам, и добавить вторую
задачу — это добавить сюда одну строку.

Все пути относительные, от корня репозитория. Данные целиком лежат в `data/` и
в Git не попадают; веса — в `models/`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DATA = Path("data")
MODELS = Path("models")

RAW = DATA / "raw"
"""Исходные фотографии — общие для всех задач."""

RUNS = DATA / "runs"
"""Куда ultralytics складывает прогоны обучения."""

PRETRAINED = MODELS / "pretrained"
"""Веса из коробки: `yolo11m.pt` и прочие точки старта."""


@dataclass(frozen=True)
class TaskLayout:
    """Папки одной задачи. Имя задачи — одно и то же в `data/` и в `models/`."""

    name: str

    @property
    def root(self) -> Path:
        """Возвращает корневой каталог данных задачи.

        Returns:
            Путь `data/<задача>`.
        """
        return DATA / self.name

    @property
    def prelabel(self) -> Path:
        """Возвращает каталог результата псевдоразметки.

        Returns:
            Путь к данным, которые уходят человеку на правку.
        """
        return self.root / "prelabel"

    @property
    def dataset(self) -> Path:
        """Возвращает каталог собранного набора.

        Returns:
            Путь к набору, разделённому на обучение, проверку и тест.
        """
        return self.root / "dataset"

    @property
    def export(self) -> Path:
        """Возвращает каталог выгрузок из разметчика.

        Returns:
            Путь к выгрузкам задачи.
        """
        return self.root / "export"

    @property
    def preview(self) -> Path:
        """Возвращает каталог контактных листов.

        Returns:
            Путь к предпросмотру разметки.
        """
        return self.root / "preview"

    @property
    def weights(self) -> Path:
        """Возвращает путь к рабочим весам задачи.

        Returns:
            Путь к файлу `best.pt` в каталоге моделей.
        """
        return MODELS / self.name / "best.pt"


DETECTION = TaskLayout("detection")
CLASSIFICATION = TaskLayout("classification")
