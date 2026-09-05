"""Отрицательные примеры со стороны: фуры без рекламы, удвоенные аугментацией.

Детектор принимает борт фуры за щит: белая плоскость на колёсах ему выглядит
как рекламная. Кадры с фурами и пустой разметкой учат его молчать. Владелец
собрал такие фотографии сам, 04.09.2026, и попросил удвоить их: копия с
отражением, сдвигом яркости и лёгким размытием считается вторым примером.
Ultralytics и так крутит аугментацию на лету, но удвоение поднимает вес этих
кадров в обучении, а ровно этого и хотелось.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter

from adlearn.core.images import find_images

AUGMENT_SUFFIX = "_aug"
"""Копия с этим суффиксом никогда не уходит в проверку или тест.

Оригинал и его отражение в разных частях набора это утечка: модель видит на
проверке то, что уже видела в обучении. Сборка набора оставляет такие копии
в обучении рядом с оригиналом.
"""


@dataclass(frozen=True)
class NegativesConfig:
    """Откуда брать фотографии и как их удваивать."""

    source: Path
    output: Path
    group: str = "truck"
    skip: tuple[str, ...] = ()
    brightness_range: tuple[float, float] = (0.7, 1.3)
    contrast_range: tuple[float, float] = (0.8, 1.2)
    crop_min_share: float = 0.85
    blur_max_radius: float = 1.2
    jpeg_quality: int = 90
    seed: int = 0

    @property
    def images_dir(self) -> Path:
        return self.output / "images"

    @property
    def labels_dir(self) -> Path:
        return self.output / "labels"


def augment(image: Image.Image, *, config: NegativesConfig, rng: random.Random) -> Image.Image:
    """Отражение, случайный кроп, яркость, контраст и лёгкое размытие."""

    result = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    width, height = result.size
    share = rng.uniform(config.crop_min_share, 1.0)
    crop_width, crop_height = int(width * share), int(height * share)
    left = rng.randint(0, width - crop_width)
    top = rng.randint(0, height - crop_height)
    result = result.crop((left, top, left + crop_width, top + crop_height))
    result = ImageEnhance.Brightness(result).enhance(rng.uniform(*config.brightness_range))
    result = ImageEnhance.Contrast(result).enhance(rng.uniform(*config.contrast_range))
    radius = rng.uniform(0.0, config.blur_max_radius)
    if radius > 0.3:
        result = result.filter(ImageFilter.GaussianBlur(radius))
    return result


def prepare(config: NegativesConfig) -> tuple[int, int, int]:
    """Раскладывает оригиналы и их копии с пустой разметкой. Отдаёт счётчики."""

    config.images_dir.mkdir(parents=True, exist_ok=True)
    config.labels_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(config.seed)
    originals = augmented = skipped = 0
    for number, path in enumerate(find_images(config.source), start=1):
        if path.name in config.skip:
            skipped += 1
            continue
        stem = f"{config.group}_{number:05d}"
        image = Image.open(path).convert("RGB")
        image.save(config.images_dir / f"{stem}.jpg", quality=config.jpeg_quality)
        (config.labels_dir / f"{stem}.txt").write_text("", encoding="utf-8")
        originals += 1
        copy = augment(image, config=config, rng=rng)
        copy.save(config.images_dir / f"{stem}{AUGMENT_SUFFIX}.jpg", quality=config.jpeg_quality)
        (config.labels_dir / f"{stem}{AUGMENT_SUFFIX}.txt").write_text("", encoding="utf-8")
        augmented += 1
    return originals, augmented, skipped
