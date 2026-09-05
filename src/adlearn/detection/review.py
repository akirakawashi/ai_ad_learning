"""Проверка псевдоразметки: каждой рамке — вердикт, каждому кадру — статус.

Набор для второй модели собрали из разметки, которую поставила первая, и никто
её не смотрел. Так модель подтверждает собственные ответы: свои промахи она
закрепляет, а кадров без щитов в наборе не оказывается вовсе — их отбросили как
«пустые». На видео это выходит боком, каждый третий видимый объект оказывается
не рекламой.

Проход устроен в два шага. Сначала детектор заново размечает папку, и рядом с
каждой рамкой сохраняется вырезка и уверенность — YOLO-формат уверенность не
хранит, а без неё нечем упорядочить проверку. Потом VLM отвечает по каждой
вырезке, что на ней: рекламная плоскость, дорожный знак, стена, машина. Человек
после этого смотрит не десять тысяч рамок, а спорные — и кадры, где рамок не
нашлось совсем.

Ответ VLM здесь не приговор, а сортировка. Мелкую вывеску она путает со стеной,
и последнее слово остаётся за глазами; зато очевидный мусор она снимает сама.
"""

from __future__ import annotations

import base64
import csv
import json
import logging
import time
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

from adlearn.core.images import find_images
from adlearn.detection.config import ReviewConfig
from adlearn.detection.labels import CVAT_OBJ_DATA

logger = logging.getLogger("review")

CATEGORY_AD = "ad_surface"
CATEGORY_STOREFRONT = "storefront_sign"
CATEGORY_VEHICLE_AD = "vehicle_ad"
CATEGORY_ROAD_SIGN = "road_sign"
CATEGORY_BUILDING = "building"
CATEGORY_VEHICLE = "vehicle"
CATEGORY_BLUR = "nature_or_blur"
CATEGORY_OTHER = "other"
CATEGORY_ERROR = "error"

CATEGORIES = (
    CATEGORY_AD,
    CATEGORY_STOREFRONT,
    CATEGORY_VEHICLE_AD,
    CATEGORY_ROAD_SIGN,
    CATEGORY_BUILDING,
    CATEGORY_VEHICLE,
    CATEGORY_BLUR,
    CATEGORY_OTHER,
)

KEEP_CATEGORIES = (CATEGORY_AD, CATEGORY_STOREFRONT, CATEGORY_VEHICLE_AD)
"""Что считается рекламной поверхностью: щиты, вывески и реклама на транспорте.

Реклама на бортах автобусов и фургонов идёт в набор по решению владельца от
04.09.2026, табло цен на заправках — нет. Дорожных указателей в его разметке
нет ни одного. Остальные категории — повод посмотреть глазами, а не приговор:
рекламный указатель с телефоном модель то и дело зовёт дорожным знаком.
"""

PROMPT = (
    "На картинке — вырезка из фотографии улицы. Детектор принял её за рекламную "
    "поверхность. Скажи, что на ней на самом деле.\n\n"
    "ad_surface — рекламная плоскость: щит, билборд, баннер, сити-формат, пилон, стела, "
    "растяжка, афиша, постер, доска объявлений, рекламный экран. Пустая или белая плоскость "
    "на рекламной конструкции тоже подходит, как и щит с надписью «место свободно» и "
    "телефоном рекламного агентства.\n"
    "storefront_sign — вывеска магазина, кафе, салона, аптеки на фасаде здания.\n"
    "vehicle_ad — реклама на борту автобуса, троллейбуса, трамвая, такси, фургона, грузовика.\n"
    "road_sign — дорожный знак, указатель направления, табличка с названием улицы или "
    "города, туристический или служебный указатель.\n"
    "building — стена, окно, балкон, дверь, забор, решётка, крыша, где рекламы нет.\n"
    "vehicle — транспорт без рекламы: кузов, стекло, фара, фонарь, номер.\n"
    "nature_or_blur — небо, деревья, дорога, асфальт или размытое пятно, где ничего не "
    "разобрать.\n"
    "other — что-то ещё.\n\n"
    "Видно рекламную плоскость целиком или почти целиком — выбирай ad_surface, даже если "
    "картинка мелкая и мутная, а надпись не читается. Рекламы нет — честно выбирай другую "
    "категорию. Ответь JSON: category и короткое reason по-русски."
)

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "category": {"type": "string", "enum": list(CATEGORIES)},
        "reason": {"type": "string"},
    },
    "required": ["category", "reason"],
    "additionalProperties": False,
}

BOX_FIELDS = (
    "box_id",
    "file",
    "index",
    "center_x",
    "center_y",
    "width",
    "height",
    "confidence",
    "area_share",
    "crop",
)

ANSWER_FIELDS = ("box_id", "crop", "category", "reason", "seconds")

VERDICT_KEEP = "оставить"
VERDICT_DROP = "убрать"


@dataclass(frozen=True)
class Box:
    """Рамка детектора: доли кадра, уверенность и имя вырезки."""

    box_id: int
    file: str
    index: int
    center_x: float
    center_y: float
    width: float
    height: float
    confidence: float

    @property
    def area_share(self) -> float:
        return self.width * self.height

    @property
    def crop(self) -> str:
        return f"{Path(self.file).stem}__{self.index}.jpg"


@dataclass(frozen=True)
class Answer:
    """Ответ модели по одной вырезке."""

    box_id: int
    crop: str
    category: str
    reason: str
    seconds: float


def batched(paths: list[Path], size: int) -> Iterator[list[Path]]:
    for start in range(0, len(paths), size):
        yield paths[start : start + size]


def cut(*, frame: np.ndarray, box: Box, margin: float, max_side: int) -> np.ndarray:
    """Вырезка по рамке с полями вокруг, уменьшенная под отправку."""

    height, width = frame.shape[:2]
    half_width = box.width / 2 + box.width * margin
    half_height = box.height / 2 + box.height * margin
    x1 = int(max(0.0, (box.center_x - half_width) * width))
    x2 = int(min(float(width), (box.center_x + half_width) * width))
    y1 = int(max(0.0, (box.center_y - half_height) * height))
    y2 = int(min(float(height), (box.center_y + half_height) * height))
    piece = frame[max(0, y1) : max(1, y2), max(0, x1) : max(1, x2)]
    if piece.size == 0:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    scale = max_side / max(piece.shape[0], piece.shape[1])
    if scale < 1.0:
        piece = cv2.resize(
            piece,
            (max(1, round(piece.shape[1] * scale)), max(1, round(piece.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return piece


def scan(config: ReviewConfig) -> list[Box]:
    """Прогон детектора по папке: рамки с уверенностью и вырезка на каждую."""

    images = find_images(config.source)
    if not images:
        raise FileNotFoundError(f"В {config.source} нет фотографий.")
    if not config.weights.exists():
        raise FileNotFoundError(config.weights)

    model = YOLO(str(config.weights))
    predict_kwargs: dict[str, object] = {
        "conf": config.confidence_min,
        "iou": config.iou,
        "imgsz": config.image_size,
        "verbose": False,
    }
    if config.device is not None:
        predict_kwargs["device"] = config.device

    config.crops_dir.mkdir(parents=True, exist_ok=True)
    boxes: list[Box] = []
    empty_frames = 0
    seen_frames = 0
    for chunk in batched(images, config.batch_size):
        results = cast(Any, model).predict([str(path) for path in chunk], **predict_kwargs)
        for path, result in zip(chunk, results, strict=True):
            height, width = result.orig_shape
            frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
            found = 0
            if result.boxes is not None and frame is not None:
                for index, raw in enumerate(result.boxes):
                    x1, y1, x2, y2 = (float(value) for value in raw.xyxy[0].detach().cpu().tolist())
                    box = Box(
                        box_id=len(boxes) + 1,
                        file=path.name,
                        index=index,
                        center_x=(x1 + x2) / 2 / width,
                        center_y=(y1 + y2) / 2 / height,
                        width=(x2 - x1) / width,
                        height=(y2 - y1) / height,
                        confidence=float(raw.conf[0]),
                    )
                    piece = cut(
                        frame=frame,
                        box=box,
                        margin=config.crop_margin,
                        max_side=config.crop_max_side,
                    )
                    cv2.imwrite(str(config.crops_dir / box.crop), piece)
                    boxes.append(box)
                    found += 1
            if found == 0:
                empty_frames += 1
            seen_frames += 1
        logger.info("кадров %s из %s, рамок %s", seen_frames, len(images), len(boxes))

    write_boxes(boxes=boxes, path=config.boxes_path)
    logger.info(
        "кадров %s, рамок %s, кадров без рамок %s",
        len(images),
        len(boxes),
        empty_frames,
    )
    return boxes


def write_boxes(*, boxes: list[Box], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(BOX_FIELDS))
        writer.writeheader()
        for box in boxes:
            writer.writerow(
                {
                    "box_id": box.box_id,
                    "file": box.file,
                    "index": box.index,
                    "center_x": f"{box.center_x:.6f}",
                    "center_y": f"{box.center_y:.6f}",
                    "width": f"{box.width:.6f}",
                    "height": f"{box.height:.6f}",
                    "confidence": f"{box.confidence:.4f}",
                    "area_share": f"{box.area_share:.6f}",
                    "crop": box.crop,
                }
            )


def read_boxes_csv(path: Path) -> list[Box]:
    with path.open(encoding="utf-8") as handle:
        return [
            Box(
                box_id=int(row["box_id"]),
                file=row["file"],
                index=int(row["index"]),
                center_x=float(row["center_x"]),
                center_y=float(row["center_y"]),
                width=float(row["width"]),
                height=float(row["height"]),
                confidence=float(row["confidence"]),
            )
            for row in csv.DictReader(handle)
        ]


def ask(crop: Path, *, config: ReviewConfig) -> tuple[str, str]:
    """Одна вырезка — одна категория. Сбой уходит в отчёт, прогон не рвётся."""

    payload = base64.b64encode(crop.read_bytes()).decode("ascii")
    body: dict[str, Any] = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{payload}"},
                    },
                    {"type": "text", "text": PROMPT},
                ],
            }
        ],
        "temperature": 0.0,
        "max_tokens": config.vlm_max_tokens,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "surface", "schema": SCHEMA, "strict": True},
        },
    }
    try:
        response = requests.post(
            f"{config.vlm_url.rstrip('/')}/v1/chat/completions",
            json=body,
            timeout=config.vlm_timeout_sec,
        )
        response.raise_for_status()
        parsed = json.loads(response.json()["choices"][0]["message"]["content"])
    except Exception as error:  # noqa: BLE001 — причина уходит в отчёт, прогон не рвём
        return CATEGORY_ERROR, str(error)[:200]
    return parsed.get("category", CATEGORY_ERROR), parsed.get("reason", "")


def judge(config: ReviewConfig) -> int:
    """Спрашивает модель про каждую вырезку. Уже отвеченные пропускаются."""

    boxes = read_boxes_csv(config.boxes_path)
    done = {answer.box_id for answer in read_answers(config.answers_path)}
    todo = [box for box in boxes if box.box_id not in done]
    logger.info("рамок %s, уже отвечено %s, осталось %s", len(boxes), len(done), len(todo))

    fresh = config.answers_path.exists()
    config.answers_path.parent.mkdir(parents=True, exist_ok=True)
    with config.answers_path.open("a" if fresh else "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ANSWER_FIELDS))
        if not fresh:
            writer.writeheader()
        for number, box in enumerate(todo, 1):
            started = time.monotonic()
            category, reason = ask(config.crops_dir / box.crop, config=config)
            writer.writerow(
                {
                    "box_id": box.box_id,
                    "crop": box.crop,
                    "category": category,
                    "reason": reason.replace("\n", " "),
                    "seconds": f"{time.monotonic() - started:.1f}",
                }
            )
            handle.flush()
            if number % 200 == 0:
                logger.info("отвечено %s из %s", number, len(todo))
    return len(todo)


def read_answers(path: Path) -> list[Answer]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [
            Answer(
                box_id=int(row["box_id"]),
                crop=row["crop"],
                category=row["category"],
                reason=row["reason"],
                seconds=float(row["seconds"]),
            )
            for row in csv.DictReader(handle)
        ]


def sheet_of_crops(
    *,
    crops: list[tuple[str, Path]],
    path: Path,
    columns: int,
    tile: int,
) -> None:
    """Контактный лист из вырезок: подпись сверху, картинка под ней."""

    caption = 20
    rows = (len(crops) + columns - 1) // columns
    board = Image.new("RGB", (columns * tile, max(1, rows) * (tile + caption)), "white")
    pen = ImageDraw.Draw(board)
    font = ImageFont.load_default(size=14)
    for position, (label, source) in enumerate(crops):
        try:
            piece = Image.open(source).convert("RGB")
        except OSError:
            continue
        piece.thumbnail((tile, tile))
        left = (position % columns) * tile
        top = (position // columns) * (tile + caption)
        board.paste(piece, (left + (tile - piece.width) // 2, top + caption))
        pen.text((left + 3, top + 3), label, fill="black", font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    board.save(path)


def sheet_of_frames(
    *,
    frames: list[tuple[str, Path, list[tuple[float, float, float, float]]]],
    path: Path,
    columns: int,
    rows: int,
    side: int,
) -> None:
    """Лист из целых кадров с нарисованными рамками — так виден пропущенный щит."""

    caption = 20
    board = Image.new("RGB", (columns * side, rows * (side + caption)), "white")
    pen = ImageDraw.Draw(board)
    font = ImageFont.load_default(size=15)
    for position, (label, source, boxes) in enumerate(frames):
        try:
            frame = Image.open(source).convert("RGB")
        except OSError:
            continue
        width, height = frame.size
        drawer = ImageDraw.Draw(frame)
        for center_x, center_y, box_width, box_height in boxes:
            x1 = (center_x - box_width / 2) * width
            y1 = (center_y - box_height / 2) * height
            x2 = (center_x + box_width / 2) * width
            y2 = (center_y + box_height / 2) * height
            drawer.rectangle((x1, y1, x2, y2), outline=(255, 40, 40), width=max(2, width // 300))
        frame.thumbnail((side, side))
        left = (position % columns) * side
        top = (position // columns) * (side + caption)
        board.paste(frame, (left + (side - frame.width) // 2, top + caption))
        pen.text((left + 3, top + 3), label, fill="black", font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    board.save(path)


def crop_sheets(
    *,
    boxes: list[Box],
    config: ReviewConfig,
    name: str,
) -> int:
    """Режет список рамок на листы по `sheet_columns × sheet_rows`."""

    per_sheet = config.sheet_columns * config.sheet_rows
    sheets = 0
    for start in range(0, len(boxes), per_sheet):
        chunk = boxes[start : start + per_sheet]
        sheet_of_crops(
            crops=[
                (f"{box.box_id} {box.confidence:.2f}", config.crops_dir / box.crop) for box in chunk
            ],
            path=config.sheets_dir / f"{name}_{start // per_sheet + 1:03d}.png",
            columns=config.sheet_columns,
            tile=config.sheet_tile,
        )
        sheets += 1
    return sheets


def frame_sheets(
    *,
    files: list[str],
    boxes_by_file: dict[str, list[tuple[float, float, float, float]]],
    config: ReviewConfig,
    name: str,
) -> int:
    """Листы целых кадров: имя кадра в подписи, рамки поверх картинки."""

    per_sheet = config.frame_columns * config.frame_rows
    sheets = 0
    for start in range(0, len(files), per_sheet):
        chunk = files[start : start + per_sheet]
        sheet_of_frames(
            frames=[
                (item[:16], config.source / item, boxes_by_file.get(item, [])) for item in chunk
            ],
            path=config.sheets_dir / f"{name}_{start // per_sheet + 1:03d}.png",
            columns=config.frame_columns,
            rows=config.frame_rows,
            side=config.frame_side,
        )
        sheets += 1
    return sheets


def frames_without_boxes(*, source: Path, boxes: list[Box]) -> list[str]:
    """Кадры, на которых детектор не нашёл ничего: там ищут пропущенный щит."""

    seen = {box.file for box in boxes}
    return [path.name for path in find_images(source) if path.name not in seen]


VERDICT_DISPUTED = "спорно"

FRAME_CLEAN = "чисто"
FRAME_FIXED = "поправлен"
FRAME_NEGATIVE = "негатив"
FRAME_DISPUTED = "в CVAT"
FRAME_NO_BOXES = "без рамок"

FRAME_FIELDS = ("file", "status", "boxes_before", "boxes_after", "note")


@dataclass(frozen=True)
class ApplyCounts:
    """Сколько кадров какой судьбы после применения вердиктов."""

    clean: int
    fixed: int
    negative: int
    disputed: int
    no_boxes: int
    boxes_before: int
    boxes_after: int


def read_verdicts(path: Path) -> dict[int, str]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        return {int(row["box_id"]): row["verdict"] for row in csv.DictReader(handle)}


def decide(*, box: Box, answer: Answer | None, verdicts: dict[int, str]) -> str:
    """Судьба рамки: слово человека важнее ответа модели."""

    human = verdicts.get(box.box_id)
    if human:
        return human
    if answer is None:
        return VERDICT_DISPUTED
    return VERDICT_KEEP if answer.category in KEEP_CATEGORIES else VERDICT_DROP


def apply_verdicts(config: ReviewConfig) -> ApplyCounts:
    """Складывает вычищенную разметку рядом, не трогая исходную.

    Кадр со спорной рамкой в разметку не попадает: за него решает человек в
    CVAT, и подставлять туда догадку вреднее, чем оставить кадр в стороне. Так
    же поступаем с кадрами, где детектор не нашёл ничего: на проверке оказалось,
    что реклама там почти всегда есть, просто модель её не увидела, и записать
    такой кадр пустым значит научить её и дальше не видеть пустые щиты.
    """

    boxes = read_boxes_csv(config.boxes_path)
    answers = {item.box_id: item for item in read_answers(config.answers_path)}
    verdicts = read_verdicts(config.verdicts_path)

    grouped: dict[str, list[Box]] = {}
    for box in boxes:
        grouped.setdefault(box.file, []).append(box)

    labels_dir = config.labels_dir
    labels_dir.mkdir(parents=True, exist_ok=True)
    for stale in labels_dir.glob("*.txt"):
        stale.unlink()

    counts = {
        FRAME_CLEAN: 0,
        FRAME_FIXED: 0,
        FRAME_NEGATIVE: 0,
        FRAME_DISPUTED: 0,
        FRAME_NO_BOXES: 0,
    }
    before = after = 0
    empty_seen = 0
    rows: list[dict[str, object]] = []
    for path in find_images(config.source):
        found = grouped.get(path.name, [])
        before += len(found)
        if not found:
            empty_seen += 1
            stride = config.empty_frame_stride
            if stride and empty_seen % stride == 0:
                (labels_dir / f"{path.stem}.txt").write_text("", encoding="utf-8")
                counts[FRAME_NEGATIVE] += 1
                rows.append(
                    {
                        "file": path.name,
                        "status": FRAME_NEGATIVE,
                        "boxes_before": 0,
                        "boxes_after": 0,
                        "note": "детектор не нашёл ничего, кадр взят негативом",
                    }
                )
                continue
            counts[FRAME_NO_BOXES] += 1
            rows.append(
                {
                    "file": path.name,
                    "status": FRAME_NO_BOXES,
                    "boxes_before": 0,
                    "boxes_after": 0,
                    "note": "детектор не нашёл ничего, кадр смотрит человек"
                    if not stride
                    else "детектор не нашёл ничего, кадр пропущен",
                }
            )
            continue

        fates = [
            decide(box=box, answer=answers.get(box.box_id), verdicts=verdicts) for box in found
        ]
        if VERDICT_DISPUTED in fates:
            counts[FRAME_DISPUTED] += 1
            rows.append(
                {
                    "file": path.name,
                    "status": FRAME_DISPUTED,
                    "boxes_before": len(found),
                    "boxes_after": 0,
                    "note": "спорная рамка, кадр уходит в CVAT",
                }
            )
            continue

        kept = [box for box, fate in zip(found, fates, strict=True) if fate == VERDICT_KEEP]
        after += len(kept)
        lines = [
            f"0 {box.center_x:.6f} {box.center_y:.6f} {box.width:.6f} {box.height:.6f}"
            for box in kept
        ]
        (labels_dir / f"{path.stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )
        if not kept:
            status = FRAME_NEGATIVE
        elif len(kept) == len(found):
            status = FRAME_CLEAN
        else:
            status = FRAME_FIXED
        counts[status] += 1
        rows.append(
            {
                "file": path.name,
                "status": status,
                "boxes_before": len(found),
                "boxes_after": len(kept),
                "note": "",
            }
        )

    with config.frames_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(FRAME_FIELDS))
        writer.writeheader()
        writer.writerows(rows)

    cvat = [
        row["file"]
        for row in rows
        if row["status"] == FRAME_DISPUTED
        or (row["status"] == FRAME_NO_BOXES and not config.empty_frame_stride)
    ]
    config.cvat_list_path.write_text(
        "\n".join(str(name) for name in cvat) + ("\n" if cvat else ""), encoding="utf-8"
    )

    return ApplyCounts(
        clean=counts[FRAME_CLEAN],
        fixed=counts[FRAME_FIXED],
        negative=counts[FRAME_NEGATIVE],
        disputed=counts[FRAME_DISPUTED],
        no_boxes=counts[FRAME_NO_BOXES],
        boxes_before=before,
        boxes_after=after,
    )


def pack_disputed(config: ReviewConfig, *, class_name: str = "ad_object") -> tuple[Path, Path, int]:
    """Комплект для CVAT из кадров со спорными рамками.

    Все рамки едут одним классом: и решённые, и спорные. Владелец правит их
    как обычную разметку, лишние удаляет, пропущенные дорисовывает. Снятые
    рамки в комплект не попадают, они уже решены.
    """

    boxes = read_boxes_csv(config.boxes_path)
    answers = {item.box_id: item for item in read_answers(config.answers_path)}
    verdicts = read_verdicts(config.verdicts_path)
    grouped: dict[str, list[Box]] = {}
    for box in boxes:
        grouped.setdefault(box.file, []).append(box)

    frames = {}
    for file, found in grouped.items():
        fates = [
            decide(box=box, answer=answers.get(box.box_id), verdicts=verdicts) for box in found
        ]
        if VERDICT_DISPUTED not in fates:
            continue
        frames[file] = [
            f"0 {box.center_x:.6f} {box.center_y:.6f} {box.width:.6f} {box.height:.6f}"
            for box, fate in zip(found, fates, strict=True)
            if fate != VERDICT_DROP
        ]

    names = sorted(frames)
    output = config.output / "cvat"
    output.mkdir(parents=True, exist_ok=True)
    images_archive = output / "disputed_images.zip"
    annotations_archive = output / "disputed_annotations.zip"
    with zipfile.ZipFile(images_archive, "w", compression=zipfile.ZIP_STORED) as bundle:
        for name in names:
            bundle.write(config.source / name, name)
    with zipfile.ZipFile(annotations_archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr("obj.names", f"{class_name}\n")
        bundle.writestr("obj.data", CVAT_OBJ_DATA)
        bundle.writestr("train.txt", "\n".join(f"obj_train_data/{name}" for name in names) + "\n")
        for name in names:
            bundle.writestr(f"obj_train_data/{Path(name).stem}.txt", "\n".join(frames[name]) + "\n")
    return images_archive, annotations_archive, len(names)


FRAME_FROM_CVAT = "из CVAT"


def import_export(config: ReviewConfig, *, archive: Path) -> tuple[int, int, int]:
    """Вливает выгрузку CVAT в чистую разметку и снимает кадры с очереди.

    Файл из выгрузки заменяет кадр целиком: человек видел всё, что там было, и
    его версия последняя. Пустой файл из CVAT значит «рекламы нет», такой кадр
    становится негативом. Отдаёт число кадров, рамок и кадров без рамок.
    """

    with zipfile.ZipFile(archive) as bundle:
        members = [
            name
            for name in bundle.namelist()
            if name.startswith("obj_train_data/") and name.endswith(".txt")
        ]
        labels = {Path(name).stem: bundle.read(name).decode("utf-8") for name in members}

    config.labels_dir.mkdir(parents=True, exist_ok=True)
    boxes = 0
    empty = 0
    for stem, text in labels.items():
        lines = [line for line in text.splitlines() if line.strip()]
        boxes += len(lines)
        empty += not lines
        (config.labels_dir / f"{stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )

    if config.frames_path.exists():
        with config.frames_path.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            stem = Path(row["file"]).stem
            if stem in labels:
                count = len([line for line in labels[stem].splitlines() if line.strip()])
                row["status"] = FRAME_NEGATIVE if count == 0 else FRAME_FROM_CVAT
                row["boxes_after"] = str(count)
                row["note"] = f"разметка из {archive.name}"
        with config.frames_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(FRAME_FIELDS))
            writer.writeheader()
            writer.writerows(rows)

    if config.cvat_list_path.exists():
        pending = [
            name
            for name in config.cvat_list_path.read_text(encoding="utf-8").split()
            if Path(name).stem not in labels
        ]
        config.cvat_list_path.write_text(
            "\n".join(pending) + ("\n" if pending else ""), encoding="utf-8"
        )
    return len(labels), boxes, empty


def merge_scans(*, primary: ReviewConfig, secondary: ReviewConfig, iou_min: float = 0.6) -> int:
    """Добавляет к рамкам основного прогона те рамки второго, которых в нём нет.

    Две модели находят разное: на кадрах с регистратора одна ловит щит, который
    вторая пропустила. Объединение даёт полноту, а мусор из обеих потом снимет
    судья. Совпадающие рамки, перекрытие от `iou_min`, считаются одной и
    остаются в версии основного прогона.
    """

    base = read_boxes_csv(primary.boxes_path)
    extra = read_boxes_csv(secondary.boxes_path)
    by_file: dict[str, list[Box]] = {}
    for box in base:
        by_file.setdefault(box.file, []).append(box)

    def overlap(one: Box, two: Box) -> float:
        left = max(one.center_x - one.width / 2, two.center_x - two.width / 2)
        right = min(one.center_x + one.width / 2, two.center_x + two.width / 2)
        top = max(one.center_y - one.height / 2, two.center_y - two.height / 2)
        bottom = min(one.center_y + one.height / 2, two.center_y + two.height / 2)
        inter = max(0.0, right - left) * max(0.0, bottom - top)
        union = one.area_share + two.area_share - inter
        return inter / union if union else 0.0

    added = 0
    next_id = max((box.box_id for box in base), default=0) + 1
    next_index: dict[str, int] = {
        file: max(box.index for box in boxes) + 1 for file, boxes in by_file.items()
    }
    for box in extra:
        twins = by_file.get(box.file, [])
        if any(overlap(box, twin) >= iou_min for twin in twins):
            continue
        index = next_index.get(box.file, 0)
        merged = Box(
            box_id=next_id,
            file=box.file,
            index=index,
            center_x=box.center_x,
            center_y=box.center_y,
            width=box.width,
            height=box.height,
            confidence=box.confidence,
        )
        source_crop = secondary.crops_dir / box.crop
        if source_crop.exists():
            (primary.crops_dir / merged.crop).write_bytes(source_crop.read_bytes())
        by_file.setdefault(box.file, []).append(merged)
        base.append(merged)
        next_index[box.file] = index + 1
        next_id += 1
        added += 1

    write_boxes(boxes=base, path=primary.boxes_path)
    return added
