# AI Ad Learning

Подготовка данных и обучение моделей AI Ad. Сейчас в работе детектор рекламных
щитов; каркас рассчитан на вторую задачу — классификацию — рядом с первой.

Соседние репозитории: `../ai_ad_ml` — пайплайн и воркер, откуда взяты веса,
`../ai_ad_backend` — канон и документация проекта.

## Как всё разложено

```text
src/adlearn/
├── paths.py            где что лежит на диске — один модуль на весь проект
├── cli.py              точка входа: adlearn <задача> <команда>
├── core/               общее для всех задач
│   ├── images.py       поиск кадров, относительные ссылки, чистые папки
│   ├── grouping.py     группы съёмки и деление на обучение / проверку / тест
│   ├── sheets.py       контактные листы
│   └── cli.py          каркас командной строки
├── detection/          детектор щитов
│   ├── config.py       все настройки задачи
│   ├── labels.py       YOLO-разметка и комплект для CVAT
│   ├── report.py       отчёт по кадрам и порядок ручной проверки
│   ├── prelabel.py     прогон фотографий через детектор
│   ├── bundle.py       пачки спорных кадров под задачи CVAT
│   ├── dataset.py      сборка набора из трёх частей
│   ├── checks.py       проверки набора перед обучением
│   ├── preview.py      разметка поверх кадров
│   ├── review.py       проверка псевдоразметки: детектор, судья-VLM, вердикты
│   ├── negatives.py    фотографии без рекламы с пустой разметкой, удвоенные копиями
│   ├── train.py        обучение и оценка
│   └── cli.py          команды adlearn detect ...
└── classification/     задача в работе, каркас готов

data/                   всё тяжёлое, в Git не попадает
├── raw/                исходные фотографии, общие для всех задач
├── dashcam/            кадры с регистратора, нарезка раз в две секунды
├── negatives/          съёмка без рекламы: фуры, автобусы, цистерны
├── detection/
│   ├── prelabel/       результат псевдоразметки
│   ├── review/         проверенная разметка: вердикты, листы, чистые labels/
│   ├── negatives/      фуры без рекламы: images/ и пустые labels/
│   ├── dataset/        собранный набор
│   ├── export/         пачки для CVAT и выгрузки из него
│   └── preview/        контактные листы
├── classification/
└── runs/               прогоны обучения

models/
├── detection/best.pt   рабочие веса, копия из ../ai_ad_ml
└── pretrained/         точки старта: yolo11m.pt и прочие
```

Ссылки внутри `data/` относительные: дерево можно перенести или переименовать
целиком, и ничего не отвалится.

## Настройка

```bash
uv sync
cp ../ai_ad_ml/models/detection/best.pt models/detection/best.pt
```

Нужны Python 3.12, `uv`, GPU (без неё прогон идёт на процессоре и занимает часы)
и фотографии в `data/raw/`.

Версия `ultralytics` закреплена ровно той, что стоит в пайплайне. Разметка должна
получиться такой же, какую увидит рабочая модель, иначе правится не то.

## Порядок работы

```bash
uv run adlearn detect prelabel                       # разметить фотографии моделью
uv run adlearn detect bundle                         # собрать спорные кадры для CVAT
#                                                      ... ручная правка в CVAT ...
uv run adlearn detect build3                         # собрать набор из всех источников
uv run adlearn detect check                          # проверить набор
uv run adlearn detect preview --split train          # посмотреть глазами

uv run adlearn detect eval --weights models/detection/best.pt   # точка отсчёта
uv run adlearn detect train                                     # обучение
uv run adlearn detect eval --weights models/detection/best.pt \
    data/runs/ad_object_v2/weights/best.pt                      # сравнение
```

Псевдоразметку перед сборкой стоит проверить, а не брать на веру:

```bash
uv run adlearn detect review --stage scan                  # детектор, рамки с уверенностью
uv run adlearn detect review --stage judge                 # VLM отвечает, что на каждой рамке
uv run adlearn detect review --stage sheets                # листы для глаз
#                                                            ... вердикты в review/verdicts.csv ...
uv run adlearn detect review --stage apply                 # чистая разметка в review/labels/
uv run adlearn detect review --stage cvat                  # спорные кадры пачкой в CVAT
uv run adlearn detect review --stage import --export x.zip # выгрузка из CVAT обратно
uv run adlearn detect negatives --source ../негатив         # фуры без рекламы, по две копии
```

`uv run adlearn detect <команда> --help` покажет ключи. Подробности —
[docs/detection.md](docs/detection.md), порядок работы в CVAT —
[docs/cvat.md](docs/cvat.md).

## Классификация

```bash
uv run adlearn cls features    # эмбеддинги и цветовые признаки, один раз
uv run adlearn cls ablate      # сравнение арм: что даёт каждый сигнал
uv run adlearn cls train       # обучить голову
uv run adlearn cls predict --source ./фото
```

Разобрать ответы глазами — [notebooks/explain.ipynb](notebooks/explain.ipynb):
кадр, карта внимания и вероятности рядом. Открывается прямо в VS Code.

### Бренд через VLM

Вторая ветка классификации: зрительно-языковая модель читает надпись и узнаёт
знак, размеченная выборка ей не нужна. Модель крутится отдельно, в `llama-server`
из llama.cpp; веса лежат в `models/vlm/`. Порт 8080 занят CVAT, пока тот поднят.

```bash
~/llama.cpp/build/bin/llama-server -m models/vlm/Qwen3VL-8B-Instruct-Q4_K_M.gguf \
    --mmproj models/vlm/mmproj-Qwen3VL-8B-Instruct-F16.gguf -ngl 99 -c 8192 --port 8080

uv run adlearn cls probe --per-brand 20 --per-hard 3 --street 12 --random-other 6 \
    --output data/classification/probe/r200                 # выборка с ответами, один раз
uv run adlearn cls vlm --source data/classification/probe/r200 \
    --labels data/classification/probe/r200/labels.csv \
    --output data/classification/vlm_runs/r1.csv            # прогон, около 3.5 с на кадр
uv run adlearn cls compare data/classification/vlm_runs/r0.csv \
    data/classification/vlm_runs/r1.csv                     # что исправилось, что сломалось
```

Дефолты описывают локальную работу — свой `llama-server` на 8080 без имени модели и
ключа. Общий сервер задаётся окружением: `PIPELINE_VLM_URL`, `PIPELINE_VLM_MODEL`,
`PIPELINE_VLM_API_KEY`, те же переменные, что читает воркер пайплайна. Разово их
можно перебить флагами `--url`, `--model`, `--api-key`. Когда ключ задан, живость
проверяется через `/v1/models`, а не `/health`: иначе неверный ключ виден только
после того, как все кадры уйдут в сбой.

Бренды, которые модель знает, перечислены в `TELECOM_BRANDS`; описания знаков и
формы названий — в `classification/vlm.py`. Кадры для проверки лежат папками с
теми же именами в `data/classification/raw/`.

Рабочая копия промпта и проверки живёт в пайплайне,
`../ai_ad_ml/ml/pipeline/scripts/vlm.py`. Здесь подбирают, там применяют: после
удачного круга правку переносят туда руками, и наоборот.

Задача ещё не поставлена, но каркас под неё стоит: папки в
`adlearn.paths.CLASSIFICATION`, деление на части, ссылки и контактные листы — в
`adlearn.core`. Новая задача добавляется своим пакетом рядом с `detection` и
одной строкой `register(tasks)` в `adlearn/cli.py`.

## Проверки

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest
```
