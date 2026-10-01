# AI Ad Learning

Подготовка данных и обучение моделей AI Ad. Основная задача — детектор рекламных щитов YOLO; каркас готов под вторую задачу — классификацию — рядом с первой.

Соседние репозитории: `../ai_ad_ml` — рабочий пайплайн, откуда берутся веса для точки отсчёта; `../ai_ad_backend` — канон проекта и его документация.

---

## Как всё разложено

```text
src/adlearn/
├── paths.py            все пути на диске — единственное место
├── cli.py              точка входа: adlearn <задача> <команда>
├── core/               общий каркас для всех задач
│   ├── images.py       поиск кадров, относительные ссылки, чистые папки
│   ├── grouping.py     деление кадров на обучение / проверку / тест по сценам
│   ├── sheets.py       контактные листы
│   └── cli.py          каркас командной строки
├── detection/          детектор щитов
│   ├── config.py       все числа задачи в одном месте
│   ├── prelabel.py     прогон папки фотографий через детектор
│   ├── bundle.py       пачки спорных кадров под задачи CVAT
│   ├── dataset.py      сборка набора из нескольких источников
│   ├── checks.py       проверка целостности набора
│   ├── preview.py      контактные листы с нарисованными рамками
│   ├── review.py       проверка псевдоразметки: детектор → VLM-судья → вердикты
│   ├── negatives.py    фотографии без рекламы с пустой разметкой, удвоенные
│   ├── train.py        обучение и оценка модели
│   └── cli.py          команды adlearn detect ...
└── classification/     задача в работе, каркас готов

data/                   всё тяжёлое — в Git не входит
├── raw/                исходные фотографии (фотографии, не кадры видео)
├── dashcam/            кадры с регистратора, нарезка раз в две секунды
├── negatives/          съёмка без рекламы: фуры, автобусы, цистерны
├── detection/
│   ├── prelabel/       результат псевдоразметки: labels/, report.csv, review.txt
│   ├── review/         проверенная разметка: labels/ после apply
│   ├── negatives/      фуры с пустой разметкой: images/ и labels/
│   ├── dataset/        собранный набор для обучения
│   ├── export/         пачки для CVAT и выгрузки из него
│   └── preview/        контактные листы
└── runs/               прогоны обучения (ultralytics)

models/
├── detection/best.pt   рабочие веса — копия из ../ai_ad_ml
└── pretrained/         точки старта: yolo11m.pt и прочие
```

Все пути внутри `data/` относительные: дерево можно перенести или переименовать, и ничего не отвалится.

---

## Настройка

```bash
uv sync
cp ../ai_ad_ml/models/detection/best.pt models/detection/best.pt
```

Нужны Python 3.12, `uv`, GPU (без неё обработка занимает часы) и фотографии в `data/raw/`.

Версия `ultralytics` закреплена совпадающей с пайплайном: разметка должна дать те же рамки, что видит рабочая модель, иначе правится не то.

---

## Команды детектора

Все команды живут под `adlearn detect`. Ключи и их значения по умолчанию: `adlearn detect <команда> --help`.

### Основной поток работы

```bash
# 1. Прогнать папку фотографий через модель
uv run adlearn detect prelabel

# 2. Собрать спорные кадры в zip-архивы для CVAT
uv run adlearn detect bundle

# --- ручная правка в CVAT ---

# 3. Собрать набор из всех источников (архив + сток + регистратор + негативы)
uv run adlearn detect build3

# 4. Проверить набор: нет ли нечитаемых файлов, пустых меток, выбросов
uv run adlearn detect check

# 5. Посмотреть глазами
uv run adlearn detect preview --split train

# 6. Оценить текущие веса как точку отсчёта
uv run adlearn detect eval --weights models/detection/best.pt

# 7. Обучить
uv run adlearn detect train

# 8. Сравнить старые и новые веса
uv run adlearn detect eval --weights models/detection/best.pt \
    data/runs/ad_object_v2/weights/best.pt
```

**Что делает каждый шаг:**

| Команда | Что происходит |
|---|---|
| `prelabel` | Прогоняет `data/raw/` через детектор. Пишет YOLO-разметку в `detection/prelabel/labels/`, отчёт с уверенностями в `report.csv` и архив для CVAT. Рамки ниже `--weak-conf` помечены в отчёте как требующие взгляда |
| `bundle` | Берёт кадры из `prelabel/`, группирует их в пачки по `--chunk` и упаковывает в zip-архивы для загрузки в CVAT |
| `build3` | Собирает набор из 4 источников: архив (ранее выгруженный из CVAT), сток (`data/raw/` + проверенные метки), регистратор (`data/dashcam/`), негативы (`data/negatives/`). Делит на train/val/test |
| `check` | Открывает каждый кадр, проверяет соответствие labels и images, считает рамки по частям, ищет подозрительные соотношения |
| `preview` | Рисует рамки поверх кадров и собирает контактные листы в `detection/preview/<split>/` |
| `eval` | Запускает YOLO-оценку на отложенной части (`test_dashcam` — кадры регистратора, которые модель не видела при обучении). Выводит mAP50, mAP50-95, precision, recall. Принимает несколько `--weights` разом |
| `train` | Запускает ultralytics training. Результаты, в том числе лучшие веса, складываются в `data/runs/<name>/` |

### Проверка псевдоразметки (review)

Вместо того чтобы брать результат `prelabel` на веру, можно прогнать его через VLM-судью: модель смотрит на каждый вырезанный кроп и отвечает, есть ли на нём рекламный щит. Это позволяет отловить ложные срабатывания до загрузки в CVAT.

```bash
uv run adlearn detect review --stage scan     # прогон детектора, сохраняет рамки с уверенностью
uv run adlearn detect review --stage judge    # VLM смотрит на каждую рамку и пишет ответ
uv run adlearn detect review --stage sheets   # контактные листы для просмотра глазами
# --- просмотреть листы, дописать вердикты в review/verdicts.csv ---
uv run adlearn detect review --stage apply    # применить вердикты → чистая разметка в review/labels/
uv run adlearn detect review --stage cvat     # спорные кадры пачкой в CVAT
uv run adlearn detect review --stage import --export x.zip  # выгрузка из CVAT обратно
```

Стадии по порядку:

| Стадия | Что происходит |
|---|---|
| `scan` | Прогоняет детектор по `--source`, пишет рамки с уверенностью и координатами в CSV |
| `judge` | Для каждой рамки вырезает кроп и отправляет в VLM (`--vlm-url`). Ответ — категория рамки: щит, не щит, спорно |
| `sheets` | Собирает контактные листы: рамки сгруппированы по категории VLM, отсортированы по уверенности |
| `apply` | Читает `review/verdicts.csv`, оставляет чистые рамки, удаляет ложные, спорные отправляет в очередь CVAT |
| `cvat` | Пакует кадры из очереди в zip-архив для CVAT |
| `import` | Разбирает выгрузку из CVAT (`--export x.zip`), добавляет исправленные метки в `review/labels/` |

Для `judge` нужен запущенный `llama-server`:

```bash
~/llama.cpp/build/bin/llama-server \
    -m ../ai_ad_learning/models/vlm/Qwen3VL-8B-Instruct-Q4_K_M.gguf \
    --mmproj ../ai_ad_learning/models/vlm/mmproj-Qwen3VL-8B-Instruct-F16.gguf \
    -ngl 99 -c 8192 --host 127.0.0.1 --port 8080
```

Порт 8080 занят CVAT, пока тот поднят.

### Негативы

Детектор путает борт фуры с рекламным щитом. Чтобы научить его молчать, нужны фотографии фур с пустой разметкой. Каждое фото удваивается: зеркало + случайный сдвиг яркости + лёгкое размытие считается вторым примером.

```bash
uv run adlearn detect negatives --source /путь/к/фотографиям/фур
```

Результат — в `data/detection/negatives/`.

---

## Классификация (каркас)

Задача ещё не поставлена, но папки, деление на части и контактные листы уже готовы. Новая задача добавляется своим пакетом рядом с `detection` и одной строкой `register(tasks)` в `adlearn/cli.py`.

Для проверки VLM-классификатора без запуска пайплайна:

```bash
uv run adlearn cls vlm --source data/classification/probe/r200 \
    --labels data/classification/probe/r200/labels.csv \
    --output data/classification/vlm_runs/r1.csv
```

Рабочая копия промпта и логика проверки живут в пайплайне (`../ai_ad_ml/ml/pipeline/scripts/vlm.py`). Здесь подбирают, там применяют: после удачного круга правку переносят вручную.

---

## Проверки

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest
```
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
