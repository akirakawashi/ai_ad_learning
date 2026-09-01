"""Классификация — задача в работе.

Каркас под неё уже стоит:

* папки — `adlearn.paths.CLASSIFICATION` (`data/classification/…`, `models/classification/best.pt`);
* деление на части — `adlearn.core.grouping.stratified_split`, страту задаёт вызывающий;
* ссылки и чистые папки — `adlearn.core.images`;
* контактные листы — `adlearn.core.sheets`;
* команды — `register(tasks)` по образцу `adlearn.detection.cli`, подключается в `adlearn.cli`.

Модули задумывались зеркалом детекции: `config`, `dataset`, `checks`, `preview`,
`train`, `cli`. Что именно классифицируется и по каким классам — ещё не задано.
"""
