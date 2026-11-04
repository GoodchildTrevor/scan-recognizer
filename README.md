# ocr-archive

Расшифровка архива сканов моделью Qwen3.6-35B-A3B через llama-server. Каждая страница читается дважды, прозрачные правила ставят флаги. Человек проверяет страницы с флагами в окне OpenWebUI. На выходе — PDF с поиском и DOCX.

Спецификация — `ocr-archive — проект для исполнителя`, редакция v2.8. Номера разделов в коде и здесь ссылаются на неё.

## Состав

| Часть | Где |
|---|---|
| REST API (`ocr serve`) | `ocr_archive/api/` |
| Воркер (`ocr worker`) | `ocr_archive/worker.py`, `pipeline.py` |
| Сборка PDF/DOCX/чанков, экспорт | `ocr_archive/build/` |
| Pipe для OpenWebUI | `owui/ocr_archive_pipe.py` |
| Прототип окна (P0.5) | `prototype/owui_window_pipe.py` |
| Docker | `Dockerfile`, `compose.yaml` |
| Установка на хост | `deploy/install.md` |

## Запуск

```bash
cp config.example.yaml config.yaml   # заполнить vlm.weights
cp .env.example .env                 # задать ключи
docker compose build
docker compose up -d
docker compose run --rm worker ocr doctor
docker compose run --rm worker ocr doctor --record-fingerprint
```

Подробно, с подготовкой каталогов и llama-server, — `deploy/install.md`.

## Команды

Все команды выполняются в контейнере, например `docker compose exec worker ocr status`.

| Команда | Что делает |
|---|---|
| `ocr add <файлы> [--title] [--ingest] [--priority]` | поставить в очередь из `/inbox` |
| `ocr status [<id>]` | состояние документов |
| `ocr rebuild <id>` | принудительная пересборка |
| `ocr draft <id>` | черновой DOCX без проверки |
| `ocr ingest <id>` | загрузка в базу знаний (при `ingester.enabled`) |
| `ocr redecide <id> \| --all` | перерешить собранные документы по текущим правилам |
| `ocr reprocess <id> [--include-reviewed]` | распознать заново |
| `ocr audit sample <id> [--n 10] [--track vlm\|pdf_text]` | выборочная проверка |
| `ocr audit report [<id>]` | отчёт выборочной проверки |
| `ocr pilot report <id>` | отчёт пилота |
| `ocr gold export <id> --pages 3,17` / `ocr gold import` | эталон |
| `ocr bench --profile pass1 --profile pass2` | сравнение профилей по эталону |
| `ocr page <файл> <n>` | отладка одной страницы, ничего не сохраняет |
| `ocr doctor [--record-fingerprint]` | проверка окружения |

Правила (`decide`, `signals`, `hints`, `checks`) меняются в `config.yaml`. Они вступают в силу после `docker compose restart worker api`. Несобранные документы перерешаются сами, собранные — по `ocr redecide`.

## Пилот

1. Выбрать 10–20 страниц, выгрузить: `ocr gold export <id> --pages ...`. Вычитать файлы в `/data/gold/` по скану, затем `ocr gold import`.
2. `ocr bench --profile pass1 --profile pass2` — выбрать разрешения и `decide.candidate`.
3. Пройти очередь пилотного документа в окне.
4. `ocr audit sample <id> --n 10` и `ocr audit sample <id> --n 10 --track pdf_text`, вычитать.
5. `ocr pilot report <id>` и подобрать пороги `signals.*`.

## Разработка

```bash
uv venv .venv --python 3.12
uv pip install --python .venv -e ".[dev]"
.venv/bin/pytest -q
.venv/bin/ruff format --check . && .venv/bin/ruff check . && .venv/bin/mypy
```

Тесты помечены маркерами фаз `p1`…`p6`. Тесты, которым нужен pandoc, пропускаются, если его нет в `PATH`.
