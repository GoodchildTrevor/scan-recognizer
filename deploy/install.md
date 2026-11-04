# Установка ocr-archive на хост

Хост — Linux с Docker Engine и Compose v2. Docker Desktop не годится для каталога данных: на нём не гарантированы блокировки (раздел 12.4).

## 1. llama-server

К текущей команде запуска добавить (раздел 3.1):

```
--mmproj mmproj-BF16.gguf --image-max-tokens 6144 -np 2 \
--host 127.0.0.1 --port 8080 --api-key "$LLAMA_API_KEY" -c 32768
```

- Если GPU не поддерживает BF16, взять `mmproj-F16.gguf`.
- Если после добавления mmproj не хватает VRAM, увеличить `--n-cpu-moe`. Смотреть `nvidia-smi` на странице профиля pass2.
- Фактическое число токенов картинки pass2 проверить в логе сервера. Оно должно быть не больше `--image-max-tokens`.

## 2. Каталоги

```bash
sudo mkdir -p /srv/ocr-archive/{data,inbox,export}
sudo chown -R 10001:10001 /srv/ocr-archive/data /srv/ocr-archive/export
```

`data` — только локальный диск (ext4, xfs), не NFS и не CIFS. Контейнеры работают от пользователя с uid 10001.

## 3. Конфигурация

```bash
cp config.example.yaml config.yaml
cp .env.example .env
sha256sum /path/to/model.gguf /path/to/mmproj-BF16.gguf
```

- В `config.yaml` заполнить `vlm.weights`: имена файлов и их sha256. Они входят в версию распознавания.
- В `.env` задать длинные случайные `OCR_API_KEY` и `LLAMA_API_KEY`.
- `api.host` — адрес моста Docker, обычно `172.17.0.1` (`ip addr show docker0`).

## 4. Сборка и запуск

```bash
docker compose build --build-arg PANDOC_VERSION=3.5 --build-arg PANDOC_SHA256=<sha256 .deb>
docker compose up -d
docker compose logs -f worker
```

## 5. Проверка

```bash
docker compose run --rm worker ocr doctor
docker compose run --rm worker ocr doctor --record-fingerprint
```

Отпечаток модели записывается после P0 и после каждого намеренного обновления модели. Второй воркер не запустится: `docker compose run --rm worker ocr worker` должен завершиться с кодом 1.

## 6. OpenWebUI

1. У контейнера OpenWebUI: `extra_hosts: ["host.docker.internal:host-gateway"]`.
2. Admin → Functions → добавить `owui/ocr_archive_pipe.py`. В Valves задать:
   - `backend_url = http://host.docker.internal:8090`;
   - `api_key` — значение `OCR_API_KEY`;
   - `allowed_user_ids` — id заказчика.
3. Workspace → Models → «OCR-архив»:
   - видимость private;
   - подсказки «Проверить накопленное» и «Что в очереди».
4. Приватность (раздел 3.3, риск 2):
   - Admin → Documents: локальные движок извлечения и модель эмбеддингов;
   - задачи OpenWebUI (название чата, теги, подсказки) выключены или выполняются текущей моделью;
   - `WEBSOCKET_EVENT_CALLER_TIMEOUT` не задавать.
5. Версию OpenWebUI зафиксировать. После каждого обновления повторить прототип P0.5 (`prototype/owui_window_pipe.py`).

## 7. Папка экспорта

`/srv/ocr-archive/export` пишет только воркер. Раздавать её можно по SFTP, позже — подключить в Nextcloud как внешнее хранилище только на чтение. `список.csv` переписывается последним: версия в нём означает, что оба файла этой версии на месте.

## 8. Ingest

По умолчанию выключен (`ingester.enabled: false`). Перед включением заказчик решает, можно ли отдавать найденные фрагменты облачным моделям OpenWebUI (раздел 22.2). Отдельной задачей нужно исправить баг частичной загрузки в qdrant-ingester (раздел 3.2).
