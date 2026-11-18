# Развёртывание на нейрокомпьютере ЧелГУ

Пошаговая инструкция для сервера Института права. Предполагается Ubuntu
22.04/24.04, GPU NVIDIA и доступ к серверу по SSH с правами `sudo`.

## 1. Что получится

```
Интернет / сеть вуза
        │
   nginx :443  ──TLS, SSO/allowlist, лимиты──┐
        │                                    │
   uvicorn 127.0.0.1:8000  (один процесс)    │
        │                                    │
   Ollama / vLLM 127.0.0.1:11434  (GPU) ─────┘
        │
   /var/lib/legal-assistant  (индексы, корпус, логи, документы)
```

Компоненты:

| Компонент | Роль |
| --- | --- |
| nginx | HTTPS, вход, ограничение размера тела, таймауты |
| uvicorn + FastAPI | поиск, чат, генерация документов |
| FAISS + BM25 | гибридный поиск в памяти процесса |
| Ollama + Qwen2.5-7B | переформулировка и ответы |
| systemd | автозапуск и перезапуск при сбое |

## 2. Требования к серверу

| Ресурс | Минимум | Рекомендуется |
| --- | --- | --- |
| CPU | 8 ядер | 16 ядер |
| RAM | 16 ГБ | 32 ГБ |
| GPU | 8 ГБ VRAM | 12–24 ГБ (RTX 4070 SUPER и выше) |
| Диск | 50 ГБ SSD | 200 ГБ SSD/NVMe |
| ОС | Ubuntu 22.04 | Ubuntu 24.04 LTS |
| Python | 3.11 | 3.12 |

Память под демонстрационный корпус (346 документов, 14 323 чанка): около
1,5 ГБ на процесс приложения плюс 5 ГБ на модель. Полный корпус (~25 тыс.
документов) требует 6–8 ГБ RAM на FAISS/BM25 и модель эмбеддингов.

## 3. Подготовка системы

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip nginx git \
    build-essential libssl-dev libffi-dev curl sqlite3

# Драйвер NVIDIA и CUDA — по инструкции администратора нейрокомпьютера
nvidia-smi
```

Создайте пользователя и каталоги:

```bash
sudo useradd --system --create-home --shell /usr/sbin/nologin legal
sudo mkdir -p /opt/legal-assistant /var/lib/legal-assistant
sudo chown -R legal:legal /opt/legal-assistant /var/lib/legal-assistant
sudo chmod 750 /var/lib/legal-assistant
```

## 4. Код и виртуальное окружение

```bash
sudo -u legal git clone https://github.com/Rovail77/Semantic_Search_Engine_And_ChatBot.git \
    /opt/legal-assistant
cd /opt/legal-assistant
sudo -u legal python3 -m venv .venv
sudo -u legal .venv/bin/pip install --upgrade pip
sudo -u legal .venv/bin/pip install -r requirements.txt
```

Если на сервере нет интернета, соберите wheel-пакеты на машине с той же ОС и
версией Python:

```bash
pip download -r requirements.txt -d wheels/
# на сервере:
pip install --no-index --find-links wheels/ -r requirements.txt
```

Для GPU-индексации замените CPU-сборку torch на CUDA-сборку с сайта PyTorch
(иначе `create_index.py` будет работать на CPU — медленнее, но корректно).

## 5. Языковая модель

### Вариант A. Ollama (проще)

```bash
curl -fsSL https://ollama.com/install.sh | sh
sudo systemctl edit ollama
```

Добавьте:

```ini
[Service]
Environment="OLLAMA_HOST=127.0.0.1:11434"
Environment="OLLAMA_MODELS=/var/lib/ollama/models"
```

```bash
sudo mkdir -p /var/lib/ollama/models
sudo chown -R ollama:ollama /var/lib/ollama
sudo systemctl restart ollama
sudo -u ollama ollama pull qwen2.5:7b
ollama run qwen2.5:7b "Ответь одним словом: работает?"
```

Если интернета нет, скачайте модель на другой машине (`ollama pull`, затем
каталог `~/.ollama/models`) и скопируйте на сервер.

### Вариант B. vLLM (для нескольких пользователей)

```bash
python3 -m venv /opt/vllm && /opt/vllm/bin/pip install vllm
/opt/vllm/bin/python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct \
  --host 127.0.0.1 --port 8001 \
  --max-model-len 8192 --gpu-memory-utilization 0.85
```

В `.env` приложения:

```dotenv
LLM_PROVIDER=openai
OPENAI_BASE_URL=http://127.0.0.1:8001/v1
OPENAI_API_KEY=local
LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
```

vLLM лучше держит параллельные запросы; Ollama проще в обслуживании.

## 6. Конфигурация приложения

```bash
cd /opt/legal-assistant
sudo -u legal cp .env.example .env
sudo -u legal nano .env
```

Рабочий пример:

```dotenv
LLM_PROVIDER=ollama
OLLAMA_HOST=http://127.0.0.1:11434
OLLAMA_MODEL=qwen2.5:7b
LLM_TIMEOUT=120
LLM_MAX_RETRIES=1
LLM_MAX_CONCURRENCY=2

HOST=127.0.0.1
PORT=8000
RELOAD=false

CORS_ORIGINS=https://pravo.chelgu.ru
TRUST_PROXY=true
ADMIN_TOKEN=<сгенерируйте: python -c "import secrets;print(secrets.token_urlsafe(32))">

RATE_LIMIT_PER_MINUTE=60
RATE_LIMIT_CHAT_PER_MINUTE=15
MAX_REQUEST_BYTES=1000000

DOCUMENTS_ENABLED=true
DOCUMENTS_DIR=/var/lib/legal-assistant/stored_documents
DOCUMENTS_RETENTION_DAYS=90

PARSER_BROWSER=chrome
PARSER_HEADLESS=true
RUN_PIPELINE_ON_START=false
```

```bash
sudo chown legal:legal .env
sudo chmod 600 .env
sudo -u legal mkdir -p /var/lib/legal-assistant/stored_documents
```

## 7. Данные и индексы

### Если корпус уже собран на другой машине

Скопируйте `parsed_texts/`, `document_index.faiss`, `chunks_data.pkl`,
`bm25_index.pkl`, `parsed_urls.json` в `/opt/legal-assistant` (или в каталоги,
указанные в `.env`). Проверьте контрольные суммы:

```bash
sha256sum document_index.faiss chunks_data.pkl bm25_index.pkl
```

### Если собираете на сервере

```bash
# Chromium и драйвер
sudo apt install -y chromium-browser chromium-chromedriver
sudo ln -sf /usr/bin/chromedriver /usr/local/bin/chromedriver

sudo -u legal .venv/bin/python parser.py --limit 200 --max-categories 2 --max-pages 5 --headless
sudo -u legal .venv/bin/python create_index.py
```

`PARSER_BROWSER=chrome` включает Chrome/Chromium вместо Edge. Selenium Manager
может попытаться скачать драйвер — на закрытом сервере заранее положите
`chromedriver` и укажите `CHROME_DRIVER_PATH=/usr/local/bin/chromedriver`.

Если парсинг с сервера недоступен (блокировка sudact.ru), собирайте корпус на
рабочей машине и синхронизируйте только `parsed_texts/` и индексы.

## 8. systemd

`/etc/systemd/system/legal-assistant.service`:

```ini
[Unit]
Description=Legal Assistant (FastAPI)
After=network-online.target ollama.service
Wants=network-online.target

[Service]
Type=simple
User=legal
Group=legal
WorkingDirectory=/opt/legal-assistant
EnvironmentFile=/opt/legal-assistant/.env
ExecStart=/opt/legal-assistant/.venv/bin/uvicorn app:app \
    --host 127.0.0.1 --port 8000 --workers 1 \
    --proxy-headers --forwarded-allow-ips=127.0.0.1
Restart=always
RestartSec=5
TimeoutStopSec=30
LimitNOFILE=65535
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=full
ReadWritePaths=/var/lib/legal-assistant

[Install]
WantedBy=multi-user.target
```

Важно: **один worker**. Каждый процесс загрузит свою копию FAISS и модели
эмбеддингов, что удвоит память.

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now legal-assistant
systemctl status legal-assistant
curl -s http://127.0.0.1:8000/api/health
```

Ожидаемый ответ:

```json
{"status":"ok","index_loaded":true,"chatbot_loaded":true,...}
```

## 9. nginx и HTTPS

`/etc/nginx/sites-available/legal-assistant`:

```nginx
limit_req_zone $binary_remote_addr zone=legal_api:10m rate=30r/m;

server {
    listen 80;
    server_name pravo.chelgu.ru;
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl http2;
    server_name pravo.chelgu.ru;

    ssl_certificate     /etc/letsencrypt/live/pravo.chelgu.ru/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/pravo.chelgu.ru/privkey.pem;

    client_max_body_size 2m;
    proxy_read_timeout 300s;
    proxy_send_timeout 300s;

    add_header Strict-Transport-Security "max-age=31536000" always;

    location / {
        limit_req zone=legal_api burst=20 nodelay;
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # Дополнительная защита админки средствами nginx
    location ~ ^/(admin|api/logs|api/log-stats|api/documents/saved|api/reload-index) {
        allow 10.0.0.0/8;      # подсеть вуза
        deny all;
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

```bash
sudo ln -s /etc/nginx/sites-available/legal-assistant /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
sudo certbot --nginx -d pravo.chelgu.ru
```

Если сертификат выдан внутренним УЦ ЧелГУ, замените пути к файлам.

Для входа через SSO вуза поставьте приложение за уже существующий прокси
Института; тогда `ADMIN_TOKEN` остаётся вторым фактором для админ-операций.

## 10. Проверка после развёртывания

```bash
# 1. Приложение живо
curl -s https://pravo.chelgu.ru/api/health | python3 -m json.tool

# 2. Поиск работает
curl -s -X POST https://pravo.chelgu.ru/api/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"залив квартиры","top_k":3}'

# 3. LLM отвечает
curl -s -X POST https://pravo.chelgu.ru/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"Сосед затопил квартиру"}' | head -c 500

# 4. Админка закрыта без токена
curl -i https://pravo.chelgu.ru/api/logs        # 401

# 5. Генерация и экспорт
curl -s -X POST https://pravo.chelgu.ru/api/documents/generate \
  -H 'Content-Type: application/json' \
  -d '{"situation":"Сосед затопил квартиру и отказывается платить","document_type":"claim","fields":{"court":"Суд","plaintiff":"Иванов","defendant":"Петров","claims":"Взыскать ущерб"},"use_llm":false}' \
  | head -c 300
```

Проверьте GPU:

```bash
nvidia-smi
ollama ps
```

## 11. Обновление версии

```bash
cd /opt/legal-assistant
sudo -u legal git fetch origin
sudo -u legal git checkout main
sudo -u legal git pull --ff-only
sudo -u legal .venv/bin/pip install -r requirements.txt
sudo systemctl restart legal-assistant
curl -s http://127.0.0.1:8000/api/health
```

Перед обновлением:

```bash
sudo -u legal .venv/bin/python -m pytest -q
sudo tar -czf /backup/legal-assistant-$(date +%F).tar.gz \
  -C /opt legal-assistant/.env /var/lib/legal-assistant
```

Откат:

```bash
sudo -u legal git log --oneline -5
sudo -u legal git checkout <commit>
sudo systemctl restart legal-assistant
```

## 12. Мониторинг и обслуживание

| Что | Команда / место |
| --- | --- |
| Живость | `curl /api/health`, systemd `Restart=always` |
| Логи приложения | `journalctl -u legal-assistant -f` |
| Логи nginx | `/var/log/nginx/access.log`, `error.log` |
| GPU | `nvidia-smi`, `ollama ps` |
| Диск | `df -h /var/lib/legal-assistant` |
| Ошибки LLM | `journalctl -u legal-assistant | grep "LLM request failed"` |
| Очистка логов | cron раз в месяц (см. `docs/SECURITY.md`) |
| Бэкап документов | ежедневно (см. `docs/DOCUMENT_STORAGE.md`) |

Простой health-check для внешнего монитора:

```bash
*/5 * * * * curl -fsS https://pravo.chelgu.ru/api/health | grep -q '"status":"ok"' \
  || echo "legal-assistant degraded" | mail -s "Alert" admin@chelgu.ru
```

## 13. Типовые проблемы

| Симптом | Причина | Решение |
| --- | --- | --- |
| `status: degraded` | не найдены индексы | проверьте пути в `.env`, запустите `create_index.py` |
| 503 на `/api/chat` | индекс не загрузился | смотрите `journalctl`, права на файлы индексов |
| Ответ «LLM недоступна» | Ollama не запущена | `systemctl status ollama`, `ollama ps` |
| Очень медленно | модель на CPU или малый VRAM | `nvidia-smi`, уменьшите модель до 3B |
| 429 | сработал rate limit | увеличьте `RATE_LIMIT_*` или проверьте, не общий ли IP за прокси |
| 413 | тело больше 1 МБ | увеличьте `MAX_REQUEST_BYTES` и `client_max_body_size` |
| Пустые сниппеты | битые тексты в `parsed_texts` | перезапустите парсер, проверьте кодировку UTF-8 |
| После reload старый индекс | nginx/uvicorn с несколькими worker | оставьте `--workers 1` |

## 14. Чек-лист приёмки

- [ ] `systemctl status legal-assistant` — active (running).
- [ ] `/api/health` — `status: ok`, `index_loaded: true`.
- [ ] Поиск возвращает 5–10 релевантных документов.
- [ ] Чат отвечает за приемлемое время (10–30 с на GPU).
- [ ] Генерация и экспорт DOCX работают.
- [ ] `/api/logs` без токена — 401.
- [ ] HTTPS с валидным сертификатом, HSTS.
- [ ] Rate limit и размер тела проверены.
- [ ] Бэкап индексов и документов настроен и проверен.
- [ ] Ответственный за сервис и порядок обновления согласованы с Институтом.
