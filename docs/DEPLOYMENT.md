# Развёртывание

## Локальный запуск

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Для Ollama:

```dotenv
LLM_PROVIDER=ollama
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b
```

```bash
ollama pull qwen2.5:7b
ollama serve
```

Затем:

```bash
python parser.py --limit 20 --max-categories 1 --max-pages 2 --headless
python create_index.py
python app.py
```

## Нейрокомпьютер ЧелГУ

Рекомендуемая схема:

1. Развернуть на GPU-сервере Ollama или vLLM с Qwen2.5-7B/14B.
2. Открыть inference endpoint только во внутренней сети.
3. Настроить приложение:

```dotenv
LLM_PROVIDER=openai
OPENAI_BASE_URL=http://<neurocomputer>:8000/v1
OPENAI_API_KEY=<внутренний ключ>
LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
LLM_TIMEOUT=120
```

4. Поставить Nginx с HTTPS перед FastAPI.
5. Ограничить CORS:

```dotenv
CORS_ORIGINS=https://law.chelgu.ru
```

6. Запускать приложение как службу (systemd, NSSM или Docker).
7. Проверить `GET /api/health`.
8. Провести нагрузочное тестирование и пилот в Институте права.

## Переменные окружения

| Переменная | По умолчанию | Назначение |
| --- | --- | --- |
| `LLM_PROVIDER` | `ollama` | `ollama` или `openai` |
| `LLM_MODEL` | `qwen2.5:7b` | Модель для OpenAI-режима |
| `OLLAMA_HOST` | `http://localhost:11434` | Адрес Ollama |
| `OLLAMA_MODEL` | `qwen2.5:7b` | Модель Ollama |
| `OPENAI_BASE_URL` | — | URL OpenAI-совместимого API |
| `OPENAI_API_KEY` | — | Ключ доступа |
| `LLM_TIMEOUT` | `120` | Таймаут запроса, секунды |
| `PARSED_TEXTS_DIR` | `parsed_texts` | Каталог текстов решений |
| `QUERY_LOGS_DB` | `query_logs.db` | База логов |
| `PARSER_HEADLESS` | `false` | Headless-режим Edge |
| `COLLECTION_INTERVAL_HOURS` | `24` | Интервал переиндексации |
| `RUN_PIPELINE_ON_START` | `false` | Сбор данных при старте |
| `CORS_ORIGINS` | `http://localhost:8000` | Разрешённые источники |
| `HOST` / `PORT` | `0.0.0.0` / `8000` | Адрес веб-сервера |
| `RELOAD` | `false` | Автоперезагрузка Uvicorn |

## Обновление индекса

После переиндексации:

```bash
curl -X POST http://localhost:8000/api/reload-index
```

Или перезапустить сервер.

## Ограничения

- Один источник данных — sudact.ru; учитывайте правила сайта.
- Для эксплуатации рекомендуется модель 7B+.
- FAISS-индекс и эмбеддинги требуют нескольких гигабайт RAM.
- Сессии чата хранятся в памяти.
- Аутентификацию лучше реализовать на уровне reverse-proxy или SSO вуза.
- Сгенерированные документы всегда требуют проверки юристом.
