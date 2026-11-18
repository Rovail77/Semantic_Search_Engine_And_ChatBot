# Semantic Search Engine & ChatBot

Юридический ассистент для поиска судебной практики по описанию жизненной
ситуации на естественном языке.

[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.135%2B-009688)](https://fastapi.tiangolo.com/)
[![FAISS](https://img.shields.io/badge/FAISS-vector%20search-4B8BBE)](https://github.com/facebookresearch/faiss)
[![Ollama](https://img.shields.io/badge/LLM-Ollama%20%7C%20OpenAI--compatible-black)](https://ollama.com/)
[![Tests](https://github.com/Rovail77/Semantic_Search_Engine_And_ChatBot/actions/workflows/tests.yml/badge.svg)](https://github.com/Rovail77/Semantic_Search_Engine_And_ChatBot/actions/workflows/tests.yml)

Проект объединяет гибридный поиск (FAISS + BM25 + RRF), переформулировку
запросов через LLM и генерацию черновиков процессуальных документов.

## Возможности

- Поиск судебных решений по смыслу, а не только по точным словам.
- Гибридное ранжирование: FAISS (эмбеддинги) + BM25 + Reciprocal Rank Fusion.
- LLM-переформулировка бытовой ситуации в юридические запросы.
- Ответы по схеме RAG со ссылками на найденные документы.
- Генерация черновиков: исковое заявление, претензия, ходатайство, жалоба.
- Проверка ссылок на нормы и предупреждения о неподтверждённых статьях.
- Экспорт документов в DOCX.
- Логирование запросов, задержек и экспертных оценок в SQLite.
- Административная панель и фоновый планировщик обновления индекса.
- Поддержка Ollama и OpenAI-совместимых API (vLLM/TGI).
- Ограничение частоты запросов, защитные заголовки и админ-токен.
- Необязательное серверное хранение черновиков (DOCX + метаданные).

## Быстрый старт

### 1. Зависимости

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

### 2. Конфигурация

```bash
copy .env.example .env
```

По умолчанию используется локальный Ollama:

```dotenv
LLM_PROVIDER=ollama
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b
```

Для сервера обязательно задайте `ADMIN_TOKEN` — иначе логи и админ-операции
открыты. Парсер работает через Edge (`PARSER_BROWSER=edge`) или Chromium
(`PARSER_BROWSER=chrome`).

### 3. Модель

```bash
ollama pull qwen2.5:7b
ollama serve
```

Если Ollama установлена портативно, используйте
`scripts\start_ollama.bat`.

### 4. Данные и индексы

```bash
python parser.py
python create_index.py
```

Для быстрого демо-набора:

```bash
python parser.py --limit 20 --max-categories 1 --max-pages 2 --headless
```

### 5. Запуск

```bash
python app.py
```

- Чат: http://localhost:8000
- Админ-панель: http://localhost:8000/admin
- Swagger: http://localhost:8000/docs

Или одной командой: `scripts\start_all.bat`.

## Структура проекта

```text
app.py                  FastAPI-приложение и REST API
chatbot.py              Диалоговый менеджер
search_engine.py        Гибридный поиск FAISS + BM25 + RRF
reformulator.py         Промпты и сценарии LLM
llm_client.py           Клиент Ollama / OpenAI-совместимого API
document_generator.py   Генерация черновиков документов и DOCX
parser.py               Сбор судебных решений с sudact.ru
create_index.py         Построение FAISS и BM25 индексов
query_logger.py         Логирование запросов в SQLite
scheduler.py            Фоновое обновление данных
config.py               Конфигурация из переменных окружения
utils.py                Токенизация, RRF, сниппеты
templates/              Веб-интерфейс и админ-панель
tests/                  Модульные тесты
scripts/                Скрипты запуска
docs/                   Архитектура и развёртывание
```

## API

| Метод | Путь | Назначение |
| --- | --- | --- |
| POST | `/api/chat` | Основной сценарий чат-бота |
| POST | `/api/search` | Гибридный поиск |
| POST | `/api/reformulate` | Переформулировка запроса |
| POST | `/api/documents/generate` | Генерация черновика документа |
| POST | `/api/documents/export` | Экспорт черновика в DOCX |
| POST | `/api/documents/save` | Сохранить черновик на сервере (админ) |
| GET | `/api/documents/saved` | Список сохранённых документов (админ) |
| GET | `/api/document-types` | Типы документов |
| GET | `/api/health` | Состояние сервиса и LLM |
| GET | `/api/stats` | Статистика индекса |

Административные методы (`/api/logs`, `/api/reload-index`, работа с сохранёнными
документами) требуют заголовок `X-Admin-Token`, если в `.env` задан `ADMIN_TOKEN`.

## Тесты

```bash
pytest -q
```

214 тестов. Тесты не требуют загруженных индексов и работающей LLM; тяжёлые
модули FAISS проверяются при установленных зависимостях.

## Документация

- [Архитектура](docs/ARCHITECTURE.md)
- [Развёртывание на нейрокомпьютере ЧелГУ](docs/DEPLOYMENT_CHELGU.md)
- [Безопасность](docs/SECURITY.md)
- [Хранение документов](docs/DOCUMENT_STORAGE.md)
- [История изменений](CHANGELOG.md)

## Дисклеймер

Проект создан в учебных и исследовательских целях для Института права ЧелГУ.
Сгенерированные документы являются черновиками и требуют проверки
квалифицированным юристом. Система не заменяет юридическую консультацию.
