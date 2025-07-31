# ITMO Master’s Programs RAG Telegram Bot

---

## Описание

Этот проект — Telegram-бот, который отвечает на вопросы абитуриентов о магистерских программах ИТМО «AI-product» и «Искусственный интеллект».  
В основе лежит Retrieval-Augmented Generation (RAG): бот автоматически

1. Парсит сайт абитуриента ИТМО,  
2. Скачивает PDF-учебные планы программ,  
3. Разбивает текст на чанки,  
4. Векторизует через **SentenceTransformers + FAISS**,  
5. Генерирует ответы с помощью **локально развернутого Qwen3-8B-AWQ на vLLM** (или `gpt-3.5-turbo` через OpenAI).

---

## Фичи

- 🔍 **Веб-парсинг** страниц программ и скачивание PDF «Учебный план»  
- 📚 **RAG-конвейер**:  
  - Chunking текста  
  - Embeddings через `sentence-transformers`  
  - Fast similarity search с **FAISS**  
- 🤖 **Telegram-бот** на `python-telegram-bot`  
- 🚀 **vLLM-сервер** с OpenAI-совместимым API для Qwen3-8B-AWQ  
- 📦 **Docker Compose**: единое развёртывание бота + vLLM  
- ✅ **Автоматические тесты** (pytest) для парсеров и RAG-pipeline

---

## Структура проекта

├── .env # Переменные окружения
├── docker-compose.yml # Docker Compose для bot + vllm
├── Dockerfile.bot # Сборка образа Telegram-бота
├── Dockerfile.vllm # Сборка образа vLLM-сервера
├── requirements.txt # Python-зависимости
├── data
│ ├── programs.json # Сырые данные программ
│ └── pdfs # Скачанные учебные планы
├── scripts
│ └── run_rag_demo.py # CLI-демо для тестирования RAG
├── src
│ ├── bot.py # Код Telegram-бота
│ ├── parsers
│ │ ├── html_parser.py # Веб-парсер + скачивание PDF
│ │ └── pdf_parser.py # PDF-парсер
│ └── rag
│ └── openai_pipeline.py# RAGService (FAISS + OpenAI/vLLM)
└── tests # pytest-тесты

## Требования

- Docker ≥ 20.10  
- docker-compose ≥ 1.29  
- Python 3.10–3.12 (локально)  
- NVIDIA GPU + драйверы (для vLLM-GPU)

---

## Конфигурация

Создайте файл `.env` в корне проекта:

```dotenv
# .env

# Если нужен внешний OpenAI:
OPENAI_API_KEY=           # Ваш ключ OpenAI.com
OPENAI_API_BASE=https://api.openai.com/v1

# Или для локального vLLM:
# OPENAI_API_KEY=
OPENAI_API_BASE=http://vllm:8000/v1

# Модель по умолчанию (vLLM vs OpenAI)
OPENAI_MODEL_NAME=Qwen/Qwen3-8B-AWQ

# Telegram Bot API token
TELEGRAM_TOKEN=TOKEN


Локальный запуск (без Docker)
Создайте и активируйте виртуальное окружение:

bash
Копировать
Редактировать
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows
