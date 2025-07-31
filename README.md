# ITMO Master’s Programs RAG Telegram Bot

Телеграм-бот, помогающий абитуриентам выбирать между магистерскими программами ИТМО «AI-Product» и «Искусственный интеллект» с использованием Retrieval-Augmented Generation (RAG).

Бот можно найти по имени в TG: **@itmo_msc_bot**
(Запущен и работает на мощностях автора)

---

## 🔎 Что делает

1. **Веб-парсинг** страниц программ ИТМО (заголовок, описание, контакты).
2. **Скачивает PDF** «Учебный план» каждой программы (пытается:( ).
3. **Извлекает и структурирует** текст из PDF.
4. **Разбивает текст на чанки**, векторизует их через `sentence-transformers`.
5. **Индексирует** с помощью FAISS.
6. **Генерирует ответы** через Chat API:
   - локально развернутый Qwen3-8B-AWQ на vLLM,  
   - или `gpt-3.5-turbo` (OpenAI).

7. **Telegram-бот** на `python-telegram-bot` обслуживает запросы пользователей.

---

## 📂 Структура проекта

```
.
├── .env                     # Переменные окружения
├── docker-compose.yml       # Определение сервисов Docker
├── Dockerfile.bot           # Сборка образа Telegram-бота
├── Dockerfile.vllm          # Сборка образа vLLM-сервера
├── requirements.txt         # Python-зависимости
├── data
│   ├── programs.json        # Данные программ (JSON)
│   └── pdfs                 # Скачанные PDF-учебные планы
├── scripts
│   └── run_rag_demo.py      # CLI-утилита для тестирования RAG-конвейера
├── src
│   ├── bot.py               # Основной код Telegram-бота
│   ├── parsers
│   │   ├── html_parser.py   # Парсер сайта + скачивание PDF
│   │   └── pdf_parser.py    # Извлечение текста из PDF
│   └── rag
│       └── openai_pipeline.py  # RAGService: FAISS + OpenAI/vLLM
└── tests                    # Автоматические pytest-тесты
```

---

## ⚙️ Требования

- **Docker** ≥ 20.10  
- **docker-compose** ≥ 1.29  
- **Python** 3.10–3.12 (для локального запуска)  
- **NVIDIA GPU** + драйверы (для vLLM на GPU)

---

## 🔑 Конфигурация `.env`

Создайте файл `.env` в корне проекта:

```dotenv
# .env

# 1) OpenAI API (если нужен внешний OpenAI)
OPENAI_API_KEY=
OPENAI_API_BASE=https://api.openai.com/v1

# 2) Локальный vLLM-сервер
# OPENAI_API_KEY=
OPENAI_API_BASE=http://localhost:8000/v1

# Модель по умолчанию
OPENAI_MODEL_NAME=Qwen/Qwen3-8B-AWQ

# Telegram Bot Token
TELEGRAM_TOKEN=123456789:ABCDEF-ghIJKlmnOPQrstUVwxyz
```


---

## 🛠 Локальный запуск (без Docker)

1. Создайте виртуальное окружение и активируйте его:
   ```bash
   python -m venv .venv
   source .venv/bin/activate    # Linux / macOS
   .venv\Scripts\activate     # Windows
   ```
2. Установите зависимости:
   ```bash
   pip install -r requirements.txt
   ```
3. Запустите предварительный парсинг и скачивание PDF:
   ```bash
   python scripts/run_rag_demo.py      --json data/programs.json      --pdf-dir data/pdfs      -q "Какие дисциплины в AI-product?"
   ```
4. Запустите Telegram-бот:
   ```bash
   python -m src.bot
   ```

---

## 🐳 Запуск через Docker Compose

1. Соберите образы:
   ```bash
   docker-compose build
   ```
2. Запустите сервисы:
   ```bash
   docker-compose up -d
   ```
3. Просмотрите логи бота:
   ```bash
   docker-compose logs -f bot
   ```

> Сервис **bot** автоматически подключается к **localhost:8000/v1)**  
> и использует `OPENAI_API_BASE=http://localhost:8000/v1`.

---

## ✅ Автоматические тесты

```bash
pytest --maxfail=1 --disable-warnings -q
```

Покрываются:
- Парсер HTML (bs4 + скачивание PDF)
- PDF-парсер (PyPDF2)
- RAGService (FAISS + OpenAI/vLLM mock)

---

## 📦 Зависимости (`requirements.txt`)

```text
beautifulsoup4>=4.10.0
requests>=2.25.1
python-dotenv>=0.19.0
python-telegram-bot>=20.0
openai>=0.27.0
faiss-cpu>=1.7.3
sentence-transformers>=2.2.2
PyPDF2>=3.0.0
vllm>=0.10.0
```

---

## 🔧 FAQ

- **Авто-парсинг при старте бота**: если `data/programs.json` отсутствует или устарел, бот сам запускает `HTMLParser` и скачивает актуальные данные.
- Для **отладки vLLM** установите переменную `VLLM_LOG_LEVEL=DEBUG` в окружении Docker.
- Интегрируйте тесты и сборку Docker-образов в ваш CI/CD pipeline.

---

## TO DO:

- Интегрировать тесты и сборку Docker-образов в ваш CI/CD pipeline.
- Использовать Selenium для парсинга PDF программы обучения.
- Интегрировать в бота настройку RAG пайплайна
- Использовать langchain для RAG
- Попробовать LLM побольше
- Реализовать выбор эмбедера для документов пользователем 
