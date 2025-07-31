# src/bot.py

import os
import logging
import asyncio
import re
from pathlib import Path

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters

from src.parsers.html_parser import HTMLParser
from src.rag.openai_pipeline import RAGService

# ─── КОНФИГ ───────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

TELEGRAM_TOKEN     = os.getenv("TELEGRAM_TOKEN")
OPENAI_MODEL_NAME  = os.getenv("OPENAI_MODEL_NAME", "gpt-3.5-turbo")
PROGRAM_URLS = [
    "https://abit.itmo.ru/program/master/ai_product",
    "https://abit.itmo.ru/program/master/ai",
]

if not TELEGRAM_TOKEN:
    raise RuntimeError("В .env не задан TELEGRAM_TOKEN")

# ─── ЛОГИРОВАНИЕ ────────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ─── УТИЛИТЫ ────────────────────────────────────────────────────────────────────
def dedupe_preserve_order(seq):
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def md_to_html(text: str) -> str:
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"\*(.+?)\*",   r"<i>\1</i>", text)
    return text

# ─── ПАРСИНГ ПРОГРАММ ────────────────────────────────────────────────────────────
# Тут при старте бота обновляем programs.json и скачиваем PDF-ы
parser = HTMLParser(
    base_url="https://abit.itmo.ru",
    pdf_dir=BASE_DIR / "data" / "pdfs"
)
programs = []
for url in PROGRAM_URLS:
    try:
        prog = parser.parse_program_page(url)
        programs.append(prog)
        logger.info("Parsed %s → %s", url, prog.get("pdf_url"))
    except Exception as e:
        logger.error("Ошибка парсинга %s: %s", url, e)
# Сохраняем
parser.save_programs_json(programs, BASE_DIR / "data" / "programs.json")

# ─── ИНИЦИАЛИЗАЦИЯ RAG ───────────────────────────────────────────────────────────
pipeline = RAGService(
    model_name = OPENAI_MODEL_NAME,
    json_path  = BASE_DIR / "data" / "programs.json",
    pdf_dir    = BASE_DIR / "data" / "pdfs",
)

# ─── ХЭНДЛЕР НА СООБЩЕНИЯ ─────────────────────────────────────────────────────────
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text.strip()
    logger.info("User asked: %s", user_text)

    try:
        result       = await asyncio.get_running_loop().run_in_executor(
            None, pipeline.ask, user_text
        )
        answer       = result["answer"]
        sources      = result["sources"]
        unique_src   = dedupe_preserve_order(sources)
        answer_html  = md_to_html(answer)
        sources_list = "\n".join(f"• {Path(s).name}" for s in unique_src)

        await update.message.reply_html(
            f"{answer_html}\n\n📚 <b>Источники:</b>\n{sources_list}"
        )

    except Exception:
        logger.exception("Ошибка при обработке вопроса")
        await update.message.reply_text(
            "Извините, при обработке вашего запроса что-то пошло не так."
        )

# ─── ТОЧКА ЗАПУСКА ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message)
    )
    logger.info("Бот запущен, начинаем polling...")
    app.run_polling()
