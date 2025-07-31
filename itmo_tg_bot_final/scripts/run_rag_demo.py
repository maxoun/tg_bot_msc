#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
from dotenv import load_dotenv

# подтягиваем HTMLParser и RAGService
from src.parsers.html_parser import HTMLParser
from src.rag.openai_pipeline import RAGService

def main():
    load_dotenv()

    p = argparse.ArgumentParser(
        description="RAG-демо: ITMO Master’s programs"
    )
    p.add_argument("-q", "--question", required=True, help="Ваш вопрос")
    p.add_argument(
        "--json",
        default="data/programs.json",
        help="Куда сохранить programs.json"
    )
    p.add_argument(
        "--pdf-dir",
        default="data/pdfs",
        help="Папка для PDF-файлов учебных планов"
    )
    # остальные опции RAG...
    p.add_argument("--model",      default=None, help="Модель: gpt-3.5-turbo или Qwen3-8B-AWQ")
    p.add_argument("--embed",      default="intfloat/multilingual-e5-large-instruct")
    p.add_argument("--chunk-size",    type=int, default=1000)
    p.add_argument("--chunk-overlap", type=int, default=100)
    p.add_argument("--top-k-ret",     type=int, default=5)
    p.add_argument("--min-score",     type=float, default=0.0)
    p.add_argument("--gen-top-k",     type=int, default=20)
    p.add_argument("--max-tokens",    type=int, default=4000)
    p.add_argument("--temp",          type=float, default=0.7)
    p.add_argument("--top-p",         type=float, default=0.8)
    p.add_argument("--penalty",       type=float, default=1.5)
    p.add_argument("--enable-thinking", action="store_true")
    p.add_argument("--system-prompt", type=str, default=None)
    args = p.parse_args()

    # 1) Парсим страницы и сохраняем programs.json + скачиваем PDF
    parser = HTMLParser(
        base_url="https://abit.itmo.ru",
        pdf_dir=Path(args.pdf_dir)
    )
    programs = []
    for url in [
        "https://abit.itmo.ru/program/master/ai_product",
        "https://abit.itmo.ru/program/master/ai",
    ]:
        try:
            prog = parser.parse_program_page(url)
            programs.append(prog)
        except Exception as e:
            print(f"⚠️ Ошибка парсинга {url}: {e}")

    out_json = Path(args.json)
    parser.save_programs_json(programs, out_json)

    # 2) Выбираем модель
    api_key = os.getenv("OPENAI_API_KEY", "")
    model_name = args.model or (
        "gpt-3.5-turbo" if api_key else "Qwen/Qwen3-8B-AWQ"
    )

    # 3) Инициализируем RAG-сервис
    rag = RAGService(
        model_name        = model_name,
        json_path         = out_json,
        pdf_dir           = Path(args.pdf_dir),
        hf_embed_model    = args.embed,
        chunk_size        = args.chunk_size,
        chunk_overlap     = args.chunk_overlap,
        top_k_retrieval   = args.top_k_ret,
        min_score         = args.min_score,
        max_tokens        = args.max_tokens,
        temperature       = args.temp,
        top_p             = args.top_p,
        top_k             = args.gen_top_k,
        presence_penalty  = args.penalty,
        enable_thinking   = args.enable_thinking,
        system_prompt     = args.system_prompt,
    )

    # 4) Отвечаем на вопрос
    out = rag.ask(args.question)
    print("\n=== Ответ ===\n", out["answer"])
    print("\n=== Источники ===")
    for src in out["sources"]:
        print(" •", src)

if __name__ == "__main__":
    main()
