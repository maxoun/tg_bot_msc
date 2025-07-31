import os
import json
import logging
from pathlib import Path
from typing import List, Dict

import faiss
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from src.parsers.pdf_parser import PDFParser

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    step = chunk_size - chunk_overlap
    if step <= 0:
        raise ValueError("chunk_size must be > chunk_overlap")
    chunks = []
    i = 0
    n = len(text)
    while i < n:
        chunks.append(text[i : i + chunk_size])
        i += step
    return chunks


class RAGService:
    def __init__(
        self,
        model_name: str,
        json_path: Path,
        pdf_dir: Path,
        hf_embed_model: str = "intfloat/multilingual-e5-large-instruct",
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        top_k_retrieval: int = 5,
        min_score: float = 0.3,               
        max_tokens: int = 4000,
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 20,
        presence_penalty: float = 1.5,
        enable_thinking: bool = False,
        system_prompt: str = None,
    ):
        # API key/base
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        self.api_base = os.getenv("OPENAI_API_BASE", "http://localhost:8000/v1")

        # RAG + generation params
        self.model_name = model_name
        self.top_k_retrieval = top_k_retrieval
        self.min_score = min_score                    
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.presence_penalty = presence_penalty
        self.enable_thinking = enable_thinking

        # Russian system prompt
        self.system_prompt = system_prompt or (
            "Вы — экспертный помощник, помогающий абитуриентам "
            "выбрать между магистерскими программами ИТМО «AI‑product» и «Искусственный интеллект». "
            "Отвечайте ТОЛЬКО на вопросы по этим программам, учебным планам, элективам и "
            "процессу поступления. Если вопрос не по теме, отвечайте: "
            "«Я могу помочь только по вопросам поступления на магистратуру ИТМО.»"
        )

        # 1) JSON
        self.docs: List[Dict] = []
        for entry in json.loads(Path(json_path).read_text(encoding="utf-8")):
            txt = ""
            if entry.get("title"):
                txt += entry["title"] + "\n\n"
            if entry.get("description"):
                txt += entry["description"] + "\n\n"
            self.docs.append({"page_content": txt, "source": entry.get("url", "")})
        logger.info("Loaded %d JSON docs", len(self.docs))

        # 2) PDF
        parser = PDFParser()
        for pdf in Path(pdf_dir).glob("*.pdf"):
            raw = parser.extract_text(pdf)
            struct = parser.parse_structured(raw)
            self.docs.append({"page_content": struct["raw"], "source": pdf.name})
        logger.info("Loaded total %d docs", len(self.docs))

        # 3) Chunking
        self.chunks: List[Dict] = []
        for doc in self.docs:
            for c in chunk_text(doc["page_content"], chunk_size, chunk_overlap):
                self.chunks.append({"page_content": c, "source": doc["source"]})
        logger.info("Split into %d chunks", len(self.chunks))

        # 4) Embeddings + FAISS
        self.embedder = SentenceTransformer(hf_embed_model)
        texts = [c["page_content"] for c in self.chunks]
        embs = self.embedder.encode(texts, show_progress_bar=True)
        dim = embs.shape[1]
        faiss.normalize_L2(embs)
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embs)
        logger.info("Built FAISS index with %d vectors (dim=%d)", embs.shape[0], dim)

        # 5) OpenAI SDK клиент
        self.client = OpenAI(api_key=self.api_key, base_url=self.api_base)

    def _retrieve(self, query: str) -> List[Dict]:
        q_emb = self.embedder.encode([query])
        faiss.normalize_L2(q_emb)
        scores, idxs = self.index.search(q_emb, self.top_k_retrieval)
        out = []
        for score, i in zip(scores[0], idxs[0]):
            if score >= self.min_score:              # отфильтровываем по порогу
                out.append(self.chunks[i])
        logger.info("Retrieved %d chunks (scores first=%s)", len(out), scores[0][: len(out)])
        return out

    def ask(self, question: str) -> Dict:
        docs = self._retrieve(question)
        messages = [{"role": "system", "content": self.system_prompt}]
        for d in docs:
            messages.append({"role": "system", "content": d["page_content"]})
        messages.append({"role": "user", "content": question})

        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            presence_penalty=self.presence_penalty,
            extra_body={
                "top_k": self.top_k,
                "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
            },
        )

        answer = resp.choices[0].message.content
        sources = [d["source"] for d in docs]
        return {"answer": answer, "sources": sources}
