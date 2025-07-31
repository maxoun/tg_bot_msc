import os
import pytest
from pathlib import Path
from PyPDF2 import PdfWriter

class DummyComp:
    @staticmethod
    def create(**kwargs):
        class Choice:
            class Message:
                content="OK"
            message=Message()
        class Resp:
            choices=[Choice()]
        return Resp()

class DummyClient:
    chat=type("X",(),{"completions":DummyComp})

@pytest.fixture(autouse=True)
def patch_openai(monkeypatch):
    from openai import OpenAI
    monkeypatch.setattr(OpenAI,"__init__",lambda self,api_key,base_url:None)
    monkeypatch.setattr(OpenAI,"chat",DummyClient.chat)

@pytest.fixture
def sample(tmp_path):
    js=tmp_path/"p.json"
    js.write_text('[{"url":"u","title":"T","description":"D"}]')
    pd=tmp_path/"pdfs";pd.mkdir()
    f=pd/"f.pdf";PdfWriter().add_blank_page(72,72).write(str(f))
    return js,pd

def test_rag(sample, monkeypatch):
    os.environ["OPENAI_API_KEY"]=""
    os.environ["OPENAI_API_BASE"]="http://localhost:8000/v1"
    js,pd=sample
    from src.rag.openai_pipeline import RAGService
    rag=RAGService(
        model_name="m",
        json_path=js,
        pdf_dir=pd,
        hf_embed_model="sentence-transformers/all-MiniLM-L6-v2",
        chunk_size=10,
        chunk_overlap=0,
        top_k_retrieval=1,
        max_tokens=8
    )
    out=rag.ask("hi")
    assert out["answer"]=="OK"
    assert isinstance(out["sources"],list)
