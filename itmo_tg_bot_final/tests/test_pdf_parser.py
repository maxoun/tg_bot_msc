import pytest
from pathlib import Path
from src.parsers.pdf_parser import PDFParser
from PyPDF2 import PdfWriter

def test_extract_not_found(tmp_path):
    p=PDFParser()
    with pytest.raises(FileNotFoundError):
        p.extract_text(tmp_path/"nofile.pdf")

def test_parse_and_extract(tmp_path):
    fp=tmp_path/"f.pdf"
    w=PdfWriter()
    w.add_blank_page(72,72).write(str(fp))
    p=PDFParser()
    text=p.extract_text(fp)
    assert isinstance(text,str)
    struct=p.parse_structured(text)
    assert "raw" in struct and "sections" in struct
