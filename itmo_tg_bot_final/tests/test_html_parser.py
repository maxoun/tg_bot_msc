import pytest
from src.parsers.html_parser import HTMLParser
from unittest.mock import patch, MagicMock

SAMPLE_HTML = """
<html><body>
  <h1>Test</h1>
  <h2>О программе</h2>
  <p>Desc</p>
  <a href="/plan.pdf">Учебный план</a>
</body></html>
"""

@patch("src.parsers.html_parser.requests.get")
def test_parse_program_page(mock_get):
    resp=MagicMock(status_code=200, text=SAMPLE_HTML)
    mock_get.return_value=resp
    p=HTMLParser("https://abit.itmo.ru")
    r=p.parse_program_page("https://abit.itmo.ru/program/master/test")
    assert r["title"]=="Test"
    assert "Desc" in r["description"]
    assert r["pdf_url"].endswith("plan.pdf")
