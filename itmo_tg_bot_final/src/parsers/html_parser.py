# src/parsers/html_parser.py

import json
import time
from pathlib import Path
import requests
from bs4 import BeautifulSoup

# Selenium-блок
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import WebDriverException, NoSuchElementException


class HTMLParser:
    """
    Парсер страниц магистратуры ИТМО.
    Инициализируется:
      - base_url: базовый URL (например, "https://abit.itmo.ru")
      - pdf_dir:   (опционально) папка для сохранения PDF-файлов
    Методы:
      - parse_program_page(url: str) -> dict
      - save_programs_json(programs: list[dict], out_json: str)
    """

    def __init__(self, base_url: str, pdf_dir: Path | str = None):
        self.base_url = base_url.rstrip("/")
        if pdf_dir:
            self.pdf_dir = Path(pdf_dir)
            self.pdf_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.pdf_dir = None

    def parse_program_page(self, url: str) -> dict:
        resp = requests.get(url)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        slug = url.rstrip("/").split("/")[-1]

        # 1) TITLE
        h1 = (
            soup.find("h1", class_=lambda c: c and "Information_information__header" in c)
            or soup.find("h1")
        )
        title = h1.get_text(strip=True) if h1 else None

        # 2) DESCRIPTION
        about = (
            soup.find("h2", id="about")
            or soup.find("h2", string=lambda t: t and "о программе" in t.lower())
        )
        description = ""
        if about:
            parts = []
            for sib in about.find_next_siblings():
                if sib.name and sib.name.startswith("h"):
                    break
                if sib.name in ("p", "span"):
                    parts.append(sib.get_text(" ", strip=True))
            description = " ".join(parts).strip()

        # 3) STATIC поиск PDF в разделе «Учебный план»
        pdf_url = None
        study = (
            soup.find("h2", id="study-plan")
            or soup.find("h2", string=lambda t: t and "учебный план" in t.lower())
        )
        if study:
            a_pdf = study.find_next(
                "a", href=lambda x: x and x.lower().endswith(".pdf")
            )
            if a_pdf:
                href = a_pdf["href"]
                pdf_url = href if href.startswith("http") else self.base_url + (href if href.startswith("/") else "/" + href)

        # 4) FALLBACK через Selenium, если статический поиск не дал результата
        if not pdf_url and self.pdf_dir:
            try:
                print(f"[selenium] пытаюсь найти PDF на странице {url}")
                pdf_url = self._find_pdf_via_selenium(url)
                print(f"[selenium] нашёл PDF: {pdf_url}")
            except Exception as e:
                print(f"[selenium] не удалось найти PDF: {e}")
                pdf_url = None

        # 5) Скачиваем PDF, если URL известен
        pdf_path = None
        if pdf_url and self.pdf_dir:
            print(f"Загружаем PDF для {slug}: {pdf_url}")
            pdf_path = self._download_pdf(pdf_url)

        # 6) Контакты
        manager_email = None
        mail_a = soup.find("a", href=lambda x: x and x.startswith("mailto:"))
        if mail_a:
            manager_email = mail_a.get_text(strip=True)

        manager_phone = None
        tel_a = soup.find("a", href=lambda x: x and x.startswith("tel:"))
        if tel_a:
            manager_phone = tel_a.get_text(strip=True)

        return {
            "slug": slug,
            "url": url,
            "title": title,
            "description": description,
            "pdf_url": pdf_url,
            "pdf_path": str(pdf_path) if pdf_path else None,
            "manager_email": manager_email,
            "manager_phone": manager_phone,
        }

    def _download_pdf(self, pdf_url: str) -> Path:
        """
        Скачиваем pdf_url → self.pdf_dir/<имя_файла>.pdf
        """
        r = requests.get(pdf_url, stream=True)
        r.raise_for_status()
        filename = pdf_url.rstrip("/").split("/")[-1]
        dest = self.pdf_dir / filename
        with dest.open("wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        return dest

    def _find_pdf_via_selenium(self, page_url: str) -> str:
        """
        Запускаем headless Chrome, загружаем страницу, ищем ссылку
        «скачать учебный план» и возвращаем её href.
        """
        opts = Options()
        opts.headless = True
        opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-dev-shm-usage")

        # Если у вас chromedriver не в PATH, укажите executable_path
        driver = webdriver.Chrome(options=opts)
        try:
            driver.get(page_url)
            # даём JS подгрузиться
            time.sleep(2)

            # ищем первую ссылку после <h2 id="study-plan">
            elem = driver.find_element(
                "xpath",
                "//h2[@id='study-plan']/following::a[contains(text(),'учебный план')][1]"
            )
            href = elem.get_attribute("href")
            if not href:
                raise NoSuchElementException("атрибут href пустой")
            return href
        finally:
            driver.quit()

    def save_programs_json(self, programs: list[dict], out_json: str):
        """
        Сохраняет список программ в JSON-файл по пути out_json.
        """
        path = Path(out_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(programs, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(programs)} programs to {path}")
