"""HTML and Web Content Parser."""

from __future__ import annotations

from pathlib import Path
from typing import List
from bs4 import BeautifulSoup

from sovereignrag.ingestion.parsers.base import BaseParser, ParsedDocument, ParsedElement


class HTMLParser(BaseParser):
    """Parses HTML documents preserving tag semantics and structure."""

    def can_parse(self, file_path: str) -> bool:
        return Path(file_path).suffix.lower() in {".html", ".htm"}

    def parse(self, file_path: str) -> ParsedDocument:
        abs_path = str(Path(file_path).resolve())
        title = Path(file_path).stem

        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            html_content = f.read()

        soup = BeautifulSoup(html_content, "html.parser")

        # Strip script and style tags
        for s in soup(["script", "style", "noscript"]):
            s.decompose()

        page_title = soup.title.string.strip() if soup.title and soup.title.string else title

        elements: List[ParsedElement] = []
        raw_parts: List[str] = []
        current_section = page_title

        # Traverse body tags
        body = soup.body or soup
        for tag in body.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "table", "pre", "code"]):
            text = tag.get_text().strip()
            if not text:
                continue

            raw_parts.append(text)
            tag_name = tag.name.lower()

            if tag_name.startswith("h"):
                current_section = text
                elements.append(
                    ParsedElement(
                        text=text,
                        element_type="heading",
                        section_title=current_section,
                        metadata={"level": int(tag_name[1])},
                    )
                )
            elif tag_name == "table":
                elements.append(
                    ParsedElement(
                        text=text,
                        element_type="table",
                        section_title=current_section,
                    )
                )
            elif tag_name in ("pre", "code"):
                elements.append(
                    ParsedElement(
                        text=text,
                        element_type="code",
                        section_title=current_section,
                    )
                )
            else:
                elements.append(
                    ParsedElement(
                        text=text,
                        element_type="paragraph",
                        section_title=current_section,
                    )
                )

        return ParsedDocument(
            doc_id=title,
            file_path=abs_path,
            title=page_title,
            doc_type="html",
            elements=elements,
            raw_text="\n\n".join(raw_parts),
        )
