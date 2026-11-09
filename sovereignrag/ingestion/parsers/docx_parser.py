"""DOCX Parser for Microsoft Word documents."""

from __future__ import annotations

from pathlib import Path
from typing import List
from docx import Document

from sovereignrag.ingestion.parsers.base import BaseParser, ParsedDocument, ParsedElement


class DocxParser(BaseParser):
    """Extracts paragraphs, headings, and tables from .docx files."""

    def can_parse(self, file_path: str) -> bool:
        return Path(file_path).suffix.lower() == ".docx"

    def parse(self, file_path: str) -> ParsedDocument:
        abs_path = str(Path(file_path).resolve())
        title = Path(file_path).stem

        doc = Document(file_path)
        elements: List[ParsedElement] = []
        raw_parts: List[str] = []
        current_section = "Introduction"

        for p in doc.paragraphs:
            text = p.text.strip()
            if not text:
                continue

            raw_parts.append(text)
            style_name = p.style.name.lower() if p.style else ""

            if "heading" in style_name or "title" in style_name:
                current_section = text
                elements.append(
                    ParsedElement(
                        text=text,
                        element_type="heading",
                        section_title=current_section,
                        metadata={"style": style_name},
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

        # Extract tables
        for table in doc.tables:
            table_rows = []
            for row in table.rows:
                row_text = [cell.text.strip() for cell in row.cells]
                table_rows.append(" | ".join(row_text))
            formatted_table = "\n".join(table_rows)
            raw_parts.append(formatted_table)
            elements.append(
                ParsedElement(
                    text=formatted_table,
                    element_type="table",
                    section_title=current_section,
                )
            )

        return ParsedDocument(
            doc_id=title,
            file_path=abs_path,
            title=title,
            doc_type="docx",
            elements=elements,
            raw_text="\n\n".join(raw_parts),
        )
