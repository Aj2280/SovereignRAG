"""PDF Parser preserving page numbers and section layouts."""

from __future__ import annotations

import re
from pathlib import Path
from typing import List
from pypdf import PdfReader

from nexusrag.core.logging import get_logger
from nexusrag.ingestion.parsers.base import BaseParser, ParsedDocument, ParsedElement

logger = get_logger("nexusrag.ingestion.parsers.pdf")


class PDFParser(BaseParser):
    """Layout-aware PDF parser preserving page numbers and section headings."""

    def can_parse(self, file_path: str) -> bool:
        return Path(file_path).suffix.lower() == ".pdf"

    def parse(self, file_path: str) -> ParsedDocument:
        abs_path = str(Path(file_path).resolve())
        title = Path(file_path).stem

        reader = PdfReader(file_path)
        elements: List[ParsedElement] = []
        full_text_parts: List[str] = []

        current_section = "Document Start"

        for page_idx, page in enumerate(reader.pages):
            page_num = page_idx + 1
            page_text = page.extract_text() or ""
            full_text_parts.append(page_text)

            lines = page_text.split("\n")
            buffer: List[str] = []

            for line in lines:
                stripped = line.strip()
                if not stripped:
                    if buffer:
                        elements.append(
                            ParsedElement(
                                text=" ".join(buffer).strip(),
                                element_type="paragraph",
                                page_number=page_num,
                                section_title=current_section,
                            )
                        )
                        buffer = []
                    continue

                # Heuristic for section headings in PDFs:
                # 1. Short lines (< 70 chars)
                # 2. Starting with numbered prefixes (e.g. "1. Introduction", "Section 2") or ALL CAPS
                is_heading = False
                if len(stripped) < 70 and (
                    re.match(r"^(\d+(\.\d+)*|[A-Z])[\.:\)]\s+[A-Z]", stripped)
                    or (stripped and stripped[0].isalpha() and stripped.isupper() and len(stripped.split()) > 1 and len(stripped.split()) < 8)
                    or stripped.lower().startswith(("section ", "chapter ", "abstract", "conclusion", "references", "table of contents"))
                ):
                    is_heading = True

                if is_heading:
                    if buffer:
                        elements.append(
                            ParsedElement(
                                text="\n".join(buffer).strip(),
                                element_type="paragraph",
                                page_number=page_num,
                                section_title=current_section,
                            )
                        )
                        buffer = []

                    current_section = stripped
                    elements.append(
                        ParsedElement(
                            text=current_section,
                            element_type="heading",
                            page_number=page_num,
                            section_title=current_section,
                        )
                    )
                else:
                    buffer.append(stripped)

            if buffer:
                elements.append(
                    ParsedElement(
                        text="\n".join(buffer).strip(),
                        element_type="paragraph",
                        page_number=page_num,
                        section_title=current_section,
                    )
                )

        return ParsedDocument(
            doc_id=title,
            file_path=abs_path,
            title=title,
            doc_type="pdf",
            elements=elements,
            raw_text="\n\n".join(full_text_parts),
            metadata={"num_pages": len(reader.pages)},
        )
