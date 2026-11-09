"""Text and Markdown parser preserving structural headings."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List

from nexusrag.ingestion.parsers.base import BaseParser, ParsedDocument, ParsedElement


class TextAndMarkdownParser(BaseParser):
    """Parses plain text, Markdown, and CSV files into structured elements."""

    SUPPORTED_EXTS = {".txt", ".md", ".markdown", ".rst", ".csv", ".tsv"}

    def can_parse(self, file_path: str) -> bool:
        ext = Path(file_path).suffix.lower()
        return ext in self.SUPPORTED_EXTS

    def parse(self, file_path: str) -> ParsedDocument:
        abs_path = str(Path(file_path).resolve())
        title = Path(file_path).stem
        ext = Path(file_path).suffix.lower()

        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        raw_text = "".join(lines)

        # Specialized handling for CSV and TSV tabular files
        if ext in {".csv", ".tsv"}:
            import csv
            delimiter = "\t" if ext == ".tsv" else ","
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                reader = list(csv.reader(f, delimiter=delimiter))

            elements: List[ParsedElement] = []
            if reader:
                headers = [h.strip() for h in reader[0]]
                md_table_lines = [" | ".join(headers), " | ".join(["---"] * len(headers))]
                for row in reader[1:]:
                    if any(cell.strip() for cell in row):
                        md_table_lines.append(" | ".join(c.strip() for c in row))
                        row_desc = ", ".join(f"{h}: {val.strip()}" for h, val in zip(headers, row) if val.strip())
                        elements.append(
                            ParsedElement(
                                text=row_desc,
                                element_type="table_row",
                                section_title=title,
                                metadata={"headers": headers},
                            )
                        )
                elements.insert(0, ParsedElement(
                    text="\n".join(md_table_lines),
                    element_type="table",
                    section_title=title,
                ))

            return ParsedDocument(
                doc_id=title,
                file_path=abs_path,
                title=title,
                doc_type="csv",
                elements=elements,
                raw_text=raw_text,
            )

        elements: List[ParsedElement] = []
        current_section = "Introduction"
        buffer: List[str] = []
        in_code_block = False

        for line in lines:
            stripped = line.strip()

            # Handle code fence
            if stripped.startswith("```"):
                in_code_block = not in_code_block
                buffer.append(line)
                if not in_code_block and buffer:
                    elements.append(
                        ParsedElement(
                            text="".join(buffer).strip(),
                            element_type="code",
                            section_title=current_section,
                        )
                    )
                    buffer = []
                continue

            if in_code_block:
                buffer.append(line)
                continue

            # Check markdown heading
            heading_match = re.match(r"^(#{1,6})\s+(.+)$", stripped)
            if heading_match:
                # Flush previous paragraph buffer
                if buffer:
                    elements.append(
                        ParsedElement(
                            text="".join(buffer).strip(),
                            element_type="paragraph",
                            section_title=current_section,
                        )
                    )
                    buffer = []

                current_section = heading_match.group(2).strip()
                elements.append(
                    ParsedElement(
                        text=current_section,
                        element_type="heading",
                        section_title=current_section,
                        metadata={"level": len(heading_match.group(1))},
                    )
                )
                continue

            # Check markdown table
            if stripped.startswith("|") and stripped.endswith("|"):
                buffer.append(line)
                continue

            if not stripped:
                if buffer:
                    elements.append(
                        ParsedElement(
                            text="".join(buffer).strip(),
                            element_type="paragraph",
                            section_title=current_section,
                        )
                    )
                    buffer = []
            else:
                buffer.append(line)

        if buffer:
            elements.append(
                ParsedElement(
                    text="".join(buffer).strip(),
                    element_type="paragraph",
                    section_title=current_section,
                )
            )

        return ParsedDocument(
            doc_id=title,
            file_path=abs_path,
            title=title,
            doc_type=ext.replace(".", ""),
            elements=elements,
            raw_text=raw_text,
        )
