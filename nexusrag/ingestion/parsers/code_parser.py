"""Source Code and Structured Config Parser."""

from __future__ import annotations

import re
from pathlib import Path
from typing import List

from nexusrag.ingestion.parsers.base import BaseParser, ParsedDocument, ParsedElement


class CodeParser(BaseParser):
    """Parses source code and config files preserving function/class boundaries."""

    SUPPORTED_EXTS = {
        ".py", ".js", ".ts", ".jsx", ".tsx", ".json", ".sql", ".sh",
        ".yaml", ".yml", ".c", ".cpp", ".h", ".hpp", ".rs", ".go", ".java"
    }

    def can_parse(self, file_path: str) -> bool:
        return Path(file_path).suffix.lower() in self.SUPPORTED_EXTS

    def parse(self, file_path: str) -> ParsedDocument:
        abs_path = str(Path(file_path).resolve())
        title = Path(file_path).name
        ext = Path(file_path).suffix.lower()

        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        raw_text = "".join(lines)
        elements: List[ParsedElement] = []

        current_section = f"File: {title}"
        buffer: List[str] = []

        # Patterns for function / class definitions
        func_class_pattern = re.compile(
            r"^\s*(def\s+\w+|class\s+\w+|function\s+\w+|export\s+(default\s+)?(function|class)\s+\w+|fn\s+\w+|pub\s+fn\s+\w+)"
        )

        for line in lines:
            stripped = line.strip()
            match = func_class_pattern.match(stripped)
            if match:
                if buffer:
                    elements.append(
                        ParsedElement(
                            text="".join(buffer).strip(),
                            element_type="code",
                            section_title=current_section,
                        )
                    )
                    buffer = []

                current_section = match.group(1)
                elements.append(
                    ParsedElement(
                        text=current_section,
                        element_type="heading",
                        section_title=current_section,
                    )
                )

            buffer.append(line)

        if buffer:
            elements.append(
                ParsedElement(
                    text="".join(buffer).strip(),
                    element_type="code",
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
