"""Parser registry and dispatcher."""

from __future__ import annotations

from typing import List, Optional

from nexusrag.ingestion.parsers.base import BaseParser
from nexusrag.ingestion.parsers.code_parser import CodeParser
from nexusrag.ingestion.parsers.docx_parser import DocxParser
from nexusrag.ingestion.parsers.html_parser import HTMLParser
from nexusrag.ingestion.parsers.pdf_parser import PDFParser
from nexusrag.ingestion.parsers.text_parser import TextAndMarkdownParser

_PARSERS: List[BaseParser] = [
    PDFParser(),
    DocxParser(),
    HTMLParser(),
    TextAndMarkdownParser(),
    CodeParser(),
]


def get_parser_for_file(file_path: str) -> Optional[BaseParser]:
    """Find the first matching parser for the given file."""
    for parser in _PARSERS:
        if parser.can_parse(file_path):
            return parser
    return None
