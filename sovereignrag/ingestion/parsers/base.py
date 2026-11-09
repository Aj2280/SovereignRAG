"""Base abstractions for document parsers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ParsedElement:
    text: str
    element_type: str  # "heading", "paragraph", "table", "code", "list"
    page_number: int = 1
    section_title: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedDocument:
    doc_id: str
    file_path: str
    title: str
    doc_type: str
    elements: List[ParsedElement]
    raw_text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseParser(ABC):
    """Abstract parser for specific file formats."""

    @abstractmethod
    def parse(self, file_path: str) -> ParsedDocument:
        """Parse file and return structured document elements."""
        pass

    @abstractmethod
    def can_parse(self, file_path: str) -> bool:
        """Check if parser handles this file extension."""
        pass
