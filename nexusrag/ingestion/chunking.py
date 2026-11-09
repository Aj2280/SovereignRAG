"""Structure-Aware Parent-Child Chunker for NexusRAG.

Splits documents by structural headings into larger parent context blocks
and smaller child chunks for high-precision vector search.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from nexusrag.core.config import ChunkingConfig
from nexusrag.ingestion.parsers.base import ParsedDocument, ParsedElement
from nexusrag.storage.sqlite_store import ChunkRecord


@dataclass
class ChunkHierarchy:
    parent_records: List[ChunkRecord]
    child_records: List[ChunkRecord]


class StructureAwareChunker:
    """Chunks documents preserving heading hierarchy, emitting parent-child pairs."""

    def __init__(
        self,
        parent_chunk_size: int = 1500,
        parent_overlap: int = 150,
        child_chunk_size: int = 400,
        child_overlap: int = 50,
    ) -> None:
        self.parent_chunk_size = parent_chunk_size
        self.parent_overlap = parent_overlap
        self.child_chunk_size = child_chunk_size
        self.child_overlap = child_overlap

    @classmethod
    def from_config(cls, config: ChunkingConfig) -> StructureAwareChunker:
        return cls(
            parent_chunk_size=config.parent_chunk_size,
            parent_overlap=config.parent_overlap,
            child_chunk_size=config.child_chunk_size,
            child_overlap=config.child_overlap,
        )

    def _split_text_with_overlap(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Sliding window text split honoring whitespace and sentence breaks."""
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + chunk_size, text_len)
            if end < text_len:
                # Seek nearby space or period to avoid cutting words
                pivot = text.rfind(" ", start, end)
                if pivot > start + (chunk_size // 2):
                    end = pivot

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            if end >= text_len:
                break
            start = max(end - overlap, start + 1)

        return chunks

    def chunk_document(self, parsed_doc: ParsedDocument) -> ChunkHierarchy:
        """Create parent and child chunks from parsed document elements."""
        parent_records: List[ChunkRecord] = []
        child_records: List[ChunkRecord] = []

        # Group elements by logical section
        sections: Dict[str, List[ParsedElement]] = {}
        for elem in parsed_doc.elements:
            sec_name = elem.section_title or "General"
            if sec_name not in sections:
                sections[sec_name] = []
            sections[sec_name].append(elem)

        chunk_counter = 0

        for sec_name, elements in sections.items():
            # Combine element texts within section
            combined_text = "\n\n".join(e.text for e in elements if e.text.strip())
            if not combined_text:
                continue

            primary_page = elements[0].page_number if elements else 1

            # Generate Parent Chunks
            parent_texts = self._split_text_with_overlap(
                combined_text, self.parent_chunk_size, self.parent_overlap
            )

            for p_idx, p_text in enumerate(parent_texts):
                p_id = f"{parsed_doc.doc_id}_p{len(parent_records)}"
                p_record = ChunkRecord(
                    chunk_id=p_id,
                    parent_id=None,
                    doc_id=parsed_doc.doc_id,
                    chunk_index=chunk_counter,
                    text=p_text,
                    parent_text=None,
                    page_number=primary_page,
                    section_title=sec_name,
                    char_start=0,
                    char_end=len(p_text),
                    metadata={
                        "title": parsed_doc.title,
                        "file_path": parsed_doc.file_path,
                        "doc_type": parsed_doc.doc_type,
                        "is_parent": True,
                    },
                )
                parent_records.append(p_record)
                chunk_counter += 1

                # Generate Child Chunks from Parent Chunk
                child_texts = self._split_text_with_overlap(
                    p_text, self.child_chunk_size, self.child_overlap
                )

                for c_idx, c_text in enumerate(child_texts):
                    c_id = f"{p_id}_c{c_idx}"
                    c_record = ChunkRecord(
                        chunk_id=c_id,
                        parent_id=p_id,
                        doc_id=parsed_doc.doc_id,
                        chunk_index=chunk_counter,
                        text=c_text,
                        parent_text=p_text,
                        page_number=primary_page,
                        section_title=sec_name,
                        char_start=0,
                        char_end=len(c_text),
                        metadata={
                            "title": parsed_doc.title,
                            "file_path": parsed_doc.file_path,
                            "doc_type": parsed_doc.doc_type,
                            "is_parent": False,
                            "parent_id": p_id,
                        },
                    )
                    child_records.append(c_record)
                    chunk_counter += 1

        return ChunkHierarchy(
            parent_records=parent_records,
            child_records=child_records,
        )
