"""Unit tests for structure-aware parent-child chunking."""

from nexusrag.ingestion.chunking import StructureAwareChunker
from nexusrag.ingestion.parsers.base import ParsedDocument, ParsedElement


def test_structure_aware_chunker():
    elements = [
        ParsedElement(text="Architecture Overview", element_type="heading", section_title="Overview"),
        ParsedElement(text="NexusRAG is a local-first RAG engine designed for high precision offline search.", element_type="paragraph", section_title="Overview"),
        ParsedElement(text="Hybrid Search Details", element_type="heading", section_title="Search"),
        ParsedElement(text="It fuses dense vectors from Qdrant with BM25 keyword rankings using RRF.", element_type="paragraph", section_title="Search"),
    ]

    doc = ParsedDocument(
        doc_id="test_doc",
        file_path="/tmp/test.md",
        title="Test Document",
        doc_type="md",
        elements=elements,
        raw_text="...",
    )

    chunker = StructureAwareChunker(
        parent_chunk_size=500,
        parent_overlap=50,
        child_chunk_size=100,
        child_overlap=20,
    )

    hierarchy = chunker.chunk_document(doc)

    assert len(hierarchy.parent_records) >= 2
    assert len(hierarchy.child_records) >= 2

    # Check parent-child linkage
    for child in hierarchy.child_records:
        assert child.parent_id is not None
        assert child.parent_id.startswith("test_doc_p")
        assert child.parent_text is not None
        assert len(child.parent_text) >= len(child.text)
