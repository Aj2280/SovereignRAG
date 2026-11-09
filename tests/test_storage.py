"""Unit tests for SQLite, BM25, and Qdrant storage layers."""

from nexusrag.storage.bm25 import BM25Index
from nexusrag.storage.sqlite_store import ChunkRecord, DocumentRecord, SQLiteStore
from nexusrag.storage.vector_store import QdrantVectorStore


def test_sqlite_store(tmp_path):
    db_path = str(tmp_path / "test.db")
    store = SQLiteStore(db_path=db_path)

    doc = DocumentRecord(
        doc_id="doc1",
        file_path="/tmp/doc1.txt",
        file_hash="hash123",
        file_size_bytes=100,
        modified_time=12345.0,
        title="Doc One",
        doc_type="txt",
        total_chunks=2,
    )
    store.upsert_document(doc)

    fetched = store.get_document_by_path("/tmp/doc1.txt")
    assert fetched is not None
    assert fetched.doc_id == "doc1"
    assert fetched.file_hash == "hash123"

    # Insert chunk
    chunk = ChunkRecord(
        chunk_id="c1",
        parent_id="p1",
        doc_id="doc1",
        chunk_index=0,
        text="Sample child text",
        parent_text="Sample parent text with more details",
        page_number=1,
        section_title="Intro",
        char_start=0,
        char_end=17,
        metadata={"is_parent": False},
    )
    store.insert_chunks([chunk])

    fetched_chunk = store.get_chunk("c1")
    assert fetched_chunk is not None
    assert fetched_chunk.text == "Sample child text"
    assert fetched_chunk.parent_id == "p1"


def test_bm25_index(tmp_path):
    idx_path = str(tmp_path / "bm25.json")
    index = BM25Index(index_path=idx_path)

    chunk_ids = ["c1", "c2"]
    texts = [
        "NexusRAG performs local-first offline hybrid retrieval.",
        "Qdrant stores high-dimensional dense vectors with metadata filters.",
    ]
    metadatas = [{"title": "Doc 1"}, {"title": "Doc 2"}]

    index.add_documents(chunk_ids, texts, metadatas)

    results = index.search("local-first hybrid retrieval", top_k=2)
    assert len(results) > 0
    assert results[0][0] == "c1"
    assert results[0][1] > 0


def test_qdrant_vector_store():
    # In-memory mode for fast unit testing
    store = QdrantVectorStore(collection_name="test_col", dimension=4, mode="memory")

    chunk_ids = ["c1", "c2"]
    vectors = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
    texts = ["vector one", "vector two"]
    metadatas = [{"tag": "alpha"}, {"tag": "beta"}]

    store.upsert_chunks(chunk_ids, vectors, texts, metadatas)

    results = store.search_dense([1.0, 0.0, 0.0, 0.0], top_k=2)
    assert len(results) >= 1
    assert results[0][0] == "c1"
    assert results[0][1] > 0.99
