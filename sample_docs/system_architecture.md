# Production RAG System Specifications

## 1. System Philosophy and Architecture
This system is designed as an ultra-reliable, local-first Retrieval-Augmented Generation system.
All vectors, relational records, indices, and weights reside strictly on local storage.
No user documents or queries ever leave the user's workstation.

## 2. Hybrid Retrieval Engine
The retrieval pipeline incorporates two orthogonal search strategies:
1. Dense Vector Search: Conducted over Qdrant collections utilizing high-dimensional embeddings from BAAI/bge-m3.
2. Sparse Lexical Search: Evaluated with BM25Plus, ensuring robust exact keyword, acronym, and code identifier recall.

Both ranked lists are fused via Reciprocal Rank Fusion (RRF):
The RRF score for document chunk d is computed as:
RRF(d) = (w_dense / (k + rank_dense(d))) + (w_sparse / (k + rank_sparse(d)))
Where k is configured to 60 by default.

## 3. Cross-Encoder Reranking
Following hybrid fusion, the candidate pool of top 50 passages is evaluated by BAAI/bge-reranker-v2-m3.
The cross-encoder performs deep bidirectional cross-attention across the query and candidate chunk pairs.
The top 5 to 7 highest-scoring chunks are retained.

## 4. Verification and Inline Citations
The system requires all answers to cite verifiable evidence using the format:
[doc: <title>, page: <page>, section: <section>]
The SelfCheckVerifier decomposes answers into atomic assertions and verifies that each is supported by the context passages.

