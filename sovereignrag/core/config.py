"""SovereignRAG Configuration Loader and Validator.

Reads config.yaml, merges environment variables, and provides strongly-typed
Pydantic models for all subsystems.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
from pydantic import BaseModel, Field

from sovereignrag.core.hardware import get_hardware_profile


class SystemSettings(BaseModel):
    name: str = "SovereignRAG"
    version: str = "0.1.0"
    local_only: bool = True
    log_level: str = "INFO"
    device: str = "auto"


class LLMProfileConfig(BaseModel):
    provider: str = "ollama"
    model: str = "qwen2.5:14b"
    temperature: float = 0.1
    max_tokens: int = 2048
    context_window: int = 16384


class EmbeddingProfileConfig(BaseModel):
    provider: str = "sentence-transformers"
    model: str = "BAAI/bge-m3"
    dimension: int = 1024
    batch_size: int = 16


class RerankerProfileConfig(BaseModel):
    provider: str = "cross-encoder"
    model: str = "BAAI/bge-reranker-v2-m3"
    top_n: int = 5


class ProfileConfig(BaseModel):
    description: str = ""
    llm: LLMProfileConfig = Field(default_factory=LLMProfileConfig)
    embeddings: EmbeddingProfileConfig = Field(default_factory=EmbeddingProfileConfig)
    reranker: RerankerProfileConfig = Field(default_factory=RerankerProfileConfig)


class OllamaProviderConfig(BaseModel):
    base_url: str = "http://localhost:11434"
    timeout_seconds: int = 120


class OpenAICompatibleConfig(BaseModel):
    base_url: str = "http://localhost:1234/v1"
    api_key: str = "dummy"
    timeout_seconds: int = 120


class ClaudeConfig(BaseModel):
    api_key_env: str = "ANTHROPIC_API_KEY"
    model: str = "claude-3-5-sonnet-20241022"


class LLMProvidersConfig(BaseModel):
    ollama: OllamaProviderConfig = Field(default_factory=OllamaProviderConfig)
    llamacpp: OpenAICompatibleConfig = Field(
        default_factory=lambda: OpenAICompatibleConfig(base_url="http://localhost:8080/v1")
    )
    vllm: OpenAICompatibleConfig = Field(
        default_factory=lambda: OpenAICompatibleConfig(base_url="http://localhost:8000/v1")
    )
    openai_compatible: OpenAICompatibleConfig = Field(default_factory=OpenAICompatibleConfig)
    claude: ClaudeConfig = Field(default_factory=ClaudeConfig)


class SentenceTransformersConfig(BaseModel):
    cache_dir: str = "./data/models/embeddings"
    device: str = "auto"


class EmbeddingProvidersConfig(BaseModel):
    sentence_transformers: SentenceTransformersConfig = Field(default_factory=SentenceTransformersConfig)
    ollama: OllamaProviderConfig = Field(default_factory=OllamaProviderConfig)
    openai_compatible: OpenAICompatibleConfig = Field(default_factory=OpenAICompatibleConfig)


class CrossEncoderConfig(BaseModel):
    cache_dir: str = "./data/models/rerankers"
    device: str = "auto"


class RerankerProvidersConfig(BaseModel):
    cross_encoder: CrossEncoderConfig = Field(default_factory=CrossEncoderConfig)


class ProvidersConfig(BaseModel):
    llm: LLMProvidersConfig = Field(default_factory=LLMProvidersConfig)
    embeddings: EmbeddingProvidersConfig = Field(default_factory=EmbeddingProvidersConfig)
    rerankers: RerankerProvidersConfig = Field(default_factory=RerankerProvidersConfig)


class QdrantConfig(BaseModel):
    mode: str = "local"  # "local", "memory", "server"
    path: str = "./data/qdrant"
    collection_name: str = "sovereignrag_chunks"
    host: str = "localhost"
    port: int = 6333


class StorageConfig(BaseModel):
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    sqlite: Dict[str, Any] = Field(default_factory=lambda: {"path": "./data/sovereignrag.db"})
    bm25: Dict[str, Any] = Field(default_factory=lambda: {"path": "./data/bm25_index.json"})
    graph: Dict[str, Any] = Field(default_factory=lambda: {"enabled": False, "backend": "networkx"})


class ChunkingConfig(BaseModel):
    strategy: str = "parent_child"
    parent_chunk_size: int = 1500
    parent_overlap: int = 150
    child_chunk_size: int = 400
    child_overlap: int = 50
    semantic_fallback: bool = True


class IngestionConfig(BaseModel):
    supported_extensions: List[str] = Field(
        default_factory=lambda: [
            ".pdf", ".docx", ".pptx", ".html", ".htm", ".md", ".txt", ".csv", ".xlsx", ".py", ".js", ".ts", ".json"
        ]
    )
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    contextual_retrieval: Dict[str, Any] = Field(
        default_factory=lambda: {"enabled": False, "max_summary_tokens": 100}
    )
    incremental: Dict[str, Any] = Field(
        default_factory=lambda: {"hash_algorithm": "sha256", "watch_interval_seconds": 5}
    )


class MMRConfig(BaseModel):
    enabled: bool = True
    diversity: float = 0.3


class AgenticConfig(BaseModel):
    enabled: bool = True
    max_iterations: int = 3
    sufficiency_threshold: float = 0.65


class RetrievalConfig(BaseModel):
    top_k_dense: int = 50
    top_k_sparse: int = 50
    rrf_k: int = 60
    dense_weight: float = 0.6
    sparse_weight: float = 0.4
    reranker_enabled: bool = True
    rerank_top_n: int = 5
    mmr: MMRConfig = Field(default_factory=MMRConfig)
    agentic: AgenticConfig = Field(default_factory=AgenticConfig)


class VerificationConfig(BaseModel):
    self_check_enabled: bool = True
    strict_no_hallucination: bool = True
    confidence_threshold: float = 0.70


class GenerationConfig(BaseModel):
    system_prompt: str = (
        "You are SovereignRAG, an authoritative, rigorous local-first assistant. "
        "Answer the user query ONLY from the provided context passages. "
        "Every factual assertion MUST be backed by an inline citation in the format [doc, page, section]."
    )
    verification: VerificationConfig = Field(default_factory=VerificationConfig)


class AppConfig(BaseModel):
    system: SystemSettings = Field(default_factory=SystemSettings)
    profiles: Dict[str, Any] = Field(default_factory=dict)
    providers: ProvidersConfig = Field(default_factory=ProvidersConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)

    def get_active_profile(self) -> ProfileConfig:
        active_name = self.profiles.get("active", "balanced")
        profile_data = self.profiles.get(active_name, {})
        return ProfileConfig.model_validate(profile_data)

    def resolve_device(self) -> str:
        if self.system.device != "auto":
            return self.system.device
        hw = get_hardware_profile()
        return hw.recommended_device


_CONFIG_CACHE: Optional[AppConfig] = None


def _substitute_env_vars(raw: str) -> str:
    """Replace ${VAR} or $VAR with environment variables."""
    pattern = re.compile(r"\$\{?([A-Za-z0-9_]+)\}?")

    def repl(match):
        var_name = match.group(1)
        return os.environ.get(var_name, match.group(0))

    return pattern.sub(repl, raw)


def load_config(config_path: str = "config.yaml") -> AppConfig:
    """Load configuration file, substituting env vars, and validate with Pydantic."""
    global _CONFIG_CACHE
    path = Path(config_path)

    if not path.exists():
        # Fallback to defaults
        _CONFIG_CACHE = AppConfig()
        return _CONFIG_CACHE

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    substituted = _substitute_env_vars(content)
    parsed = yaml.safe_load(substituted) or {}
    _CONFIG_CACHE = AppConfig.model_validate(parsed)
    return _CONFIG_CACHE


def get_config() -> AppConfig:
    """Get loaded config or load default config.yaml."""
    global _CONFIG_CACHE
    if _CONFIG_CACHE is None:
        cfg_path = os.environ.get("SOVEREIGNRAG_CONFIG_PATH", "config.yaml")
        return load_config(cfg_path)
    return _CONFIG_CACHE

