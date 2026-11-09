"""Unit tests for configuration loader."""

from sovereignrag.core.config import AppConfig, load_config


def test_load_config_defaults(tmp_path):
    cfg_file = tmp_path / "test_config.yaml"
    cfg_file.write_text(
        """
system:
  name: "TestSovereign"
  local_only: true
profiles:
  active: "speed_demon"
  speed_demon:
    llm:
      provider: "ollama"
      model: "llama3.1:8b"
    embeddings:
      provider: "mock"
      model: "mock-embed"
      dimension: 128
    reranker:
      provider: "mock"
      model: "mock-rerank"
"""
    )

    config = load_config(str(cfg_file))
    assert config.system.name == "TestSovereign"
    assert config.system.local_only is True

    profile = config.get_active_profile()
    assert profile.llm.model == "llama3.1:8b"
    assert profile.embeddings.dimension == 128
