"""SovereignRAG Structured Logging.

Provides rich console and file logging with support for pipeline step tracing.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any, Dict, Optional

try:
    from rich.console import Console
    from rich.logging import RichHandler
    _CONSOLE = Console(stderr=True)
    _HAS_RICH = True
except ImportError:
    _CONSOLE = None
    _HAS_RICH = False

_INITIALIZED = False


def setup_logger(
    name: str = "sovereignrag",
    level: str = "INFO",
    log_file: Optional[str] = "./data/sovereignrag.log",
) -> logging.Logger:
    """Configure and return a structured logger."""
    global _INITIALIZED
    logger = logging.getLogger(name)

    if _INITIALIZED:
        return logger

    log_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(log_level)
    logger.propagate = False

    if _HAS_RICH and _CONSOLE:
        # Rich console handler
        console_handler = RichHandler(
            console=_CONSOLE,
            show_time=True,
            show_path=False,
            rich_tracebacks=True,
            markup=True,
        )
        formatter = logging.Formatter("%(message)s", datefmt="[%X]")
    else:
        # Standard fallback console handler
        console_handler = logging.StreamHandler(sys.stderr)
        formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", datefmt="%X")

    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Optional file handler
    if log_file:
        try:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setLevel(log_level)
            file_fmt = logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s"
            )
            file_handler.setFormatter(file_fmt)
            logger.addHandler(file_handler)
        except Exception as e:
            logger.warning(f"Failed to initialize file logger at {log_file}: {e}")

    _INITIALIZED = True
    return logger


def get_logger(name: str = "sovereignrag") -> logging.Logger:
    """Retrieve an existing logger or create a default one."""
    return logging.getLogger(name)


def log_pipeline_step(
    step_name: str,
    details: Dict[str, Any],
    logger: Optional[logging.Logger] = None,
) -> None:
    """Structured trace for a retrieval/generation pipeline step."""
    log = logger or get_logger("sovereignrag.pipeline")
    summary = ", ".join(f"{k}={v}" for k, v in details.items() if k != "content")
    log.debug(f"[bold cyan]Pipeline Step:[/bold cyan] {step_name} -> {summary}")

