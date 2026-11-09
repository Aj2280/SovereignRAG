"""Folder Watcher for automatic incremental re-indexing."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Callable, Dict, Optional, Set

from sovereignrag.core.logging import get_logger
from sovereignrag.ingestion.pipeline import IngestionPipeline, calculate_file_hash

logger = get_logger("sovereignrag.ingestion.watcher")


class FolderWatcher:
    """Monitors a local folder and incrementally indexes new, modified, or deleted files."""

    def __init__(
        self,
        folder_path: str,
        pipeline: IngestionPipeline,
        poll_interval: float = 5.0,
    ) -> None:
        self.folder_path = Path(folder_path).resolve()
        self.pipeline = pipeline
        self.poll_interval = poll_interval
        self._file_hashes: Dict[str, str] = {}
        self._running = False

    def scan_once(self) -> Dict[str, int]:
        """Perform a single scan of the directory."""
        if not self.folder_path.exists():
            return {"added": 0, "modified": 0, "removed": 0}

        current_files: Set[str] = set()
        added = 0
        modified = 0
        removed = 0

        for root, _, files in os.walk(self.folder_path):
            for file in files:
                ext = Path(file).suffix.lower()
                if ext in self.pipeline.config.ingestion.supported_extensions:
                    full_path = str(Path(os.path.join(root, file)).resolve())
                    current_files.add(full_path)
                    try:
                        f_hash = calculate_file_hash(full_path)
                        if full_path not in self._file_hashes:
                            self._file_hashes[full_path] = f_hash
                            logger.info(f"Watcher detected new file: {Path(full_path).name}")
                            self.pipeline.ingest_file(full_path)
                            added += 1
                        elif self._file_hashes[full_path] != f_hash:
                            self._file_hashes[full_path] = f_hash
                            logger.info(f"Watcher detected modified file: {Path(full_path).name}")
                            self.pipeline.ingest_file(full_path, force_reindex=True)
                            modified += 1
                    except Exception as e:
                        logger.error(f"Error inspecting {full_path}: {e}")

        # Check for deleted files
        known_files = set(self._file_hashes.keys())
        deleted = known_files - current_files
        for del_path in deleted:
            logger.info(f"Watcher detected removed file: {Path(del_path).name}")
            doc = self.pipeline.sqlite_store.get_document_by_path(del_path)
            if doc:
                self.pipeline._remove_document(doc.doc_id)
            del self._file_hashes[del_path]
            removed += 1

        return {"added": added, "modified": modified, "removed": removed}

    def start_polling(self, stop_condition: Optional[Callable[[], bool]] = None) -> None:
        """Run folder watcher poll loop."""
        self._running = True
        logger.info(f"Starting FolderWatcher on {self.folder_path} (interval: {self.poll_interval}s)...")
        while self._running:
            if stop_condition and stop_condition():
                break
            self.scan_once()
            time.sleep(self.poll_interval)

    def stop(self) -> None:
        self._running = False
