"""NetworkX Knowledge Graph Integration for GraphRAG multi-hop retrieval."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Set, Tuple
import networkx as nx

from nexusrag.core.logging import get_logger

logger = get_logger("nexusrag.storage.graph")


class KnowledgeGraphStore:
    """In-memory and file-backed entity-relation knowledge graph using NetworkX."""

    def __init__(self, persistence_path: str = "./data/knowledge_graph.json") -> None:
        self.persistence_path = persistence_path
        self.graph: nx.DiGraph = nx.DiGraph()
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.persistence_path):
            try:
                with open(self.persistence_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.graph = nx.node_link_graph(data)
                logger.debug(f"Loaded graph with {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges.")
            except Exception as e:
                logger.warning(f"Could not load graph from {self.persistence_path}: {e}")

    def save(self) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self.persistence_path)), exist_ok=True)
        data = nx.node_link_data(self.graph)
        with open(self.persistence_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    def add_triplet(
        self,
        subject: str,
        predicate: str,
        obj: str,
        chunk_id: Optional[str] = None,
        weight: float = 1.0,
    ) -> None:
        """Add a (subject, predicate, object) relation to the knowledge graph."""
        s = subject.strip().lower()
        o = obj.strip().lower()
        p = predicate.strip().lower()

        if not s or not o:
            return

        self.graph.add_node(s, label=subject)
        self.graph.add_node(o, label=obj)

        if self.graph.has_edge(s, o):
            # Update existing edge
            edge_data = self.graph[s][o]
            edge_data["weight"] = edge_data.get("weight", 1.0) + weight
            if chunk_id:
                chunks = edge_data.get("chunk_ids", [])
                if chunk_id not in chunks:
                    chunks.append(chunk_id)
                edge_data["chunk_ids"] = chunks
        else:
            self.graph.add_edge(
                s,
                o,
                predicate=p,
                weight=weight,
                chunk_ids=[chunk_id] if chunk_id else [],
            )

    def get_neighbors(self, entity: str, max_hops: int = 2) -> List[Dict[str, Any]]:
        """Traverse neighbors up to max_hops for an entity to support multi-hop reasoning."""
        ent = entity.strip().lower()
        if ent not in self.graph:
            return []

        results: List[Dict[str, Any]] = []
        visited: Set[str] = {ent}
        frontier: List[Tuple[str, int]] = [(ent, 0)]

        while frontier:
            curr, depth = frontier.pop(0)
            if depth >= max_hops:
                continue

            # Outgoing edges
            for nbr in self.graph.successors(curr):
                edge = self.graph[curr][nbr]
                results.append({
                    "subject": curr,
                    "predicate": edge.get("predicate", "related_to"),
                    "object": nbr,
                    "chunk_ids": edge.get("chunk_ids", []),
                    "depth": depth + 1,
                })
                if nbr not in visited:
                    visited.add(nbr)
                    frontier.append((nbr, depth + 1))

        return results
