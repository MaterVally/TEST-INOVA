"""
CockroachDB-backed graph storage — drop-in replacement for NetworkXStorage.

Same interface as storage/graph_storage.py's NetworkXStorage, so it can be
swapped in without touching builder.py, text2graph.py, or fusion.py at all.
Those files only ever call the BaseGraphStorage methods below — they don't
know or care whether the graph lives in a local .graphml file or in
CockroachDB rows.

Requires: pip install "psycopg[binary,pool]"
Env var:  COCKROACH_DATABASE_URL="postgresql://user:pass@host:26257/defaultdb?sslmode=verify-full"
          (get this from the CockroachDB Cloud Console, free-tier cluster)

Usage — same call sites as NetworkXStorage, just swap the class:
    self.graph = CockroachGraphStorage(
        namespace="chunk_entity_relation",
        workspace_id=workspace_id,   # NEW — required, scopes every query
    )
"""
import os
from dataclasses import dataclass

import psycopg
from psycopg_pool import AsyncConnectionPool

from .storage.graph_storage import BaseGraphStorage

_DSN = os.environ.get("COCKROACH_DATABASE_URL")
_pool: AsyncConnectionPool | None = None


def _get_pool() -> AsyncConnectionPool:
    global _pool
    if _pool is None:
        if not _DSN:
            raise RuntimeError(
                "COCKROACH_DATABASE_URL is not set. Get a free-tier connection "
                "string from the CockroachDB Cloud Console and set it as an "
                "env var before using CockroachGraphStorage."
            )
        _pool = AsyncConnectionPool(_DSN, min_size=1, max_size=10, open=False)
    return _pool


@dataclass
class CockroachGraphStorage(BaseGraphStorage):
    workspace_id: str = None  # required — scopes every row so workspaces never mix

    def __post_init__(self):
        if not self.workspace_id:
            raise ValueError("CockroachGraphStorage requires workspace_id")

    async def _pool(self):
        pool = _get_pool()
        if pool.closed:  # psycopg_pool lazy-open
            await pool.open()
        return pool

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    async def has_node(self, node_id: str) -> bool:
        pool = await self._pool()
        async with pool.connection() as conn:
            row = await (await conn.execute(
                "SELECT 1 FROM graph_nodes WHERE workspace_id=%s AND node_id=%s",
                (self.workspace_id, node_id),
            )).fetchone()
            return row is not None

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        pool = await self._pool()
        async with pool.connection() as conn:
            row = await (await conn.execute(
                "SELECT 1 FROM graph_edges WHERE workspace_id=%s AND source_id=%s AND target_id=%s",
                (self.workspace_id, source_node_id, target_node_id),
            )).fetchone()
            return row is not None

    async def get_node(self, node_id: str) -> dict | None:
        pool = await self._pool()
        async with pool.connection() as conn:
            row = await (await conn.execute(
                "SELECT entity_type, description, source_id FROM graph_nodes "
                "WHERE workspace_id=%s AND node_id=%s",
                (self.workspace_id, node_id),
            )).fetchone()
            if not row:
                return None
            return {"entity_type": row[0], "description": row[1], "source_id": row[2]}

    async def get_edge(self, source_node_id: str, target_node_id: str) -> dict | None:
        pool = await self._pool()
        async with pool.connection() as conn:
            row = await (await conn.execute(
                "SELECT description, weight FROM graph_edges "
                "WHERE workspace_id=%s AND source_id=%s AND target_id=%s",
                (self.workspace_id, source_node_id, target_node_id),
            )).fetchone()
            if not row:
                return None
            return {"description": row[0], "weight": row[1]}

    async def get_node_edges(self, source_node_id: str):
        pool = await self._pool()
        async with pool.connection() as conn:
            rows = await (await conn.execute(
                "SELECT source_id, target_id FROM graph_edges "
                "WHERE workspace_id=%s AND (source_id=%s OR target_id=%s)",
                (self.workspace_id, source_node_id, source_node_id),
            )).fetchall()
            return [(r[0], r[1]) for r in rows] or None

    async def node_degree(self, node_id: str) -> int:
        edges = await self.get_node_edges(node_id)
        return len(edges) if edges else 0

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        return await self.node_degree(src_id) + await self.node_degree(tgt_id)

    # ------------------------------------------------------------------
    # Writes — this is the actual "dynamic graph" behavior: UPSERT means
    # a second document's entities are added to the existing row set,
    # never overwriting the whole table.
    # ------------------------------------------------------------------

    async def upsert_node(self, node_id: str, node_data: dict[str, str]):
        pool = await self._pool()
        async with pool.connection() as conn:
            await conn.execute(
                """
                INSERT INTO graph_nodes (workspace_id, node_id, entity_type, description, source_id, updated_at)
                VALUES (%s, %s, %s, %s, %s, now())
                ON CONFLICT (workspace_id, node_id) DO UPDATE SET
                    entity_type = excluded.entity_type,
                    description = excluded.description,
                    source_id   = excluded.source_id,
                    updated_at  = now()
                """,
                (
                    self.workspace_id,
                    node_id,
                    node_data.get("entity_type"),
                    node_data.get("description"),
                    node_data.get("source_id"),
                ),
            )

    async def upsert_edge(self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]):
        pool = await self._pool()
        async with pool.connection() as conn:
            await conn.execute(
                """
                INSERT INTO graph_edges (workspace_id, source_id, target_id, description, weight, updated_at)
                VALUES (%s, %s, %s, %s, %s, now())
                ON CONFLICT (workspace_id, source_id, target_id) DO UPDATE SET
                    description = excluded.description,
                    weight      = excluded.weight,
                    updated_at  = now()
                """,
                (
                    self.workspace_id,
                    source_node_id,
                    target_node_id,
                    edge_data.get("description"),
                    edge_data.get("weight"),
                ),
            )

    async def index_done_callback(self):
        # NetworkXStorage writes the whole .graphml file here. CockroachDB
        # already persisted every upsert individually above, so there's
        # nothing batched left to flush — this is just a no-op for
        # interface compatibility with callers that await it.
        pass
