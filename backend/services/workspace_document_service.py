"""
Workspace-aware document processing service.

Replaces MultiDocumentService / DocumentService for authenticated requests.
All paths are scoped to data/users/{user_id}/cases/{case_id}/ — no global
data/working, data/output, or data/cache directories are used.

The MMKGBuilder and GraphRAGQuery pipeline files are NOT modified — they
accept working_dir / output_dir as constructor parameters, which we supply
from UserWorkspace.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import UTC
from pathlib import Path

import networkx as nx

from backend.auth.workspace import UserWorkspace
from backend.builder import MMKGBuilder
from backend.compliance.evidence_engine import EvidenceEngine
from backend.config import MMKG_NAME
from backend.retrieval.query import GraphRAGQuery

logger = logging.getLogger(__name__)


async def _update_case_status(user_id: str, case_id: str, status: str) -> None:
    """Best-effort update of public.cases.status — never raises."""
    try:
        from backend.auth.supabase_client import get_supabase_client
        supabase = await get_supabase_client()
        await (
            supabase
            .from_("cases")
            .update({"status": status})
            .eq("id", case_id)
            .eq("user_id", user_id)
            .execute()
        )
        logger.info("Case %s status → %s", case_id, status)
    except Exception as exc:
        logger.warning("Could not update case status for %s: %s", case_id, exc)


class WorkspaceDocumentService:
    """Process documents inside a user's isolated workspace."""

    def __init__(self, user_id: str, case_id: str):
        self.user_id = user_id
        self.case_id = case_id
        self.ws = UserWorkspace(user_id=user_id, case_id=case_id).ensure()

    # ------------------------------------------------------------------
    # Upload + pipeline
    # ------------------------------------------------------------------

    async def process_documents(
        self,
        file_paths: list[str],
        file_results: list | None = None,
    ) -> dict:
        """Run MMKGBuilder over *file_paths* inside the user workspace.

        Parameters
        ----------
        file_paths:
            Absolute or relative paths to files already saved inside
            ``self.ws.uploads/``.
        file_results:
            Existing save-phase metadata list (passed through to response).

        Returns
        -------
        dict
            Processing summary including per-file status and graph summary.
        """
        if not file_paths:
            raise ValueError("No files supplied.")

        # Each case gets its own builder instance with isolated dirs
        builder = MMKGBuilder(
            working_dir=str(self.ws.working),
            output_dir=str(self.ws.output),
            mmkg_name=MMKG_NAME,
        )

        processed:    list[str]  = []
        failed:       list[dict] = []
        proc_results: list[dict] = []

        for file_path in file_paths:
            fname = Path(file_path).name
            logger.info("⚙️  Processing %s → workspace %s", fname, self.ws.root)
            try:
                await builder.index(file_path)
                processed.append(fname)
                proc_results.append({"file": fname, "status": "processed"})
            except Exception as exc:
                logger.error("❌ Failed %s: %s", fname, exc)
                failed.append({"file": fname, "error": str(exc)})
                proc_results.append({"file": fname, "status": "failed", "error": str(exc)})

        graph_summary = self._graph_summary()

        # Persist a report.json in output/
        self._save_report_json(graph_summary, proc_results)

        # Mark case status in DB: completed if all files processed, failed otherwise
        final_status = "completed" if len(failed) == 0 else "failed"
        await _update_case_status(self.user_id, self.case_id, final_status)

        return {
            "success":         len(failed) == 0,
            "total_uploaded":  len(file_paths),
            "total_processed": len(processed),
            "total_failed":    len(failed),
            "processed":       processed,
            "failed":          failed,
            "file_results":    proc_results,
            "knowledge_graph": graph_summary,
            "workspace": {
                "output_dir": str(self.ws.output),
                "graph_path": str(self.ws.graph_path),
            },
        }

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    async def query(self, question: str, top_k: int = 10) -> dict:
        """Run a GraphRAG query scoped to this user's workspace.

        Raises
        ------
        FileNotFoundError
            If no graph exists yet for this case.
        """
        if not self.ws.graph_exists():
            raise FileNotFoundError(
                f"No knowledge graph found for this case. "
                f"Upload and process documents first. "
                f"(Expected: {self.ws.graph_path})"
            )

        start = time.time()

        # GraphRAGQuery reads graph from working_dir and embeddings from output_dir
        query_engine = GraphRAGQuery(
            working_dir=str(self.ws.working),
            embedding_path=str(self.ws.output / f"{MMKG_NAME}_emb.npy"),
        )

        from backend.config import QueryParam
        # Create a per-request param copy to avoid mutating the global class
        class _Param:
            top_k = top_k
            response_type = QueryParam.response_type
            local_max_token_for_local_context = QueryParam.local_max_token_for_local_context
            number_of_mmentities = QueryParam.number_of_mmentities
            local_max_token_for_text_unit = QueryParam.local_max_token_for_text_unit

        result = await query_engine.query(question, param=_Param, return_context=True)

        answer            = result["answer"]
        retrieval_context = result["retrieval"]
        processing_time   = round(time.time() - start, 2)

        evidence = EvidenceEngine().collect(
            retrieval_context=retrieval_context,
            graph=query_engine.graph,
            text_chunks=query_engine.text_chunks,
            image_data=query_engine.image_data,
        )

        return {
            "answer":                    answer,
            "evidence":                  evidence,
            "processing_time_seconds":   processing_time,
            "graph": {
                "nodes": query_engine.graph.number_of_nodes(),
                "edges": query_engine.graph.number_of_edges(),
            },
        }

    # ------------------------------------------------------------------
    # Graph summary / report
    # ------------------------------------------------------------------

    def _graph_summary(self) -> dict:
        """Read the workspace graph and return node/edge/type counts."""
        # Try workspace-specific path first, then legacy MMKG_NAME path
        candidates = [
            self.ws.graph_path,
            self.ws.output / f"{MMKG_NAME}.graphml",
        ]
        graph_path = next((p for p in candidates if p.exists()), None)

        if graph_path is None:
            return {"available": False}

        try:
            G             = nx.read_graphml(str(graph_path))
            entity_types: dict = {}
            for _, data in G.nodes(data=True):
                etype = data.get("entity_type", "UNKNOWN").replace('"', "")
                entity_types[etype] = entity_types.get(etype, 0) + 1
            return {
                "available":    True,
                "graph_path":   str(graph_path),
                "nodes":        G.number_of_nodes(),
                "edges":        G.number_of_edges(),
                "entity_types": entity_types,
            }
        except Exception as exc:
            logger.warning("Could not read graph for summary: %s", exc)
            return {"available": False, "error": str(exc)}

    def _save_report_json(self, graph_summary: dict, file_results: list) -> None:
        """Persist a report.json in the workspace output directory."""
        from datetime import datetime
        report = {
            "generated_at":    datetime.now(tz=UTC).isoformat(),
            "user_id":         self.ws.user_id,
            "case_id":         self.ws.case_id,
            "knowledge_graph": graph_summary,
            "file_results":    file_results,
        }
        try:
            self.ws.report_path.write_text(
                json.dumps(report, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning("Could not write report.json: %s", exc)
