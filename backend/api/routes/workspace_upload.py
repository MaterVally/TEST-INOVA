"""
Workspace-aware upload route.

Replaces the global upload.py for authenticated requests.
Files are saved to:
    data/users/{user_id}/cases/{case_id}/uploads/

The case_id is generated here and returned to the client.
All pipeline processing happens inside the user's workspace.

The original upload.py is NOT modified.
"""
from __future__ import annotations

import logging
import re
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from backend.auth.dependencies import get_current_user
from backend.auth.middleware.jwt_middleware import AuthContext
from backend.auth.workspace import UserWorkspace
from backend.services.workspace_document_service import WorkspaceDocumentService

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/upload",
    tags=["Document Upload"],
)

SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".xlsx", ".xls",
    ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff",
    ".mp3", ".wav", ".m4a", ".flac", ".ogg",
}

_UNSAFE = re.compile(r"[^\w.\-]")


def _sanitize(name: str) -> str:
    name = Path(name).name
    name = _UNSAFE.sub("_", name)
    return name or "unnamed_file"


@router.post("/")
async def upload_documents(
    files: list[UploadFile] = File(...),
    auth: AuthContext = Depends(get_current_user),
):
    """Upload documents into the authenticated user's workspace.

    - Generates a new case_id for this upload session.
    - Saves files to data/users/{user_id}/cases/{case_id}/uploads/
    - Runs the MMKGBuilder pipeline inside the same workspace.
    - Graph output lands in data/users/{user_id}/cases/{case_id}/output/

    user_id is taken from the JWT — never from the request.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    # Create an isolated workspace for this upload session
    case_id = str(uuid.uuid4())
    ws      = UserWorkspace(user_id=auth.user_id, case_id=case_id).ensure()

    logger.info(
        "📁 New upload — user=%s case=%s files=%d",
        auth.user_id, case_id, len(files),
    )

    start = time.time()

    # ── Save files ──────────────────────────────────────────────────────
    file_results: list = []
    saved_paths:  list = []

    for upload in files:
        original_name = upload.filename or "unnamed"
        safe_name     = _sanitize(original_name)
        extension     = Path(safe_name).suffix.lower()

        if extension not in SUPPORTED_EXTENSIONS:
            file_results.append({
                "filename": original_name,
                "status":   "skipped",
                "reason":   f"Unsupported file type '{extension}'",
            })
            continue

        destination = ws.uploads / safe_name

        # Deduplicate filenames within the same case
        if destination.exists():
            stem        = Path(safe_name).stem
            safe_name   = f"{stem}_{uuid.uuid4().hex[:6]}{extension}"
            destination = ws.uploads / safe_name

        try:
            content = await upload.read()
            destination.write_bytes(content)
            saved_paths.append(str(destination))
            file_results.append({
                "filename":   original_name,
                "stored_as":  safe_name,
                "size_bytes": destination.stat().st_size,
                "status":     "saved",
            })
            logger.info("✅ Saved %s → %s", original_name, destination)
        except Exception as exc:
            logger.error("❌ Failed to save %s: %s", original_name, exc)
            file_results.append({
                "filename": original_name,
                "status":   "failed",
                "reason":   f"Could not save file: {exc}",
            })

    if not saved_paths:
        raise HTTPException(
            status_code=400,
            detail="No supported files could be saved. Check formats and try again.",
        )

    # ── Process through workspace-scoped pipeline ────────────────────────
    svc       = WorkspaceDocumentService(user_id=auth.user_id, case_id=case_id)
    kg_result = await svc.process_documents(
        file_paths=saved_paths,
        file_results=file_results,
    )

    # Merge per-file processing status back into file_results
    processing_map = {r["file"]: r for r in kg_result.pop("file_results", [])}
    for entry in file_results:
        name = entry.get("stored_as") or entry.get("filename", "")
        if name in processing_map:
            entry["processing"] = processing_map[name].get("status", "unknown")
            if "error" in processing_map[name]:
                entry["error"] = processing_map[name]["error"]

    processing_time = round(time.time() - start, 2)

    return {
        "success":                 kg_result.get("total_processed", 0) > 0,
        "case_id":                 case_id,
        "user_id":                 auth.user_id,
        "workspace":               str(ws.root),
        "message":                 "Enterprise Knowledge Graph processing complete.",
        "files":                   file_results,
        "processing_time_seconds": processing_time,
        "knowledge_graph":         kg_result,
    }
