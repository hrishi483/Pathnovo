"""FastAPI entry point for deterministic document ingestion."""

from __future__ import annotations

import logging
import os
import tempfile
import time
import json
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from src.canonical.model import DocumentCanonicalRepresentation
from src.chat.agent import DeltaAgent
from src.chat.artifacts import discover_pairs
from src.delta.compare import DocumentComparer
from src.ingest.base import AdapterRegistry, FormatAdapter
from src.ingest.dwg import DwgAdapter, DwgConfig
from src.ingest.pdf_native import NativePdfAdapter, NativePdfConfig
from src.observability.logging import configure_logging
from src.viewer.reports import discover_reports, resolve_image_file, resolve_report_file

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATIC_ROOT = PROJECT_ROOT / "static"
TEMPLATE_ROOT = PROJECT_ROOT / "templates"
logger = logging.getLogger(__name__)


class IngestResponse(BaseModel):
    document_id: str
    # canonical_representation: DocumentCanonicalRepresentation
    # debug_artifacts: dict[str, str]


class ChatRequest(BaseModel):
    question: str
    document_pair_id: str | None = None


class ChatResponse(BaseModel):
    document_pair_id: str
    answer: str
    tool_calls: list[dict[str, Any]]
    usage_metadata: dict[str, Any] | None = None
    model: str | None = None


def _output_root() -> Path:
    return Path(
        os.getenv("DELTA_CHAT_OUTPUT_DIR", PROJECT_ROOT / "data")
    ).resolve()


def _get_or_create_agent(
    cache: dict[str, DeltaAgent],
    artifact_root: Path,
    document_pair_id: str | None,
) -> tuple[DeltaAgent, str]:
    if document_pair_id and document_pair_id in cache:
        return cache[document_pair_id], document_pair_id
    agent, resolved_pair_id = DeltaAgent.from_artifacts(
        artifact_root, document_pair_id
    )
    cache[resolved_pair_id] = agent
    return agent, resolved_pair_id

def _default_adapter_registry() -> AdapterRegistry:
    zoom = float(os.getenv("DELTA_CHAT_RENDER_ZOOM", "2.0"))
    return AdapterRegistry(
        [
            NativePdfAdapter(
                NativePdfConfig(
                    zoom=zoom,
                    min_residual_area=int(
                        os.getenv("DELTA_CHAT_MIN_RESIDUAL_AREA", "64")
                    ),
                )
            ),
            DwgAdapter(DwgConfig(zoom=zoom)),
        ]
    )


async def _extract_canonical_from_upload(
    upload: UploadFile,
    registry: AdapterRegistry,
    output_root: Path,
    *,
    label: str = "upload",
) -> DocumentCanonicalRepresentation:
    filename = Path(upload.filename or "input.bin").name
    try:
        adapter = registry.resolve_for_filename(filename)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=str(exc),
        ) from exc

    output_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    logger.info(
        "ingest[%s] start filename=%s adapter=%s",
        label,
        filename,
        type(adapter).__name__,
    )

    with tempfile.TemporaryDirectory(prefix="delta-chat-") as temporary:
        upload_path = Path(temporary) / filename
        write_started = time.perf_counter()
        bytes_written = 0
        with upload_path.open("wb") as destination:
            while chunk := await upload.read(1024 * 1024):
                destination.write(chunk)
                bytes_written += len(chunk)
        logger.info(
            "ingest[%s] upload written bytes=%s elapsed=%.3fs",
            label,
            bytes_written,
            time.perf_counter() - write_started,
        )

        extract_started = time.perf_counter()
        canonical = await run_in_threadpool(
            adapter.extract_canonical, upload_path, output_root
        )
        logger.info(
            "ingest[%s] extract done document_id=%s text=%s geometry=%s "
            "extract_elapsed=%.3fs total_elapsed=%.3fs",
            label,
            canonical.document_id,
            len(canonical.text_elements),
            len(canonical.geometry_elements),
            time.perf_counter() - extract_started,
            time.perf_counter() - started,
        )
        return canonical


def create_app(
    adapter: FormatAdapter | AdapterRegistry | None = None,
    comparer: DocumentComparer | None = None,
) -> FastAPI:
    configure_logging()
    app = FastAPI(
        title="delta-chat",
        version="0.1.0",
        description="Loss-aware canonical extraction for engineering documents",
    )
    if isinstance(adapter, AdapterRegistry):
        registry = adapter
    elif adapter is not None:
        registry = AdapterRegistry([adapter])
    else:
        registry = _default_adapter_registry()
    document_comparer = comparer or DocumentComparer()
    agent_cache: dict[str, DeltaAgent] = {}
    app.mount("/static", StaticFiles(directory=STATIC_ROOT), name="static")
    templates = Jinja2Templates(directory=TEMPLATE_ROOT)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/", include_in_schema=False)
    async def index(request: Request):
        return templates.TemplateResponse(
            request,
            "index.html",
        )

    @app.get("/chat", include_in_schema=False)
    async def chat_page(request: Request):
        return templates.TemplateResponse(request, "chat.html")

    @app.get("/api/chat/pairs")
    async def list_chat_pairs() -> dict[str, Any]:
        artifact_root = _output_root()
        artifact_root.mkdir(parents=True, exist_ok=True)
        return {"pairs": discover_pairs(artifact_root)}

    @app.post("/api/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest) -> ChatResponse:
        question = request.question.strip()
        if not question:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Question must not be empty",
            )
        artifact_root = _output_root()
        try:
            agent, pair_id = _get_or_create_agent(
                agent_cache, artifact_root, request.document_pair_id
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            ) from exc

        try:
            result = await run_in_threadpool(
                agent.ask_with_trace, pair_id, question
            )
        except RuntimeError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(exc),
            ) from exc
        except Exception as exc:
            logger.exception("chat request failed: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Agent failed: {exc}",
            ) from exc

        return ChatResponse(
            document_pair_id=pair_id,
            answer=result["answer"],
            tool_calls=result["tool_calls"],
            usage_metadata=result.get("usage_metadata"),
            model=result.get("model"),
        )

    @app.get("/api/reports")
    async def list_reports() -> dict[str, Any]:
        artifact_root = _output_root()
        artifact_root.mkdir(parents=True, exist_ok=True)
        return discover_reports(artifact_root)

    @app.get("/api/reports/{report_id}")
    async def get_report(report_id: str) -> dict[str, Any]:
        artifact_root = _output_root()
        try:
            report_path = resolve_report_file(artifact_root, report_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Report not found") from exc
        try:
            return json.loads(report_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Invalid JSON report: {report_path.name}",
            ) from exc

    @app.get("/api/images/{image_path:path}")
    async def get_image(image_path: str) -> FileResponse:
        artifact_root = _output_root()
        try:
            image_file = resolve_image_file(artifact_root, image_path)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail="Image not found") from exc
        if not image_file.is_file():
            raise HTTPException(status_code=404, detail="Image not found")
        return FileResponse(image_file)

    @app.post("/ingest", response_model=IngestResponse)
    async def ingest(file: Annotated[UploadFile, File()]) -> IngestResponse:
        output_root = _output_root()
        try:
            canonical = await _extract_canonical_from_upload(
                file, registry, output_root
            )
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            ) from exc
        finally:
            await file.close()

        return IngestResponse(
            document_id=canonical.document_id,
            # canonical_representation=canonical,
            # debug_artifacts={str(key): str(value) for key, value in artifacts.items()},
        )

    @app.post("/compare")
    async def compare(
        baseline: Annotated[UploadFile, File()],
        revision: Annotated[UploadFile, File()],
    ) -> dict[str, Any]:
        output_root = _output_root()
        started = time.perf_counter()
        logger.info(
            "compare request start baseline=%s revision=%s",
            baseline.filename,
            revision.filename,
        )
        try:
            baseline_output_path = output_root / "baseline"
            baseline_canonical = await _extract_canonical_from_upload(
                baseline, registry, baseline_output_path, label="baseline"
            )

            revision_output_path = output_root / "revision"
            revision_canonical = await _extract_canonical_from_upload(
                revision, registry, revision_output_path, label="revision"
            )
            logger.info(
                "compare alignment start baseline_id=%s revision_id=%s "
                "baseline_text=%s baseline_geometry=%s "
                "revision_text=%s revision_geometry=%s "
                "geometry_pair_budget=%s",
                baseline_canonical.document_id,
                revision_canonical.document_id,
                len(baseline_canonical.text_elements),
                len(baseline_canonical.geometry_elements),
                len(revision_canonical.text_elements),
                len(revision_canonical.geometry_elements),
                len(baseline_canonical.geometry_elements)
                * len(revision_canonical.geometry_elements),
            )
            align_started = time.perf_counter()

            delta_output_path = output_root / "delta"
            result = await run_in_threadpool(
                document_comparer.compare,
                baseline_canonical,
                revision_canonical,
                delta_output_path,
            )
            logger.info(
                "compare request complete align_elapsed=%.3fs total_elapsed=%.3fs",
                time.perf_counter() - align_started,
                time.perf_counter() - started,
            )
            return result
        except (ValueError, RuntimeError) as exc:
            logger.exception(
                "compare request failed after %.3fs: %s",
                time.perf_counter() - started,
                exc,
            )
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            ) from exc
        finally:
            await baseline.close()
            await revision.close()

    return app


app = create_app()
