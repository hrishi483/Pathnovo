"""LangSmith helpers for the delta-chat agent (Gemini usage + safe inputs)."""

from __future__ import annotations

from typing import Any


def strip_store_from_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    """Drop the bound DeltaStore from tool traces (not JSON-serializable / huge)."""
    return {key: value for key, value in inputs.items() if key != "store"}


def usage_from_gemini_response(response: Any) -> dict[str, int]:
    """Normalize Gemini ``usage_metadata`` into LangSmith ``usage_metadata`` fields.

    LangSmith recognizes ``input_tokens`` / ``output_tokens`` / ``total_tokens``
    on LLM runs (and can cost-track when ``ls_provider`` + ``ls_model_name`` are set).
    """
    raw = getattr(response, "usage_metadata", None)
    if raw is None:
        return {}

    if isinstance(raw, dict):
        prompt = raw.get("prompt_token_count") or raw.get("prompt_tokens") or 0
        candidates = (
            raw.get("candidates_token_count")
            or raw.get("candidates_tokens")
            or raw.get("completion_tokens")
            or 0
        )
        thoughts = raw.get("thoughts_token_count") or 0
        cached = raw.get("cached_content_token_count") or 0
        total = raw.get("total_token_count") or (prompt + candidates + thoughts)
    else:
        prompt = getattr(raw, "prompt_token_count", None) or 0
        candidates = getattr(raw, "candidates_token_count", None) or 0
        thoughts = getattr(raw, "thoughts_token_count", None) or 0
        cached = getattr(raw, "cached_content_token_count", None) or 0
        total = getattr(raw, "total_token_count", None) or (prompt + candidates + thoughts)

    usage: dict[str, Any] = {
        "input_tokens": int(prompt),
        "output_tokens": int(candidates),
        "total_tokens": int(total),
    }
    if thoughts:
        usage["output_token_details"] = {"reasoning": int(thoughts)}
    if cached:
        usage["input_token_details"] = {"cache_read": int(cached)}
    return usage


def merge_usage(total: dict[str, Any], delta: dict[str, Any]) -> dict[str, Any]:
    """Accumulate token counts across multiple Gemini turns."""
    if not delta:
        return total
    merged = dict(total)
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        merged[key] = int(merged.get(key, 0)) + int(delta.get(key, 0))

    if "output_token_details" in delta:
        details = dict(merged.get("output_token_details") or {})
        for key, value in delta["output_token_details"].items():
            details[key] = int(details.get(key, 0)) + int(value)
        merged["output_token_details"] = details

    if "input_token_details" in delta:
        details = dict(merged.get("input_token_details") or {})
        for key, value in delta["input_token_details"].items():
            details[key] = int(details.get(key, 0)) + int(value)
        merged["input_token_details"] = details

    return merged


def attach_usage_to_current_run(usage: dict[str, Any]) -> None:
    """Write aggregated usage onto the active LangSmith run (parent chain)."""
    if not usage:
        return
    try:
        from langsmith.run_helpers import get_current_run_tree
    except ImportError:
        return

    run_tree = get_current_run_tree()
    if run_tree is None:
        return
    run_tree.set(outputs={**(run_tree.outputs or {}), "usage_metadata": usage})
    # Also stash on metadata so dashboards that read ls usage fields can find it.
    metadata = dict(run_tree.extra.get("metadata") or {})
    metadata["usage_metadata"] = usage
    run_tree.extra["metadata"] = metadata
