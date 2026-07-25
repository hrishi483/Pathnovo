from types import SimpleNamespace

from src.chat.tracing import (
    merge_usage,
    strip_store_from_inputs,
    usage_from_gemini_response,
)


def test_usage_from_gemini_response_object() -> None:
    response = SimpleNamespace(
        usage_metadata=SimpleNamespace(
            prompt_token_count=120,
            candidates_token_count=45,
            thoughts_token_count=10,
            cached_content_token_count=0,
            total_token_count=175,
        )
    )
    usage = usage_from_gemini_response(response)
    assert usage["input_tokens"] == 120
    assert usage["output_tokens"] == 45
    assert usage["total_tokens"] == 175
    assert usage["output_token_details"]["reasoning"] == 10


def test_usage_from_gemini_response_dict() -> None:
    response = SimpleNamespace(
        usage_metadata={
            "prompt_token_count": 10,
            "candidates_token_count": 4,
            "total_token_count": 14,
        }
    )
    assert usage_from_gemini_response(response) == {
        "input_tokens": 10,
        "output_tokens": 4,
        "total_tokens": 14,
    }


def test_merge_usage_accumulates_turns() -> None:
    total = {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12}
    delta = {
        "input_tokens": 5,
        "output_tokens": 3,
        "total_tokens": 8,
        "output_token_details": {"reasoning": 1},
    }
    merged = merge_usage(total, delta)
    assert merged["input_tokens"] == 15
    assert merged["output_tokens"] == 5
    assert merged["total_tokens"] == 20
    assert merged["output_token_details"]["reasoning"] == 1


def test_strip_store_from_inputs() -> None:
    assert strip_store_from_inputs({"store": object(), "query": "valve"}) == {
        "query": "valve"
    }
