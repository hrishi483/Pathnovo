"""
Reproduces the worked example from the design doc (section 13) end to end:

    "What changed around the valve?"
    -> find_entities("valve")
    -> get_nearby_changes(entity_id, radius=100)
    -> synthesized answer

This does NOT call the Anthropic API -- it's a deterministic walkthrough of
the retrieval layer (store + tools) so you can verify the data model and
spatial query work correctly before wiring up the LLM in agent.py.

Run with:  python -m delta_agent.demo
"""

from __future__ import annotations

import json

from src.chat.ingest import ingest_geometry_delta, ingest_text_delta, ingest_vlm_delta, register_entity
from src.chat.store import DeltaStore
from src.chat.tools import find_entities, get_nearby_changes


def build_example_store() -> tuple[DeltaStore, str]:
    store = DeltaStore()
    pair_id = "pair_123"

    valve = register_entity(
        store, pair_id, name="Valve V-101", type_="valve", page=4,
        bbox=(120, 340, 180, 390), aliases=["V-101", "valve"],
    )
    line_20 = register_entity(
        store, pair_id, name="Line L-20", type_="line", page=4,
        bbox=(0, 330, 120, 400), aliases=["L-20"],
    )
    line_25 = register_entity(
        store, pair_id, name="Line L-25", type_="line", page=4,
        bbox=(180, 330, 300, 400), aliases=["L-25"],
    )
    bypass = register_entity(
        store, pair_id, name="Bypass Line", type_="line", page=4,
        bbox=(200, 300, 260, 340), aliases=["bypass"],
    )

    # delta_text
    ingest_text_delta(
        store, pair_id, page=4,
        text_delta="Connection of Valve V-101 changed from Line L-20 to Line L-25.",
        entity=valve, related_entities=[line_20, line_25],
        bbox=(0, 330, 300, 400),
    )

    # delta_geometry: valve moved right by 60px
    ingest_geometry_delta(
        store, pair_id, page=4, entity=valve,
        old_position=(120, 340), new_position=(180, 340),
        old_bbox=(120, 340, 180, 390), new_bbox=(180, 340, 240, 390),
        event="moved",
    )
    # delta_geometry: bypass line added near the valve
    ingest_geometry_delta(
        store, pair_id, page=4, entity=bypass,
        old_bbox=None, new_bbox=(200, 300, 260, 340), event="added",
    )

    # delta_vlm_comparison
    ingest_vlm_delta(
        store, pair_id, page=4,
        vlm_text=(
            "The valve appears to have been relocated to the right side of the "
            "pipeline. A new bypass connection is visible near the valve."
        ),
        bbox=(20, 240, 280, 490), entity=valve, related_entities=[bypass],
    )

    return store, pair_id


def synthesize_answer(entity_name: str, nearby: dict) -> str:
    """Very small non-LLM synthesizer, just to prove the evidence bundle is
    complete and coherent. Replace with a Claude call (see agent.py) for a
    real natural-language answer."""
    lines = []
    for change in nearby["nearby_changes"]:
        lines.append(f"- [{change['type']}] {change['summary']}")
    return f"Changes found around {entity_name}:\n" + "\n".join(lines)


def main() -> None:
    store, pair_id = build_example_store()

    question = "What changed around the valve?"
    print(f"User question: {question!r}\n")

    # 1. find_entities("valve")
    candidates = find_entities(store, pair_id, "valve")
    print("Tool call: find_entities(query='valve')")
    print(json.dumps(candidates, indent=2))

    target = candidates[0]
    print(f"\n-> Disambiguated target: {target['entity_name']} (confidence {target['confidence']})\n")

    # 2. get_nearby_changes(entity_id, radius=100)
    nearby = get_nearby_changes(store, target["entity_id"], radius=100)
    print("Tool call: get_nearby_changes(entity_id=%r, radius=100)" % target["entity_id"])
    print(json.dumps(nearby, indent=2))

    # 3. synthesize
    print("\nFinal answer:")
    print(synthesize_answer(target["entity_name"], nearby))


if __name__ == "__main__":
    main()
