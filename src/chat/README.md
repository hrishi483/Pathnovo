# delta_agent

Implementation of the entity-first, spatial-neighbourhood retrieval design
for the document-delta agent (see the original design doc for the
rationale). Turns three independent pipelines —

- `delta_text`
- `delta_geometry`
- `delta_vlm_comparison`

— into a single unified index, so questions like **"what changed around
the valve?"** are answered by: find the entity → expand a search radius
around its bounding box → pull back every change (text, geometry, VLM)
that falls inside that region → let the agent synthesize an answer.

## Files

| File | Responsibility |
|---|---|
| `models.py` | `Entity`, `ChangeRecord`, bbox math (expand/intersect/distance) |
| `store.py` | `DeltaStore` — in-memory index (swap for Postgres/vector DB later); `find_entities`, `nearby_changes` |
| `ingest.py` | Converters from normalized values → `Entity` / `ChangeRecord` |
| `artifacts.py` | Loader for real baseline/revision canonical JSON and text/geometry/VLM delta reports |
| `tools.py` | Gemini-facing versioned entity, neighborhood, and change tools |
| `agent.py` | `DeltaAgent` — Gemini function-calling loop wired to the tools above |
| `demo.py` | Deterministic, no-API-key walkthrough of the doc's valve example |
| `tests/test_core.py` | pytest coverage of entity matching + spatial query |

## Quick start (no API key needed)

```bash
cd /path/containing/delta_agent
python -m delta_agent.demo
python -m pytest delta_agent/tests -q
```

## Ingesting your real pipeline output

The real artifacts can be loaded directly:

```python
from pathlib import Path
from src.chat.agent import DeltaAgent

agent, pair_id = DeltaAgent.from_artifacts(Path("data/artifacts"))
answer = agent.ask(pair_id, "What is near Pump in the revised diagram?")
print(answer)
```

The loader reads:

- `baseline/{document_id}/text_elements.json`
- `baseline/{document_id}/geometry_elements.json`
- `revision/{document_id}/text_elements.json`
- `revision/{document_id}/geometry_elements.json`
- `delta/results-{baseline_id}-{revision_id}/*_text.json`
- `delta/results-{baseline_id}-{revision_id}/*_geometry.json`
- `delta/results-{baseline_id}-{revision_id}/*_vlm_comparison.json`

```python
from delta_agent.store import DeltaStore
from delta_agent.ingest import register_entity, ingest_text_delta, ingest_geometry_delta, ingest_vlm_delta

store = DeltaStore()
pair_id = "pair_123"

valve = register_entity(store, pair_id, name="Valve V-101", type_="valve",
                         page=4, bbox=(120, 340, 180, 390), aliases=["V-101"])

ingest_text_delta(store, pair_id, page=4,
                   text_delta="Connection of Valve V-101 changed from Line L-20 to Line L-25.",
                   entity=valve)

ingest_geometry_delta(store, pair_id, page=4, entity=valve,
                       old_position=(120, 340), new_position=(180, 340),
                       old_bbox=(120, 340, 180, 390), new_bbox=(180, 340, 240, 390))

ingest_vlm_delta(store, pair_id, page=4, entity=valve, bbox=(20, 240, 280, 490),
                  vlm_text="The valve appears to have been relocated to the right...")
```

Map this onto your actual pipeline's output format inside `ingest.py` —
the field names there mirror the example payloads from the design doc, but
you'll likely need to adjust them (e.g. if your entity/bbox detection is a
separate upstream step, call `register_entity` from that step's output
instead of inventing new entities per delta record).

## Wiring up the LLM agent

```bash
pip install anthropic --break-system-packages
export ANTHROPIC_API_KEY=sk-...
```

```python
from delta_agent.agent import DeltaAgent
agent = DeltaAgent(store)
print(agent.ask(pair_id, "What changed around the valve?"))
```

`DeltaAgent.ask` runs the standard Anthropic tool-use loop: Claude decides
which of the 3 tools to call (and in what order), `dispatch_tool` executes
them against your `DeltaStore`, and results are fed back until Claude
produces a final text answer. The system prompt (in `agent.py`) encodes the
retrieval strategy from the design doc (identify entity → disambiguate if
needed → gather nearby evidence → synthesize, without conflating unrelated
changes).

## Design notes / where to extend

- **Entity matching** (`Entity.matches`) is currently exact/substring/token
  overlap — replace with embedding similarity for fuzzier queries ("the
  thing near the pump") once you have more entity types.
- **`DeltaStore`** is a plain in-memory dict-of-lists index. The public
  query surface (`find_entities`, `nearby_changes`, `changes_for_entity`)
  is what matters — reimplement it against a real database without
  touching `tools.py` or `agent.py`.
- **Radius** defaults to 100 (same units as your bbox coordinates, e.g.
  PDF points or pixels). Expose it as a tool parameter (already done) so
  the agent can widen/narrow the search based on the question.
- Multiple ambiguous entity matches: `find_entities` returns a ranked list
  with confidence scores; the agent's system prompt tells it to
  disambiguate from context or ask the user, per section 6 of the design
  doc.
