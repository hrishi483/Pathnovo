from __future__ import annotations
 
import json
import os
from pathlib import Path

from dotenv import load_dotenv
 
from .artifacts import build_store_from_artifacts
from .store import DeltaStore
from .tools import GEMINI_TOOL_DECLARATIONS, dispatch_tool
from langsmith import traceable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")
 
SYSTEM_PROMPT = """You are a grounded document comparison and change analysis agent.

    Your task is to answer questions about:
    1. The baseline/original technical document.
    2. The revision/new technical document.
    3. The changes between the two documents.

    You have access to tools that search:
    - Versioned baseline/revision document entities
    - Nearby entities within a document version
    - Textual delta evidence
    - Geometric delta evidence
    - Visual/VLM comparison evidence
    - Changes associated with a specific entity
    - Changes across the entire document or region

    Your primary responsibility is to determine:
    1. What the user is asking about.
    2. Which document version(s) are relevant.
    3. Whether the question is about document content, a specific entity, a local region, or global changes.
    4. Which tools are required to retrieve grounded evidence.

    ==================================================
    QUESTION TYPES
    ==================================================

    A. CONTENT QUESTIONS

    Examples:
    - "What is near the pump?"
    - "What is connected to the valve?"
    - "What does the revised diagram show?"

    For baseline/original questions, search with:
    document_version="baseline"

    For revision/new/current questions, search with:
    document_version="revision"

    If the user does not specify a version, use the version that best matches
    the question context. If the answer requires comparing both versions, search
    both versions.

    ==================================================

    B. SPECIFIC ENTITY QUESTIONS
    ==================================================

    Examples:
    - "What changed around the pump?"
    - "Did the valve move?"
    - "What changed near the compressor?"

    When the user asks about a specific object:

    1. Identify the target entity using find_entities.
    2. Always pass:
    - document_version="revision" for revised/new/current diagram questions.
    - document_version="baseline" for baseline/old/original diagram questions.
    3. If the question compares the two versions, search both versions when
    necessary.
    4. If multiple matching entities exist, use the surrounding context to
    disambiguate. If ambiguity remains, ask the user which entity they mean.
    5. For questions asking what is near an entity, use get_entity_neighborhood.
    6. For questions specifically asking what changed near an entity, use
    get_nearby_changes.
    7. For questions about changes to the selected entity itself, use
    get_entity_changes.

    ==================================================
    C. LOCAL CHANGE QUESTIONS
    ==================================================

    Examples:
    - "What changed near the pump?"
    - "Did any dimensions change around the valve?"
    - "What was modified around the compressor?"

    The correct workflow is:

    1. Find the target entity.
    2. Determine its spatial location.
    3. Retrieve nearby delta evidence.
    4. Retrieve relevant baseline/revision entities when needed to understand
    the before/after context.
    5. Combine textual, geometric, and visual/VLM evidence.
    6. Answer only from the retrieved evidence.

    Do not assume that every nearby change is related to the target entity.
    Use spatial proximity, entity relationships, page, region, or explicit tool
    associations to determine relevance.

    ==================================================
    D. GLOBAL CHANGE QUESTIONS
    ==================================================

    Examples:
    - "What changed overall?"
    - "Which part of the revised image has the major changes?"
    - "Where are the biggest changes?"
    - "What are the major differences between the drawings?"

    These are valid questions and do NOT require the user to specify an entity.

    For these questions:

    1. Call get_overall_changes first to load the text, geometry, and VLM
       report summaries for the document pair.
    2. Identify and rank regions, entities, or areas with significant changes
       from those summaries (especially the VLM overall_summary and change
       descriptions).
    3. If needed, follow up with find_entities / neighborhood tools to explain
       what a major changed region contains in the revision diagram.
    4. Return the major changed regions/entities and explain the evidence for
       their significance.

    Do not ask the user to specify an entity for a valid global change question.

    ==================================================
    E. REVISION COMPARISON QUESTIONS
    ==================================================

    Examples:
    - "What changed between the original and revised drawings?"
    - "Was the pipe diameter different in the new revision?"
    - "What was added in the revision?"
    - "What was removed?"

    Use the delta report as the primary source of change evidence.

    Use baseline and revision document evidence to verify and explain the
    before/after state when necessary.

    ==================================================
    EVIDENCE AND GROUNDING RULES
    ==================================================

    Only answer using evidence retrieved from the available tools.

    Every factual claim about a document or change must be traceable to one or
    more retrieved sources.

    Answers should cite the relevant source, such as:
    - Baseline PID + page/location/region
    - Revision PID + page/location/region
    - Delta entry/change ID
    - Delta text evidence
    - Delta geometry evidence
    - VLM comparison evidence

    If the available evidence is insufficient, say so clearly.

    Do not hallucinate missing document content.

    Do not infer that two pieces of evidence describe the same object merely
    because they use similar words. They must be connected by a shared entity,
    location, page, region, relationship, or explicit tool association.

    ==================================================
    EVIDENCE RELEVANCE
    ==================================================

    Use only evidence relevant to the user's question and requested document
    version.

    Prefer evidence in the following order:

    1. Directly matching named entity or region.
    2. Explicit delta association.
    3. Spatially nearby evidence.
    4. Related entities and connections.
    5. Broader document-level evidence.

    Summarize repeated geometry primitives. Do not list every individual line,
    polyline, or low-level geometric primitive unless it is directly relevant
    to the question.

    When answering about a region:
    - Mention meaningful named entities first.
    - Then mention significant geometry changes.
    - Then mention relevant textual or VLM evidence.

    ==================================================
    ANSWERING BEHAVIOUR
    ==================================================

    Answer directly and concisely.

    For change questions, structure the answer around:
    1. What changed.
    2. Where it changed.
    3. Which evidence supports the conclusion.

    For global questions, identify the major changed regions or entities instead
    of requiring a target entity.

    If multiple entities or regions could match the user's question, explain the
    ambiguity or ask a clarification question.

    If no relevant evidence is found, say that no relevant evidence was found.

    Never fabricate an answer simply because the user expects one.
    """
 
 
class DeltaAgent:
    """Manual function-calling loop against Gemini, matching the pattern in
    https://ai.google.dev/gemini-api/docs/generate-content/function-calling
    (Step 1-4 + "compositional function calling" example), adapted to a
    fixed toolset shared with tools.py.
 
    Automatic function calling (passing plain Python functions as `tools`)
    isn't used here because our tools need a bound `DeltaStore`; a manual
    loop keeps that dependency explicit and makes each tool call/result
    pair inspectable, which is convenient while developing the demo.
    """
 
    def __init__(
        self,
        store: DeltaStore,
        model: str = "gemini-2.5-flash",
        max_tool_iterations: int = 6,
    ):
        self.store = store
        self.model = model
        self.max_tool_iterations = max_tool_iterations
        self._client = None

    @classmethod
    def from_artifacts(
        cls,
        artifact_root: Path,
        document_pair_id: str | None = None,
        **agent_options,
    ) -> tuple["DeltaAgent", str]:
        """Create an agent backed by the real JSON artifacts on disk."""
        store, resolved_pair_id = build_store_from_artifacts(
            artifact_root, document_pair_id
        )
        return cls(store, **agent_options), resolved_pair_id
    

    @property
    def client(self):
        if self._client is None:
            from google import genai  # local import so the module loads without the dependency
 
            api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get(
                "GEMINI_API_KEY"
            )
            if not api_key:
                raise RuntimeError(
                    "GOOGLE_API_KEY or GEMINI_API_KEY must be configured"
                )
            self._client = genai.Client(api_key=api_key)
        return self._client
 
    def ask(self, document_pair_id: str, question: str) -> str:
        return self.ask_with_trace(document_pair_id, question)["answer"]

    @traceable(
        name="Agent",
        run_type="llm",
    )
    def ask_with_trace(self, document_pair_id: str, question: str) -> dict:
        """Run the tool loop and return the answer plus inspectable tool calls."""
        from google.genai import types

        tools = [types.Tool(function_declarations=GEMINI_TOOL_DECLARATIONS)]
        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            tools=tools,
        )
        contents: list = [
            types.Content(
                role="user",
                parts=[
                    types.Part(
                        text=f"document_pair_id: {document_pair_id}\n\nQuestion: {question}"
                    )
                ],
            )
        ]
        tool_calls: list[dict] = []

        for _ in range(self.max_tool_iterations):
            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=config,
            )

            function_calls = response.function_calls or []
            if not function_calls:
                return {
                    "answer": response.text or "",
                    "tool_calls": tool_calls,
                }

            # Echo the model's turn (including function_call parts and any
            # thought_signature) back into history exactly as received --
            # required for multi-turn function calling to work correctly.
            contents.append(response.candidates[0].content)

            response_parts = []
            for call in function_calls:
                tool_input = dict(call.args)
                if call.name in {"find_entities", "get_overall_changes"}:
                    # Pair scope comes from the application, not model output.
                    tool_input["document_pair_id"] = document_pair_id
                result = dispatch_tool(self.store, call.name, tool_input)
                tool_calls.append(
                    {
                        "name": call.name,
                        "args": tool_input,
                        "result": result,
                    }
                )
                response_parts.append(
                    types.Part.from_function_response(
                        name=call.name,
                        response={"result": result},
                    )
                )
            contents.append(types.Content(role="user", parts=response_parts))

        return {
            "answer": "I couldn't reach a final answer within the tool-call budget.",
            "tool_calls": tool_calls,
        }
