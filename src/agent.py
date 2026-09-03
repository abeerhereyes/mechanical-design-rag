"""LangGraph agent for deterministic calculations and grounded retrieval."""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Literal, Optional, TypedDict, Union

from langgraph.graph import END, StateGraph

from src.calc_tools import (
    bearing_l10_life_hours,
    bolt_joint_load,
    bolt_preload,
    spring_max_shear_stress,
    spring_rate,
    tensile_stress_area_metric,
    weld_primary_shear,
)
from src.llm import OllamaClient


class AgentState(TypedDict, total=False):
    query: str
    intent: Literal["calculate", "retrieve", "clarify"]
    calc_type: Optional[str]
    params: dict
    missing_params: list[str]
    retrieved_chunks: list[dict]
    answer: str
    citations: list[dict]


CALC_PATTERNS = {
    "spring_rate": [r"spring rate", r"spring stiffness"],
    "spring_stress": [r"spring.*(?:stress|wahl)", r"shear stress.*spring"],
    "bolt_preload": [r"(?:calculate|compute|find|what is|how much).*preload", r"proof load"],
    "bolt_joint_load": [r"(?:bolt|joint).*(?:load sharing|joint load|separat)"],
    "tensile_stress_area": [r"tensile stress area"],
    "weld_shear": [r"(?:calculate|compute|find|what is).*weld.*shear", r"direct shear.*weld"],
    "bearing_life": [r"(?:calculate|compute|find|what is).*bearing life", r"\bl10\b.*(?:hours|rpm)"],
}

CALC_REQUIRED_PARAMS = {
    "spring_rate": ["g_mpa", "d_wire_mm", "d_mean_mm", "n_active"],
    "spring_stress": ["force_n", "d_mean_mm", "d_wire_mm"],
    "bolt_preload": ["at_mm2", "proof_strength_mpa"],
    "bolt_joint_load": ["fi_n", "kb", "km", "external_load_n"],
    "tensile_stress_area": ["dp_mm", "pitch_mm"],
    "weld_shear": ["force_n", "leg_size_mm", "weld_length_mm"],
    "bearing_life": ["c_rating_n", "p_equiv_n", "speed_rpm"],
}

PARAM_PATTERNS = {
    "g_mpa": [r"(?:\bG\b|shear modulus)\s*(?:=|of|is)?\s*(-?\d+(?:\.\d+)?)"],
    "d_wire_mm": [r"(?:wire(?: diameter)?|\bd\b)\s*(?:=|of|is)?\s*(-?\d+(?:\.\d+)?)"],
    "d_mean_mm": [r"(?:mean(?: coil)? (?:diameter|D)|\bD\b)\s*(?:=|of|is)?\s*(-?\d+(?:\.\d+)?)"],
    "n_active": [r"(-?\d+(?:\.\d+)?)\s*active coils?", r"(?:active coils?|Na)\s*(?:=|of|is)?\s*(-?\d+(?:\.\d+)?)"],
    "force_n": [r"(?:force|load|\bF\b)\s*(?:=|of|is)?\s*(-?\d+(?:\.\d+)?)"],
    "at_mm2": [r"(?:stress area|\bAt\b)\s*(?:=|of|is)?\s*(-?\d+(?:\.\d+)?)"],
    "proof_strength_mpa": [r"(?:proof strength|\bSp\b)\s*(?:=|of|is)?\s*(-?\d+(?:\.\d+)?)"],
    "fi_n": [r"(?:preload|\bFi\b)\s*(?:=|of|is)?\s*(-?\d+(?:\.\d+)?)"],
    "kb": [r"\bkb\b\s*(?:=|of|is)?\s*(-?\d+(?:\.\d+)?)"],
    "km": [r"\bkm\b\s*(?:=|of|is)?\s*(-?\d+(?:\.\d+)?)"],
    "external_load_n": [r"(?:external load|\bP\b)\s*(?:=|of|is)?\s*(-?\d+(?:\.\d+)?)"],
    "dp_mm": [r"(?:major diameter|\bdp\b|M)\s*(?:=|of|is)?\s*(-?\d+(?:\.\d+)?)"],
    "pitch_mm": [r"(?:pitch|[x×])\s*(?:=|of|is)?\s*(-?\d+(?:\.\d+)?)"],
    "leg_size_mm": [r"(?:leg size|\bh\b)\s*(?:=|of|is)?\s*(-?\d+(?:\.\d+)?)"],
    "weld_length_mm": [r"(?:weld length|length|\bL\b)\s*(?:=|of|is)?\s*(-?\d+(?:\.\d+)?)"],
    "c_rating_n": [r"(?:dynamic (?:load )?rating|C rating|\bC\b)\s*(?:=|of|is)?\s*(-?\d+(?:\.\d+)?)"],
    "p_equiv_n": [r"(?:equivalent (?:dynamic )?load|\bP\b)\s*(?:=|of|is)?\s*(-?\d+(?:\.\d+)?)"],
    "speed_rpm": [r"(?:speed|n)\s*(?:=|of|is)?\s*(-?\d+(?:\.\d+)?)\s*(?:rpm)?"],
}

CALC_CITATIONS = {
    "spring_rate": ("springs.md", "S1", 1),
    "spring_stress": ("springs.md", "S2", 2),
    "bolt_preload": ("bolts.md", "B1", 1),
    "bolt_joint_load": ("bolts.md", "B3", 3),
    "tensile_stress_area": ("bolts.md", "B2", 2),
    "weld_shear": ("welds.md", "W2", 2),
    "bearing_life": ("bearings.md", "BR1/BR4", 1),
}


def classify_intent(state: AgentState) -> AgentState:
    query = state["query"].lower()
    conceptual = re.search(
        r"\b(?:formula|equation|explain|why|how (?:do|can) i (?:calculate|compute))\b",
        query,
    )
    for calc_type, patterns in CALC_PATTERNS.items():
        if any(re.search(pattern, query, re.IGNORECASE) for pattern in patterns):
            if conceptual:
                return {**state, "intent": "retrieve", "calc_type": None}
            return {**state, "intent": "calculate", "calc_type": calc_type}
    return {**state, "intent": "retrieve"}


def extract_params(state: AgentState) -> AgentState:
    calc_type = state["calc_type"]
    required = CALC_REQUIRED_PARAMS[calc_type]
    params = dict(state.get("params") or {})
    query = state["query"]

    for name in required:
        if name in params:
            continue
        for pattern in PARAM_PATTERNS[name]:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                params[name] = float(match.group(1))
                break

    # Natural ordered input remains supported when no labels were recognized.
    if not params:
        numbers = [float(value) for value in re.findall(r"-?\d+(?:\.\d+)?", query)]
        params.update(dict(zip(required, numbers)))

    if calc_type == "bolt_preload":
        params["reused"] = not bool(re.search(r"\b(permanent|new)\b", query, re.IGNORECASE))
    if calc_type == "bearing_life":
        params["bearing_type"] = (
            "roller" if re.search(r"\broller\b", query, re.IGNORECASE) else "ball"
        )

    missing = [name for name in required if name not in params]
    return {
        **state,
        "params": params,
        "missing_params": missing,
        "intent": "clarify" if missing else "calculate",
    }


def clarify_node(state: AgentState) -> AgentState:
    readable = ", ".join(name.replace("_", " ") for name in state["missing_params"])
    return {
        **state,
        "answer": f"I can calculate this once you provide: {readable}.",
        "citations": [],
    }


def _format_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:,.4g}"
    return str(value)


def calculate_node(state: AgentState) -> AgentState:
    calc_type = state["calc_type"]
    params = state["params"]
    functions = {
        "spring_rate": spring_rate,
        "spring_stress": spring_max_shear_stress,
        "bolt_preload": bolt_preload,
        "bolt_joint_load": bolt_joint_load,
        "tensile_stress_area": tensile_stress_area_metric,
        "weld_shear": weld_primary_shear,
        "bearing_life": bearing_l10_life_hours,
    }
    units = {
        "spring_rate": "N/mm",
        "tensile_stress_area": "mm²",
        "weld_shear": "MPa",
    }
    try:
        result = functions[calc_type](**params)
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        return {**state, "answer": f"Invalid calculation input: {exc}", "citations": []}

    if isinstance(result, dict):
        details = ", ".join(
            f"{key.replace('_', ' ')} = {_format_value(value)}"
            for key, value in result.items()
        )
    else:
        details = f"{_format_value(result)} {units.get(calc_type, '')}".strip()

    filename, section, page = CALC_CITATIONS[calc_type]
    citation = {
        "id": f"{filename}::{section}",
        "source": filename,
        "section": section,
        "page": page,
    }
    return {
        **state,
        "answer": f"Calculated result: {details}.",
        "citations": [citation],
    }


class RetrievalEngine:
    """Lazily constructs and caches the local retrieval pipeline."""

    def __init__(
        self,
        corpus_dir: Optional[Union[str, Path]] = None,
        persist_dir: Optional[Union[str, Path]] = None,
    ):
        root = Path(__file__).resolve().parent.parent
        self.corpus_dir = Path(corpus_dir or root / "data" / "corpus")
        self.persist_dir = Path(persist_dir or root / "data" / "chroma")
        self._dense = None
        self._sparse = None
        self._reranker = None

    def _initialize(self) -> None:
        from src.chunker import load_corpus
        from src.dense_retriever import build_dense_index, get_collection
        from src.reranker import LexicalCrossEncoder
        from src.sparse_retriever import SparseIndex

        chunks = load_corpus(str(self.corpus_dir))
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        try:
            dense = get_collection(str(self.persist_dir))
            if dense.count() != len(chunks):
                dense = build_dense_index(chunks, str(self.persist_dir))
        except Exception:
            dense = build_dense_index(chunks, str(self.persist_dir))
        self._dense = dense
        self._sparse = SparseIndex(chunks)
        self._reranker = LexicalCrossEncoder()

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        from src.hybrid_retriever import hybrid_search

        if self._dense is None:
            self._initialize()
        candidates = hybrid_search(
            self._dense, self._sparse, query, k=10, fetch_k=10
        )
        return self._reranker.rerank(query, candidates, top_k=top_k)


class MechanicalDesignAgent:
    def __init__(
        self,
        retriever: Optional[RetrievalEngine] = None,
        llm: Optional[OllamaClient] = None,
    ):
        self.retriever = retriever or RetrievalEngine()
        self.llm = llm or OllamaClient()
        self.graph = self._build_graph()

    def _retrieve_node(self, state: AgentState) -> AgentState:
        chunks = self.retriever.search(state["query"])
        citations = [
            {
                "id": item["id"],
                "source": item["metadata"]["source"],
                "section": item["metadata"]["section"],
                "page": item["metadata"]["page"],
            }
            for item in chunks
        ]
        answer = self.llm.generate(state["query"], chunks)
        return {
            **state,
            "retrieved_chunks": chunks,
            "citations": citations,
            "answer": answer,
        }

    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("classify", classify_intent)
        graph.add_node("extract_params", extract_params)
        graph.add_node("calculate", calculate_node)
        graph.add_node("retrieve", self._retrieve_node)
        graph.add_node("clarify", clarify_node)
        graph.set_entry_point("classify")
        graph.add_conditional_edges(
            "classify",
            lambda state: "extract_params" if state["intent"] == "calculate" else "retrieve",
            {"extract_params": "extract_params", "retrieve": "retrieve"},
        )
        graph.add_conditional_edges(
            "extract_params",
            lambda state: "clarify" if state["intent"] == "clarify" else "calculate",
            {"clarify": "clarify", "calculate": "calculate"},
        )
        graph.add_edge("calculate", END)
        graph.add_edge("retrieve", END)
        graph.add_edge("clarify", END)
        return graph.compile()

    def ask(self, query: str, params: Optional[dict] = None) -> AgentState:
        if not query.strip():
            raise ValueError("Query cannot be empty")
        return self.graph.invoke({"query": query.strip(), "params": params or {}})


def build_graph():
    """Compatibility helper returning the default compiled graph."""
    return MechanicalDesignAgent().graph


if __name__ == "__main__":
    agent = MechanicalDesignAgent()
    print("Mechanical Design RAG. Type 'quit' to exit.")
    while True:
        query = input("\nYou: ").strip()
        if query.lower() in {"quit", "exit"}:
            break
        result = agent.ask(query)
        print("Assistant:", result["answer"])
        for citation in result.get("citations", []):
            print(
                f"  - {citation['source']} — Section {citation['section']} "
                f"(p. {citation['page']})"
            )
