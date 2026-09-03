"""Course-aware LangGraph agent with reviewed calculations and cited retrieval."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Literal, Optional, TypedDict

from langgraph.graph import END, StateGraph

from src.calculations import (
    FormulaError,
    execute,
    get_formula,
    list_formulas,
)
from src.llm import GroundedAnswer, OllamaClient


SUPPORTED_COURSES = {
    "mechanical-design": "Mechanical Design",
    "aerodynamics": "Aerodynamics",
    "qrm": "Quality, Reliability, and Maintenance",
}


class AgentState(TypedDict, total=False):
    query: str
    course_id: str
    include_general: bool
    intent: Literal["calculate", "retrieve", "clarify"]
    formula_id: Optional[str]
    calc_type: Optional[str]
    params: Dict[str, Any]
    units: Dict[str, str]
    missing_params: list[str]
    retrieved_chunks: list[dict]
    answer: str
    course_answer: str
    general_answer: Optional[str]
    citations: list[dict]
    warnings: list[str]
    assumptions: list[str]


FORMULA_PATTERNS = {
    "mechanical.metric_thread_tensile_area": [r"tensile stress area", r"metric thread area"],
    "mechanical.bolt_preload": [r"bolt preload", r"proof load"],
    "mechanical.bolt_joint_load": [r"joint load", r"load sharing", r"joint separat"],
    "mechanical.weld_primary_shear": [r"weld.*(?:direct|primary).*shear", r"fillet weld shear"],
    "mechanical.helical_spring_rate": [r"spring rate", r"spring stiffness"],
    "mechanical.wahl_factor": [r"wahl factor"],
    "mechanical.spring_max_shear_stress": [r"spring.*shear stress", r"spring.*maximum stress"],
    "mechanical.bearing_l10_life": [r"bearing.*(?:l10|life)", r"\bl10\b"],
    "aero.force_decomposition": [r"(?:lift|drag).*(?:normal|axial)", r"force decomposition"],
    "aero.point_source_velocity": [r"(?:point )?(?:source|sink).*velocity"],
    "aero.doublet_velocity": [r"doublet.*velocity"],
    "aero.point_vortex_velocity": [r"(?:point )?vortex.*velocity"],
    "aero.cylinder_flow_velocity": [r"(?:cylinder|circular cylinder).*velocity"],
    "aero.pressure_coefficient": [r"pressure coefficient", r"\bcp\b"],
    "aero.kutta_joukowski_cylinder": [r"kutta.?joukowski", r"lifting cylinder", r"lift per.*span"],
    "qrm.normal_cdf": [r"normal.*(?:cdf|cumulative|probability)"],
    "qrm.binomial_probability": [r"binomial.*probability"],
    "qrm.poisson_probability": [r"poisson.*probability"],
    "qrm.acceptance_oc_binomial": [r"(?:oc|acceptance).*(?:binomial|large lot)"],
    "qrm.acceptance_oc_hypergeometric": [r"(?:oc|acceptance).*(?:hypergeometric|finite lot)"],
    "qrm.rectifying_sampling_performance": [r"\b(?:aoq|ati|asn)\b", r"rectifying sampling"],
    "qrm.average_run_length": [r"average run length", r"\barl\b"],
    "qrm.xbar_control_limits": [r"x.?bar.*control limit", r"mean chart.*limit"],
    "qrm.p_control_limits": [r"\bp.?chart.*limit", r"fraction nonconforming.*limit"],
    "qrm.exponential_reliability": [r"exponential.*reliability", r"constant failure rate"],
    "qrm.weibull_reliability": [r"weibull.*reliability", r"weibull.*failure"],
}

PARAM_ALIASES = {
    "dp_mm": ("major diameter", "dp", "m"),
    "pitch_mm": ("pitch", "p"),
    "area_mm2": ("stress area", "area", "at"),
    "proof_strength_mpa": ("proof strength", "sp"),
    "preload_n": ("preload", "fi"),
    "bolt_stiffness_n_per_mm": ("bolt stiffness", "kb"),
    "member_stiffness_n_per_mm": ("member stiffness", "km"),
    "external_load_n": ("external load", "load", "p"),
    "force_n": ("force", "load", "f"),
    "leg_size_mm": ("leg size", "h"),
    "weld_length_mm": ("weld length", "length", "l"),
    "shear_modulus_mpa": ("shear modulus", "g"),
    "wire_diameter_mm": ("wire diameter", "wire", "d"),
    "mean_diameter_mm": ("mean diameter", "mean d"),
    "active_coils": ("active coils", "na"),
    "spring_index": ("spring index", "c"),
    "dynamic_rating_n": ("dynamic rating", "c rating"),
    "equivalent_load_n": ("equivalent load", "p"),
    "speed_rpm": ("speed", "rpm", "n"),
    "normal_force_n": ("normal force", "n"),
    "axial_force_n": ("axial force", "a"),
    "angle_of_attack_rad": ("angle of attack", "alpha"),
    "x_m": ("x",),
    "y_m": ("y",),
    "strength_m2_per_s": ("source strength", "strength", "lambda"),
    "strength_m3_per_s": ("doublet strength", "strength", "kappa"),
    "circulation_m2_per_s": ("circulation", "gamma"),
    "freestream_speed_m_per_s": ("freestream speed", "freestream", "u infinity"),
    "cylinder_radius_m": ("cylinder radius", "radius", "r"),
    "radial_position_m": ("radial position", "position radius", "r"),
    "theta_rad": ("theta", "angle"),
    "local_speed_m_per_s": ("local speed", "velocity", "v"),
    "density_kg_per_m3": ("density", "rho"),
    "standard_deviation": ("standard deviation", "sigma"),
    "mean": ("mean", "mu"),
    "x": ("x", "threshold"),
    "trials": ("trials", "n"),
    "successes": ("successes", "x"),
    "event_probability": ("event probability", "probability", "p"),
    "event_count": ("event count", "count", "x"),
    "mean_count": ("mean count", "lambda"),
    "sample_size": ("sample size", "sample", "n"),
    "acceptance_number": ("acceptance number", "c"),
    "fraction_defective": ("fraction defective", "defective fraction", "p"),
    "lot_size": ("lot size", "n lot"),
    "defective_units": ("defective units", "defectives", "d"),
    "acceptance_probability": ("acceptance probability", "pa"),
    "signal_probability": ("signal probability", "pd"),
    "process_mean": ("process mean", "mean", "mu"),
    "process_standard_deviation": ("process standard deviation", "sigma"),
    "sigma_multiplier": ("sigma multiplier", "z", "k"),
    "average_fraction_nonconforming": ("average fraction nonconforming", "p bar"),
    "time": ("time", "t"),
    "failure_rate_per_time": ("failure rate", "lambda"),
    "scale_time": ("scale", "eta"),
    "shape": ("shape", "beta"),
}

CONCEPTUAL_RE = re.compile(
    r"\b(?:formula|equation|derive|derivation|explain|why|meaning|definition|"
    r"how (?:do|can) i (?:calculate|compute))\b",
    re.IGNORECASE,
)
NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?(?:e[+-]?\d+)?", re.IGNORECASE)


def _select_formula(query: str, course_id: str) -> Optional[str]:
    for formula_id, patterns in FORMULA_PATTERNS.items():
        definition = get_formula(formula_id)
        if definition.course_id != course_id:
            continue
        if any(re.search(pattern, query, re.IGNORECASE) for pattern in patterns):
            return formula_id
    return None


def classify_intent(state: AgentState) -> AgentState:
    course_id = state.get("course_id", "mechanical-design")
    explicit_formula = bool(state.get("formula_id"))
    formula_id = state.get("formula_id") or _select_formula(state["query"], course_id)
    if formula_id and (explicit_formula or not CONCEPTUAL_RE.search(state["query"])):
        return {
            **state,
            "intent": "calculate",
            "formula_id": formula_id,
            "calc_type": formula_id,
        }
    return {**state, "intent": "retrieve", "formula_id": None, "calc_type": None}


def _extract_labeled_number(query: str, aliases: tuple[str, ...]) -> Optional[float]:
    for alias in sorted(aliases, key=len, reverse=True):
        escaped = re.escape(alias).replace(r"\ ", r"\s*")
        match = re.search(
            rf"(?<![A-Za-z]){escaped}\s*(?:=|:|of|is)?\s*({NUMBER_RE.pattern})",
            query,
            re.IGNORECASE,
        )
        if match:
            return float(match.group(1))
        reverse = re.search(
            rf"({NUMBER_RE.pattern})\s*(?:[A-Za-z0-9/^²³.-]+\s*)?"
            rf"(?<![A-Za-z]){escaped}(?![A-Za-z])",
            query,
            re.IGNORECASE,
        )
        if reverse:
            return float(reverse.group(1))
    return None


def extract_params(state: AgentState) -> AgentState:
    definition = get_formula(state["formula_id"])
    params = dict(state.get("params") or {})
    units = dict(state.get("units") or {})
    query = state["query"]

    for variable in definition.variables:
        if variable.name in params:
            continue
        if variable.value_type in {"number", "integer"}:
            value = _extract_labeled_number(
                query, PARAM_ALIASES.get(variable.name, (variable.name,))
            )
            if value is not None:
                params[variable.name] = int(value) if variable.value_type == "integer" else value
        elif variable.value_type == "boolean" and variable.name == "reused":
            params[variable.name] = not bool(
                re.search(r"\b(?:permanent|new)\b", query, re.IGNORECASE)
            )
        elif variable.value_type == "string" and variable.name == "bearing_type":
            params[variable.name] = (
                "roller" if re.search(r"\broller\b", query, re.IGNORECASE) else "ball"
            )

    supplied_required = [
        variable
        for variable in definition.variables
        if variable.required and variable.name in params
    ]
    if not supplied_required:
        numbers = [float(value) for value in NUMBER_RE.findall(query)]
        numeric_required = [
            variable
            for variable in definition.variables
            if variable.required and variable.value_type in {"number", "integer"}
        ]
        for variable, value in zip(numeric_required, numbers):
            params[variable.name] = int(value) if variable.value_type == "integer" else value

    for variable in definition.variables:
        if variable.name in params and variable.unit != "1":
            units.setdefault(variable.name, variable.unit)

    missing = [
        variable.name
        for variable in definition.variables
        if variable.required and variable.name not in params
    ]
    return {
        **state,
        "params": params,
        "units": units,
        "missing_params": missing,
        "intent": "clarify" if missing else "calculate",
    }


def clarify_node(state: AgentState) -> AgentState:
    definition = get_formula(state["formula_id"])
    variables = {variable.name: variable for variable in definition.variables}
    needed = [
        f"{variables[name].description} ({variables[name].unit})"
        for name in state["missing_params"]
    ]
    text = "I can use the reviewed course formula, but I still need: " + "; ".join(needed) + "."
    return {
        **state,
        "answer": text,
        "course_answer": text,
        "general_answer": None,
        "citations": [],
        "warnings": [],
        "assumptions": list(definition.assumptions),
    }


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:,.6g}"
    return str(value)


def calculate_node(state: AgentState) -> AgentState:
    definition = get_formula(state["formula_id"])
    try:
        result = execute(state["formula_id"], state["params"], state["units"])
    except FormulaError as exc:
        text = f"Invalid calculation input: {exc}"
        return {
            **state,
            "answer": text,
            "course_answer": text,
            "general_answer": None,
            "citations": [],
            "warnings": [str(exc)],
            "assumptions": list(definition.assumptions),
        }

    details = ", ".join(
        f"{name.replace('_', ' ')} = {_format_value(value)}"
        + (f" {result.units.get(name)}" if result.units.get(name, "1") != "1" else "")
        for name, value in result.values.items()
    )
    course_answer = f"Using {definition.title}: {details}."
    first_page = result.source.pages[0]
    page_match = re.search(r"\d+", first_page)
    citation = {
        "id": result.formula_id,
        "source": result.source.document,
        "section": result.source.section,
        "page": int(page_match.group()) if page_match else 1,
        "pages": list(result.source.pages),
        "course_id": result.course_id,
        "formula_id": result.formula_id,
    }
    return {
        **state,
        "answer": course_answer,
        "course_answer": course_answer,
        "general_answer": None,
        "citations": [citation],
        "warnings": list(result.warnings),
        "assumptions": list(result.assumptions),
    }


class RetrievalEngine:
    """Lazily loads all approved courses into a shared, filtered index."""

    def __init__(
        self,
        courses_dir: Optional[Path] = None,
        corpus_dir: Optional[Path] = None,
        persist_dir: Optional[Path] = None,
    ):
        root = Path(__file__).resolve().parent.parent
        self.courses_dir = Path(courses_dir or root / "data" / "courses")
        self.corpus_dir = Path(corpus_dir or root / "data" / "corpus")
        self.persist_dir = Path(persist_dir or root / "data" / "chroma")
        self._dense = None
        self._sparse = None
        self._reranker = None

    def _initialize(self) -> None:
        from src.chunker import load_course_corpus
        from src.dense_retriever import ensure_dense_index
        from src.reranker import LexicalCrossEncoder
        from src.sparse_retriever import SparseIndex

        chunks = load_course_corpus(
            str(self.courses_dir), str(self.corpus_dir)
        )
        if not chunks:
            raise RuntimeError("No reviewed course content is available")
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._dense = ensure_dense_index(chunks, str(self.persist_dir))
        self._sparse = SparseIndex(chunks)
        self._reranker = LexicalCrossEncoder()

    def search(self, query: str, course_id: str, top_k: int = 4) -> list[dict]:
        from src.hybrid_retriever import hybrid_search

        if self._dense is None:
            self._initialize()
        candidates = hybrid_search(
            self._dense,
            self._sparse,
            query,
            k=12,
            fetch_k=20,
            course_id=course_id,
        )
        return self._reranker.rerank(query, candidates, top_k=top_k)


class MechanicalDesignAgent:
    """Compatibility name retained for existing callers; now supports all courses."""

    def __init__(
        self,
        retriever: Optional[RetrievalEngine] = None,
        llm: Optional[OllamaClient] = None,
    ):
        self.retriever = retriever or RetrievalEngine()
        self.llm = llm or OllamaClient()
        self.graph = self._build_graph()

    def _retrieve_node(self, state: AgentState) -> AgentState:
        try:
            chunks = self.retriever.search(
                state["query"], state["course_id"], top_k=4
            )
        except TypeError:
            chunks = self.retriever.search(state["query"], top_k=4)
        if not chunks:
            text = "No reviewed material is indexed for this course yet."
            return {
                **state,
                "answer": text,
                "course_answer": text,
                "general_answer": None,
                "retrieved_chunks": [],
                "citations": [],
                "warnings": [],
                "assumptions": [],
            }

        if hasattr(self.llm, "generate_structured"):
            generated = self.llm.generate_structured(
                state["query"], chunks, state.get("include_general", True)
            )
        else:
            generated = GroundedAnswer(
                course_answer=self.llm.generate(state["query"], chunks),
                general_answer=None,
                citation_ids=[item["id"] for item in chunks],
            )
        selected_ids = set(generated.citation_ids)
        cited_chunks = [item for item in chunks if item["id"] in selected_ids]
        citations = [
            {
                "id": item["id"],
                "source": item["metadata"]["source"],
                "section": item["metadata"].get("section", ""),
                "page": item["metadata"]["page"],
                "slide": item["metadata"].get("slide"),
                "course_id": item["metadata"].get("course_id"),
                "document_id": item["metadata"].get("document_id"),
            }
            for item in cited_chunks
        ]
        answer = generated.course_answer
        if generated.general_answer:
            answer += f"\n\nGeneral clarification: {generated.general_answer}"
        return {
            **state,
            "retrieved_chunks": chunks,
            "citations": citations,
            "answer": answer,
            "course_answer": generated.course_answer,
            "general_answer": generated.general_answer,
            "warnings": [],
            "assumptions": [],
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
        for node in ("calculate", "retrieve", "clarify"):
            graph.add_edge(node, END)
        return graph.compile()

    def ask(
        self,
        query: str,
        params: Optional[dict] = None,
        course_id: str = "mechanical-design",
        include_general: bool = True,
        formula_id: Optional[str] = None,
        units: Optional[dict] = None,
    ) -> AgentState:
        if not query.strip():
            raise ValueError("Query cannot be empty")
        if course_id not in SUPPORTED_COURSES:
            raise ValueError(
                f"Unknown course '{course_id}'. Choose from: "
                + ", ".join(SUPPORTED_COURSES)
            )
        if formula_id and get_formula(formula_id).course_id != course_id:
            raise ValueError("Selected formula does not belong to the selected course")
        return self.graph.invoke(
            {
                "query": query.strip(),
                "course_id": course_id,
                "include_general": include_general,
                "formula_id": formula_id,
                "params": params or {},
                "units": units or {},
            }
        )

    @staticmethod
    def formulas(course_id: Optional[str] = None):
        return list_formulas(course_id=course_id)


CourseRagAgent = MechanicalDesignAgent


def build_graph():
    return MechanicalDesignAgent().graph
