"""Verified multi-course engineering calculation API.

Use ``execute`` for validated canonical-unit execution, or inspect ``registry``
for complete formula metadata.
"""
from typing import Any, Mapping, Optional, Tuple

from . import aerodynamics, mechanical, qrm
from .registry import (
    CalculationResult,
    FormulaDefinition,
    FormulaError,
    FormulaNotFoundError,
    FormulaRegistry,
    FormulaReviewError,
    FormulaValidationError,
    OutputDefinition,
    ReviewState,
    SourceReference,
    VariableDefinition,
)

registry = FormulaRegistry()
mechanical.register_formulas(registry)
aerodynamics.register_formulas(registry)
qrm.register_formulas(registry)


def execute(
    formula_id: str,
    inputs: Mapping[str, Any],
    units: Optional[Mapping[str, str]] = None,
) -> CalculationResult:
    """Execute a reviewed formula after validating values and canonical units."""
    return registry.execute(formula_id, inputs, units)


def get_formula(formula_id: str) -> FormulaDefinition:
    """Return metadata for one formula without executing it."""
    return registry.get(formula_id)


def list_formulas(
    course_id: Optional[str] = None,
    review_state: Optional[ReviewState] = None,
) -> Tuple[FormulaDefinition, ...]:
    """List formulas, optionally filtered by course and review state."""
    return registry.list(course_id=course_id, review_state=review_state)


__all__ = [
    "CalculationResult",
    "FormulaDefinition",
    "FormulaError",
    "FormulaNotFoundError",
    "FormulaRegistry",
    "FormulaReviewError",
    "FormulaValidationError",
    "OutputDefinition",
    "ReviewState",
    "SourceReference",
    "VariableDefinition",
    "execute",
    "get_formula",
    "list_formulas",
    "registry",
]
