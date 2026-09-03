"""Typed, provenance-aware formula registration and execution."""
from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple


class FormulaError(ValueError):
    """Base error raised by the calculation system."""


class FormulaNotFoundError(FormulaError):
    """Raised when a formula ID is unknown."""


class FormulaReviewError(FormulaError):
    """Raised when execution is attempted for an unreviewed formula."""


class FormulaValidationError(FormulaError):
    """Raised when formula inputs or units are invalid."""


class ReviewState(str, Enum):
    DRAFT = "draft"
    REVIEWED = "reviewed"
    REJECTED = "rejected"


@dataclass(frozen=True)
class SourceReference:
    document: str
    pages: Tuple[str, ...]
    section: str = ""


@dataclass(frozen=True)
class VariableDefinition:
    name: str
    description: str
    unit: str = "1"
    value_type: str = "number"
    required: bool = True
    default: Any = None
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    minimum_exclusive: bool = False
    maximum_exclusive: bool = False
    choices: Tuple[Any, ...] = ()


@dataclass(frozen=True)
class OutputDefinition:
    name: str
    description: str
    unit: str = "1"


@dataclass(frozen=True)
class FormulaDefinition:
    formula_id: str
    course_id: str
    title: str
    equation: str
    source: SourceReference
    variables: Tuple[VariableDefinition, ...]
    outputs: Tuple[OutputDefinition, ...]
    assumptions: Tuple[str, ...]
    review_state: ReviewState
    executor: Callable[..., Mapping[str, Any]] = field(repr=False, compare=False)
    discrepancy_warning: Optional[str] = None


@dataclass(frozen=True)
class CalculationResult:
    formula_id: str
    course_id: str
    values: Mapping[str, Any]
    units: Mapping[str, str]
    assumptions: Tuple[str, ...]
    warnings: Tuple[str, ...]
    source: SourceReference


class FormulaRegistry:
    """Registry that validates canonical-unit inputs before execution."""

    def __init__(self) -> None:
        self._formulas: Dict[str, FormulaDefinition] = {}

    def register(self, definition: FormulaDefinition) -> None:
        if not definition.formula_id or not definition.course_id:
            raise FormulaValidationError("formula_id and course_id must be non-empty")
        if definition.formula_id in self._formulas:
            raise FormulaValidationError(
                "Duplicate formula ID: {}".format(definition.formula_id)
            )
        variable_names = [variable.name for variable in definition.variables]
        output_names = [output.name for output in definition.outputs]
        if len(variable_names) != len(set(variable_names)):
            raise FormulaValidationError("Variable names must be unique")
        if len(output_names) != len(set(output_names)):
            raise FormulaValidationError("Output names must be unique")
        if not definition.source.document or not definition.source.pages:
            raise FormulaValidationError("Formula source document and pages are required")
        self._formulas[definition.formula_id] = definition

    def get(self, formula_id: str) -> FormulaDefinition:
        try:
            return self._formulas[formula_id]
        except KeyError:
            raise FormulaNotFoundError("Unknown formula ID: {}".format(formula_id))

    def list(
        self,
        course_id: Optional[str] = None,
        review_state: Optional[ReviewState] = None,
    ) -> Tuple[FormulaDefinition, ...]:
        formulas = self._formulas.values()
        if course_id is not None:
            formulas = (f for f in formulas if f.course_id == course_id)
        if review_state is not None:
            formulas = (f for f in formulas if f.review_state == review_state)
        return tuple(sorted(formulas, key=lambda formula: formula.formula_id))

    def execute(
        self,
        formula_id: str,
        inputs: Mapping[str, Any],
        units: Optional[Mapping[str, str]] = None,
    ) -> CalculationResult:
        definition = self.get(formula_id)
        if definition.review_state != ReviewState.REVIEWED:
            raise FormulaReviewError(
                "Formula '{}' is {}; only reviewed formulas may execute".format(
                    formula_id, definition.review_state.value
                )
            )
        values = self._validate_inputs(definition, inputs, units or {})
        try:
            output_values = dict(definition.executor(**values))
        except FormulaError:
            raise
        except (ArithmeticError, OverflowError) as exc:
            raise FormulaValidationError(
                "Formula execution failed: {}".format(exc)
            ) from exc

        output_units = {output.name: output.unit for output in definition.outputs}
        expected = set(output_units)
        actual = set(output_values)
        if actual != expected:
            raise FormulaValidationError(
                "Executor output mismatch; expected {}, got {}".format(
                    sorted(expected), sorted(actual)
                )
            )
        for name, value in output_values.items():
            if isinstance(value, float) and not math.isfinite(value):
                raise FormulaValidationError("Output '{}' is not finite".format(name))
        warnings = (
            (definition.discrepancy_warning,)
            if definition.discrepancy_warning
            else ()
        )
        return CalculationResult(
            formula_id=formula_id,
            course_id=definition.course_id,
            values=output_values,
            units=output_units,
            assumptions=definition.assumptions,
            warnings=warnings,
            source=definition.source,
        )

    @staticmethod
    def _validate_inputs(
        definition: FormulaDefinition,
        supplied: Mapping[str, Any],
        units: Mapping[str, str],
    ) -> Dict[str, Any]:
        variables = {variable.name: variable for variable in definition.variables}
        unknown = set(supplied) - set(variables)
        unknown_units = set(units) - set(variables)
        if unknown or unknown_units:
            names = sorted(unknown | unknown_units)
            raise FormulaValidationError("Unknown inputs: {}".format(", ".join(names)))

        values: Dict[str, Any] = {}
        for name, variable in variables.items():
            was_supplied = name in supplied
            if name in supplied:
                value = supplied[name]
            elif variable.required:
                raise FormulaValidationError("Missing required input: {}".format(name))
            else:
                value = variable.default

            FormulaRegistry._validate_value(variable, value)
            if variable.unit != "1" and was_supplied:
                if name not in units:
                    raise FormulaValidationError(
                        "Unit required for '{}'; expected '{}'".format(
                            name, variable.unit
                        )
                    )
                if units[name] != variable.unit:
                    raise FormulaValidationError(
                        "Invalid unit for '{}': expected '{}', got '{}'".format(
                            name, variable.unit, units[name]
                        )
                    )
            elif name in units and units[name] != variable.unit:
                raise FormulaValidationError(
                    "Invalid unit for '{}': expected '{}', got '{}'".format(
                        name, variable.unit, units[name]
                    )
                )
            values[name] = value
        return values

    @staticmethod
    def _validate_value(variable: VariableDefinition, value: Any) -> None:
        if variable.value_type == "number":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise FormulaValidationError(
                    "Input '{}' must be a number".format(variable.name)
                )
            if not math.isfinite(float(value)):
                raise FormulaValidationError(
                    "Input '{}' must be finite".format(variable.name)
                )
            numeric = float(value)
            if variable.minimum is not None:
                invalid = (
                    numeric <= variable.minimum
                    if variable.minimum_exclusive
                    else numeric < variable.minimum
                )
                if invalid:
                    comparator = ">" if variable.minimum_exclusive else ">="
                    raise FormulaValidationError(
                        "Input '{}' must be {} {}".format(
                            variable.name, comparator, variable.minimum
                        )
                    )
            if variable.maximum is not None:
                invalid = (
                    numeric >= variable.maximum
                    if variable.maximum_exclusive
                    else numeric > variable.maximum
                )
                if invalid:
                    comparator = "<" if variable.maximum_exclusive else "<="
                    raise FormulaValidationError(
                        "Input '{}' must be {} {}".format(
                            variable.name, comparator, variable.maximum
                        )
                    )
        elif variable.value_type == "integer":
            if isinstance(value, bool) or not isinstance(value, int):
                raise FormulaValidationError(
                    "Input '{}' must be an integer".format(variable.name)
                )
            FormulaRegistry._validate_value(
                VariableDefinition(
                    name=variable.name,
                    description=variable.description,
                    minimum=variable.minimum,
                    maximum=variable.maximum,
                    minimum_exclusive=variable.minimum_exclusive,
                    maximum_exclusive=variable.maximum_exclusive,
                ),
                value,
            )
        elif variable.value_type == "boolean":
            if not isinstance(value, bool):
                raise FormulaValidationError(
                    "Input '{}' must be boolean".format(variable.name)
                )
        elif variable.value_type == "string":
            if not isinstance(value, str):
                raise FormulaValidationError(
                    "Input '{}' must be a string".format(variable.name)
                )
        else:
            raise FormulaValidationError(
                "Unsupported value_type '{}'".format(variable.value_type)
            )
        if variable.choices and value not in variable.choices:
            raise FormulaValidationError(
                "Input '{}' must be one of {}".format(
                    variable.name, variable.choices
                )
            )
