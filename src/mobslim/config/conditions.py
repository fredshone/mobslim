"""Person filters: conditions on person attributes.

A condition selects persons by their attributes. It is written either in a
structured form::

    {age: {gt: 50}}                     # ops: eq ne lt le gt ge in not_in
    {age: {ge: 18, lt: 65}}             # several ops: all must hold
    {subpopulation: freight}            # a plain value: eq
    {all: [...]}  {any: [...]}  {not: {...}}

or as a Python-style expression string::

    age > 50
    has_license == False and age >= 18
    18 <= age < 65
    subpopulation in ["freight", "taxi"]
    not (income < 1000 or subpopulation == "freight")

Strings are parsed with :mod:`ast` and only comparisons of an attribute
name with literals, ``and``, ``or``, ``not`` and parentheses are accepted;
nothing is evaluated as code. Attribute names in strings must be Python
identifiers; use the structured form for others.

Both forms compile to the same tree, evaluated on whole
:class:`~mobslim.attributes.PersonAttributes` columns into a boolean mask.
A person **missing** the attribute matches no comparison (``ne`` and
``not_in`` included). ``not`` negates the whole condition, so
``not (age > 50)`` does select persons without an age.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any, Union

import numpy as np

from mobslim.attributes import PersonAttributes

OPS = ("eq", "ne", "lt", "le", "gt", "ge", "in", "not_in")
_ORDER_OPS = ("lt", "le", "gt", "ge")
_SYMBOL = {"eq": "==", "ne": "!=", "lt": "<", "le": "<=", "gt": ">", "ge": ">=",
           "in": "in", "not_in": "not in"}
_AST_OPS = {ast.Eq: "eq", ast.NotEq: "ne", ast.Lt: "lt", ast.LtE: "le", ast.Gt: "gt",
            ast.GtE: "ge", ast.In: "in", ast.NotIn: "not_in"}
_FLIP = {"lt": "gt", "le": "ge", "gt": "lt", "ge": "le", "eq": "eq", "ne": "ne"}


class ConditionError(ValueError):
    """A condition that cannot be parsed or does not fit the attributes."""


@dataclass(frozen=True)
class Compare:
    attribute: str
    op: str
    value: Any  # a scalar, or a tuple for in / not_in

    def __str__(self) -> str:
        v = list(self.value) if isinstance(self.value, tuple) else self.value
        return f"{self.attribute} {_SYMBOL[self.op]} {v!r}"


@dataclass(frozen=True)
class All:
    parts: tuple

    def __str__(self) -> str:
        return " and ".join(f"({p})" if isinstance(p, AnyOf) else str(p) for p in self.parts)


@dataclass(frozen=True)
class AnyOf:
    parts: tuple

    def __str__(self) -> str:
        return " or ".join(str(p) for p in self.parts)


@dataclass(frozen=True)
class Not:
    part: Any

    def __str__(self) -> str:
        return f"not ({self.part})"


Condition = Union[Compare, All, AnyOf, Not]


# ------------------------------------------------------------------ parsing

def parse_condition(raw) -> Condition:
    """A condition from its structured (dict) or string form."""
    if isinstance(raw, str):
        return _parse_string(raw)
    if isinstance(raw, dict):
        return _parse_dict(raw)
    if isinstance(raw, bool):
        raise ConditionError(f"expected a condition, got {raw!r}")
    raise ConditionError(f"expected a condition (a mapping or a string), got {raw!r}")


def _scalar(v, where: str):
    if isinstance(v, (list, tuple)):
        raise ConditionError(f"{where}: a list is only allowed with 'in' / 'not_in'")
    if isinstance(v, dict):
        raise ConditionError(f"{where}: expected a value, got a mapping")
    if v is None:
        raise ConditionError(f"{where}: cannot compare with null (missing values never match)")
    return v


def _compare(attribute: str, op: str, value) -> Compare:
    where = f"{attribute} {_SYMBOL.get(op, op)}"
    if op in ("in", "not_in"):
        if not isinstance(value, (list, tuple)) or not value:
            raise ConditionError(f"{where}: expected a non-empty list")
        return Compare(attribute, op, tuple(_scalar(v, where) for v in value))
    value = _scalar(value, where)
    if op in _ORDER_OPS and (isinstance(value, bool) or not isinstance(value, (int, float))):
        raise ConditionError(f"{where}: expected a number, got {value!r}")
    return Compare(attribute, op, value)


def _parse_dict(raw: dict) -> Condition:
    if len(raw) != 1:
        if not raw:
            raise ConditionError("empty condition")
        return All(tuple(_parse_dict({k: v}) for k, v in raw.items()))
    (key, value), = raw.items()
    if key == "all" or key == "any":
        if not isinstance(value, list) or not value:
            raise ConditionError(f"'{key}' needs a non-empty list of conditions")
        parts = tuple(parse_condition(v) for v in value)
        return All(parts) if key == "all" else AnyOf(parts)
    if key == "not":
        return Not(parse_condition(value))
    if not isinstance(key, str) or not key:
        raise ConditionError(f"expected an attribute name, got {key!r}")
    if isinstance(value, dict):
        if not value:
            raise ConditionError(f"{key}: empty comparison")
        parts = []
        for op, v in value.items():
            if op not in OPS:
                raise ConditionError(f"{key}: unknown operator {op!r} (use one of {', '.join(OPS)})")
            parts.append(_compare(key, op, v))
        return parts[0] if len(parts) == 1 else All(tuple(parts))
    return _compare(key, "eq", value)


def _parse_string(text: str) -> Condition:
    try:
        tree = ast.parse(text.strip(), mode="eval")
    except SyntaxError as e:
        col = (e.offset or 1) - 1
        raise ConditionError(f"cannot parse {text!r} at column {col + 1}: {e.msg}") from None
    return _from_ast(tree.body, text)


def _literal(node, text: str):
    try:
        return ast.literal_eval(node)
    except ValueError:
        src = ast.get_source_segment(text, node) or "?"
        raise ConditionError(f"{text!r}: {src!r} is not a literal value "
                             "(compare an attribute with a number, string, True/False or a list)") from None


def _from_ast(node, text: str) -> Condition:
    if isinstance(node, ast.BoolOp):
        parts = tuple(_from_ast(v, text) for v in node.values)
        return All(parts) if isinstance(node.op, ast.And) else AnyOf(parts)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return Not(_from_ast(node.operand, text))
    if isinstance(node, ast.Compare):
        operands = [node.left, *node.comparators]
        parts = []
        for left, op, right in zip(operands, node.ops, operands[1:]):
            name = _AST_OPS.get(type(op))
            if name is None:
                raise ConditionError(f"{text!r}: operator {type(op).__name__} is not supported")
            if isinstance(left, ast.Name) and not isinstance(right, ast.Name):
                parts.append(_compare(left.id, name, _literal(right, text)))
            elif isinstance(right, ast.Name) and not isinstance(left, ast.Name) and name in _FLIP:
                parts.append(_compare(right.id, _FLIP[name], _literal(left, text)))
            else:
                raise ConditionError(f"{text!r}: each comparison needs one attribute name "
                                     "and one literal value")
        return parts[0] if len(parts) == 1 else All(tuple(parts))
    if isinstance(node, ast.Name):
        raise ConditionError(f"{text!r}: compare {node.id!r} with a value, e.g. {node.id} == True")
    src = ast.get_source_segment(text, node) or text
    raise ConditionError(f"{text!r}: {src!r} is not a comparison")


# --------------------------------------------------------------- inspection

def attributes_of(cond: Condition) -> list[str]:
    """Attribute names a condition reads, in order of first use."""
    out: dict[str, None] = {}

    def walk(c):
        if isinstance(c, Compare):
            out[c.attribute] = None
        elif isinstance(c, Not):
            walk(c.part)
        else:
            for p in c.parts:
                walk(p)

    walk(cond)
    return list(out)


def _comparisons(cond: Condition):
    if isinstance(cond, Compare):
        yield cond
    elif isinstance(cond, Not):
        yield from _comparisons(cond.part)
    else:
        for p in cond.parts:
            yield from _comparisons(p)


def column_kind(attributes: PersonAttributes, name: str) -> str:
    """``"number"``, ``"bool"``, ``"str"``, ``"mixed"`` or ``"empty"`` (no values)."""
    col = attributes[name]
    if col.dtype == bool:
        return "bool"
    if col.dtype.kind in "iuf":
        return "empty" if attributes.missing(name).all() else "number"
    present = [v for v, m in zip(col.tolist(), attributes.missing(name).tolist()) if not m]
    if not present:
        return "empty"
    kinds = {_value_kind(v) for v in present}
    return kinds.pop() if len(kinds) == 1 else "mixed"


def _value_kind(v) -> str:
    if isinstance(v, (bool, np.bool_)):
        return "bool"
    if isinstance(v, (int, float, np.integer, np.floating)):
        return "number"
    return "str"


def check_condition(cond: Condition, attributes: PersonAttributes) -> list[str]:
    """Problems with ``cond`` on these attributes: attributes no person has,
    and comparisons of mismatched types. Empty if it fits."""
    problems = []
    known = attributes.names
    for c in _comparisons(cond):
        if c.attribute not in attributes:
            listed = ", ".join(sorted(known)) if known else "none"
            problems.append(f"no person attribute {c.attribute!r} (attributes are: {listed})")
            continue
        kind = column_kind(attributes, c.attribute)
        if kind in ("empty", "mixed"):
            continue
        if c.op in _ORDER_OPS and kind != "number":
            problems.append(f"{c}: '{c.attribute}' holds {kind} values, so it cannot be "
                            f"compared with {_SYMBOL[c.op]}")
            continue
        values = c.value if isinstance(c.value, tuple) else (c.value,)
        for v in values:
            if _value_kind(v) != kind:
                problems.append(f"{c}: '{c.attribute}' holds {kind} values, not "
                                f"{_value_kind(v)} values like {v!r}")
                break
    return problems


# --------------------------------------------------------------- evaluation

def _compare_mask(c: Compare, attributes: PersonAttributes) -> np.ndarray:
    n = attributes.n_persons
    col = attributes.get(c.attribute)
    if col is None:
        return np.zeros(n, dtype=bool)
    present = ~attributes.missing(c.attribute)
    values = c.value if isinstance(c.value, tuple) else (c.value,)
    if c.op in _ORDER_OPS:
        if col.dtype.kind not in "iuf":
            return np.zeros(n, dtype=bool)
        x = col.astype(float)
        with np.errstate(invalid="ignore"):
            hit = {"lt": x < c.value, "le": x <= c.value, "gt": x > c.value, "ge": x >= c.value}[c.op]
        return hit & present
    hit = np.zeros(n, dtype=bool)
    for v in values:
        if col.dtype == object:
            hit |= np.array([(_value_kind(a) == _value_kind(v)) and a == v for a in col.tolist()],
                            dtype=bool)
        elif _value_kind(v) == ("bool" if col.dtype == bool else "number"):
            hit |= col == v
    if c.op in ("ne", "not_in"):
        hit = ~hit
    return hit & present


def evaluate(cond: Condition, attributes: PersonAttributes) -> np.ndarray:
    """Boolean mask: the persons that match ``cond``."""
    if isinstance(cond, Compare):
        return _compare_mask(cond, attributes)
    if isinstance(cond, Not):
        return ~evaluate(cond.part, attributes)
    masks = [evaluate(p, attributes) for p in cond.parts]
    out = masks[0].copy()
    for m in masks[1:]:
        if isinstance(cond, All):
            out &= m
        else:
            out |= m
    return out
