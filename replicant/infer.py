import ast
import inspect
import textwrap
from typing import Any, Callable

from replicant.bounds import CategoricalBound, InputBound, NumericBound

EPS = 1


class InputSpaceInferer(ast.NodeVisitor):
    def __init__(self):
        self.param_conditions: dict[str, list[tuple[str, Any]]] = {}

    def visit_If(self, node: ast.If):
        self._extract_conditions(node.test)
        if node.orelse:
            negated = ast.UnaryOp(op=ast.Not(), operand=node.test)
            self._extract_conditions(negated)
        self.generic_visit(node)

    def _extract_conditions(self, test: ast.AST):
        if isinstance(test, ast.Compare):
            self._handle_compare(test)
        elif isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
            self._handle_negated_expression(test.operand)
        elif isinstance(test, ast.BoolOp):
            for value in test.values:
                self._extract_conditions(value)
        elif isinstance(test, ast.Name):
            self.param_conditions.setdefault(test.id, []).append(("Eq", True))

    def _handle_compare(self, node: ast.Compare):
        if isinstance(node.left, ast.Name):
            var_name = node.left.id
            for op, comparator in zip(node.ops, node.comparators):
                value = getattr(comparator, "value", None)
                if value is not None:
                    op_type = type(op).__name__
                    self.param_conditions.setdefault(var_name, []).append((op_type, value))

    def _handle_negated_expression(self, node: ast.AST):
        if isinstance(node, ast.Compare):
            self._handle_negated_compare(node)
        elif isinstance(node, ast.BoolOp):
            flipped_op = ast.Or() if isinstance(node.op, ast.And) else ast.And()
            flipped_values = [ast.UnaryOp(op=ast.Not(), operand=v) for v in node.values]
            flipped_boolop = ast.BoolOp(op=flipped_op, values=flipped_values)
            self._extract_conditions(flipped_boolop)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            self._extract_conditions(node.operand)

    def _handle_negated_compare(self, node: ast.Compare):
        if not isinstance(node.left, ast.Name):
            return
        var_name = node.left.id
        for op, comparator in zip(node.ops, node.comparators):
            value = getattr(comparator, "value", None)
            if value is not None:
                op_type = self._negate_op(type(op).__name__)
                self.param_conditions.setdefault(var_name, []).append((op_type, value))

    def _negate_op(self, op_type: str) -> str:
        return {
            "Gt": "LtE",
            "GtE": "Lt",
            "Lt": "GtE",
            "LtE": "Gt",
            "Eq": "NotEq",
            "NotEq": "Eq",
        }.get(op_type, op_type)

    def infer_bounds(self) -> dict[str, InputBound]:
        inferred = {}
        for var, conditions in self.param_conditions.items():
            if all(val in [True, False] for _, val in conditions):
                inferred[var] = CategoricalBound([True, False])
                continue
            if all(op in ("Eq", "NotEq") for op, _ in conditions):
                values = list({val for _, val in conditions})
                inferred[var] = CategoricalBound(values)
                continue

            lower, upper = None, None
            for op, val in conditions:
                if op == "Gt":
                    lower = min(lower, val) if lower is not None else val
                elif op == "GtE":
                    lower = max(lower, val) if lower is not None else val
                elif op == "Lt":
                    upper = max(upper, val) if upper is not None else val
                elif op == "LtE":
                    upper = min(upper, val) if upper is not None else val
                else:
                    raise RuntimeError("unrecognized op: {op!r}")
            inferred[var] = NumericBound(lower - EPS, upper + EPS)
        return inferred


def infer_input_space_from_function(fn: Callable[..., Any]) -> dict[str, InputBound]:
    source = textwrap.dedent(inspect.getsource(fn))
    tree = ast.parse(source)
    inferer = InputSpaceInferer()
    inferer.visit(tree)
    return inferer.infer_bounds()
