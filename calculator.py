"""Restricted arithmetic parser, extending the course calculator conventions."""
import ast
import math
import operator

FUNCTIONS = dict(exp=math.exp, log=math.log, sqrt=math.sqrt, abs=abs, min=min, max=max)


def calculate(expression, variables):
    expression = expression.strip().removeprefix('=').strip().replace('^', '**')
    if not expression or len(expression) > 600:
        raise ValueError('Enter an expression of 1 to 600 characters.')
    try:
        tree = ast.parse(expression, mode='eval')
        if sum(1 for _ in ast.walk(tree)) > 150:
            raise ValueError('Please shorten this expression.')
        def visit(n):
            if isinstance(n, ast.Constant) and type(n.value) in (int, float):
                x = float(n.value)
            elif isinstance(n, ast.Name):
                if n.id not in variables:
                    raise ValueError('Unknown symbol: ' + n.id)
                x = variables[n.id]
            elif isinstance(n, ast.UnaryOp) and type(n.op) in (ast.UAdd, ast.USub):
                x = visit(n.operand) * (-1 if isinstance(n.op, ast.USub) else 1)
            elif isinstance(n, ast.BinOp) and type(n.op) in (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow):
                a, b = visit(n.left), visit(n.right)
                if isinstance(n.op, ast.Pow) and abs(b) > 100:
                    raise ValueError('Keep power exponents between -100 and 100.')
                x = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
                     ast.Div: operator.truediv, ast.Pow: operator.pow}[type(n.op)](a, b)
            elif isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in FUNCTIONS and not n.keywords and 1 <= len(n.args) <= 8:
                x = FUNCTIONS[n.func.id](*[visit(a) for a in n.args])
            else:
                raise ValueError('Use arithmetic, listed symbols, and exp, log, sqrt, abs, min, max only.')
            if isinstance(x, complex) or not math.isfinite(x) or abs(x) > 1e12:
                raise ValueError('The result must be finite and within one trillion in magnitude.')
            return float(x)
        return visit(tree.body)
    except (SyntaxError, ZeroDivisionError, OverflowError, TypeError, RecursionError) as exc:
        raise ValueError('Check parentheses, function arguments, powers, and nonzero denominators.') from exc
