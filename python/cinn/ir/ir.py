from cinn import ir
from .ir_context import ForContext


def sequential(min, extent=None):
    if extent is None:
        extent = min
        min = ir.Expr(0)
    if not isinstance(min, ir.Expr):
        min = ir.Expr(min)
    if not isinstance(extent, ir.Expr):
        extent = ir.Expr(extent)
    return ForContext(min, extent)
