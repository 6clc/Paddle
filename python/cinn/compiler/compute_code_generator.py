# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ast
from typing import List, Union

from cinn import ir
from cinn.runtime.data_array import DataArray

from .utils import node_is_schedule


class ComputeCodeGenerator(ast.NodeVisitor):
    """
    Convert python ast to CINN Lower Level IR,
    containing only the semantics of the compute part
    """

    def __init__(self, function_name, inputs_signature):
        self.function_name = function_name
        self.inputs_signature = inputs_signature
        self.cinn_llir_func = None
        self.variables_table = {}
        self.schedule_block_iter_var2expr = {}

    def visit_FunctionDef(self, node) -> None:
        """
        Parse CINN Low Level IR FunctionDef.

        Args:
            node(ast.FunctionDef): The ast FunctionDef Node
        """
        arg_names = self.visit(node.args)

        assert len(node.args.defaults) == 0, "Not support default args"

        # 1. Construct args of function
        llir_args = []
        for i, arg_name in enumerate(arg_names):
            # Obj of Argument is ir::Buffer
            if hasattr(self.inputs_signature[i], "dtype"):
                llir_value = ir._Buffer_.make(
                    "_" + arg_name, self.inputs_signature[i].dtype
                )
                llir_args.append(
                    ir.Argument(llir_value, ir.Argument.IO.kUnknown)
                )
                tensor_shape = [
                    ir.Expr(dim) for dim in self.inputs_signature[i].shape
                ]

                # The computational logic of CINN is implemented through Tensor,
                # so ir::_Tensor_ is stored in local variables
                llir_value = ir._Tensor_.make(
                    arg_name,
                    self.inputs_signature[i].dtype,
                    tensor_shape,
                    tensor_shape,
                )
            # Obj of Argument is ir::Var
            else:
                llir_value = ir.Var(arg_name)
                llir_args.append(ir.Argument(llir_value))
                # The computational logic of CINN is implemented through Expr<Var>,
                # so ir::Expr is stored in local variables
                llir_value = ir.Expr(llir_value)
            self.set_value(arg_name, llir_value)

        # 2. Construct body of function
        body = self.visit_compound_statement(node.body)
        body = ir.ScheduleBlockRealize.make(
            [], ir.ScheduleBlock.make([], [], [], "root", body)
        )
        body = ir.Block.make([body])

        # 3. Construct LoweredFunc
        self.cinn_llir_func = ir.LoweredFunc.make(
            self.function_name, llir_args, body
        )

    def visit_compound_statement(self, stmts):
        cinn_stmts = []
        for stmt in stmts:
            cinn_stmt = self.visit(stmt)
            assert cinn_stmt is not None, f"Unsupport parse:\n{stmt}"
            # Some situations do not require conversion to CINN IR
            if str(cinn_stmt) in ["no compute"]:
                continue
            cinn_stmts.append(cinn_stmt)
        return ir.Block.make(cinn_stmts)

    def visit_arguments(self, node):
        """
        Parse CINN Low Level IR Argument.
        If it is not jit mode, it will get information from arg.annoatation.

        Args:
            node(ast.arguments): The ast argument Node

        Returns:
            list[string]: A list of parameter names
        """
        arg_names = [arg.arg for arg in node.args]

        if len(self.inputs_signature) != len(arg_names):
            self.inputs_signature = []
            for arg in node.args:
                arg_annotation = arg.annotation
                if isinstance(arg_annotation, ast.Call):
                    data_array_args = [
                        self.visit(item) for item in arg_annotation.args
                    ]
                    self.inputs_signature.append(DataArray(*data_array_args))
                elif isinstance(arg_annotation, int):
                    if (
                        -(2**21) <= arg_annotation
                        and arg_annotation <= 2**31 - 1
                    ):
                        self.inputs_signature.append("i32")
                    elif (
                        2**63 <= arg_annotation
                        and arg_annotation <= 2**64 - 1
                    ):
                        self.inputs_signature.append("u64")
                    else:
                        self.inputs_signature.append("i64")
                elif isinstance(arg_annotation, float):
                    return self.inputs_signature.append("fp32")
                else:
                    raise TypeError(
                        f'Unsupported type {type(arg_annotation)} for {arg_annotation}'
                    )

        return arg_names

    def visit_For(self, node) -> ir.Expr:
        """
        parse CINN Low Level IR For.

        Args:
            node(ast.For): The ast For node

        Returns:
            ir.Expr, Points to the Expr of ir::ExprNode<For>
        """
        # 1. Parse the iter of the For loop
        iter_args = [self.visit(arg) for arg in node.iter.args]
        assert (
            len(iter_args) <= 2
        ), "CINN Low Level IR does not support setting the range step"
        ast_min = iter_args[0] if len(iter_args) > 1 else self.visit(ast.Num(0))
        ast_extent = iter_args[1] if len(iter_args) > 1 else iter_args[0]

        # TODO(6clc): support sub region's local variable
        # AS code in  `visit_FunctionDef`, store  ir::Expr in local variables
        llir_var = ir.Var(node.target.id)
        llir_var_expr = ir.Expr(llir_var)
        self.set_value(node.target.id, llir_var_expr)

        llir_for_min = ir.Expr(ast_min)
        llir_for_extent = ir.Expr(ast_extent)

        # 2. Parse the body of the For loop
        llir_for_body = self.visit_compound_statement(node.body)
        for_expr = ir.For.make(
            llir_var, llir_for_min, llir_for_extent, llir_for_body
        )
        return for_expr

    def visit_Name(self, node):
        # Store Node
        if type(node.ctx) == ast.Store:
            if node.id in self.variables_table:
                return self.variables_table[node.id]

            # ir::Tensor is not the same store as ir::Var.
            # Let's hand it over to the call
            return node.id
        # Load Node
        assert (
            node.id in self.variables_table
        ), f"{node.id} is not defined in context"
        return self.variables_table[node.id]

    def visit_Subscript(self, node):
        expr_tensor = self.visit(node.value)
        if isinstance(node.slice, (ast.List, ast.Tuple)):
            indices = [self.visit(x) for x in node.slice.elts]
        else:
            indices = [self.visit(node.slice)]
        if type(node.ctx) == ast.Load:
            return ir.Load.make(expr_tensor, indices)
        return expr_tensor, indices

    def visit_Tuple(self, node):
        args = [self.visit(x) for x in node.elts]
        return args

    def visit_Index(self, node):
        return self.visit(node.value)

    def visit_Assign(self, node):
        """
        parse CINN Low Level IR Store.

        Args:
            node(ast.Assign): The ast Assign node

        Returns:
            ir.Expr, Points to the Expr of ir::ExprNode<Store>
        """

        if isinstance(node.value, ast.Call) and node_is_schedule(node.value):
            return "no compute"

        assert (
            len(node.targets) == 1
        ), "Unsupport targets is a \
               list of nodes, like 'a = b = c'"
        lhs = node.targets[0]

        # 1 parse RHS
        rhs_expr = self.visit(node.value)

        # 2 parse LHS
        if isinstance(lhs, ast.Subscript):
            expr_tensor, expr_indices = self.visit(lhs)

            tensor_body = ir.Store.make(expr_tensor, rhs_expr, expr_indices)
            schedule_block = ir.ScheduleBlockRealize.make(
                list(self.schedule_block_iter_var2expr.values()),
                ir.ScheduleBlock.make(
                    list(self.schedule_block_iter_var2expr.keys()),
                    [],
                    [],
                    lhs.value.id,
                    tensor_body,
                ),
            )
            self.schedule_block_iter_var2expr = {}
            return schedule_block
        else:
            iter_var_ids = self.visit(lhs)
            rhs_exprs = rhs_expr
            if not isinstance(iter_var_ids, List):
                iter_var_ids = [iter_var_ids]
                rhs_exprs = [rhs_expr]
            for i in range(len(iter_var_ids)):
                iter_var = ir.Var(iter_var_ids[i])
                iter_var_expr = ir.Expr(iter_var)
                self.set_value(iter_var_ids[i], iter_var_expr)
                self.schedule_block_iter_var2expr[iter_var] = rhs_exprs[i]

            return "no compute"

    def visit_Constant(self, node):
        return ir.Expr(node.value)

    def visit_With(self, node):
        blocks = []
        for ast_block in node.body:
            blocks.append(self.visit(ast_block))
        return ir.Block.make(blocks)

    def visit_Expr(self, node):
        return self.visit(node.value)

    def visit_Call(self, node):
        func_name = node.func.attr
        if node_is_schedule(node) is not None:
            return "no compute"

    # visit Expressions
    def visit_BinOp(self, node):
        args = [self.visit(node.left), self.visit(node.right)]
        ast2cinn = {
            # Binary Op
            ast.Add: ir.Add,
            ast.Sub: ir.Sub,
            ast.Mult: ir.Mul,
            ast.Div: ir.Div,
            ast.Mod: ir.Mod,
            ast.And: ir.And,
            ast.Or: ir.Or,
        }
        return ast2cinn[type(node.op)].make(*args)

    def visit_UnaryOp(self, node):
        args = [self.visit(node.operand)]
        ast2cinn = {
            ast.USub: ir.Minus,
            ast.Not: ir.Not,
        }
        return ast2cinn[type(node.op)].make(*args)

    def visit_Compare(self, node):
        assert (
            len(node.ops) == 1
        ), "Only binary comparison symbols are supported. Expressions such as '1 <= a < 10' are not supported."
        args = [node.left, *node.comparators]
        args = [self.visit(item) for item in args]
        ast2cinn = {
            ast.Eq: ir.EQ,
            ast.NotEq: ir.NE,
            ast.Lt: ir.LT,
            ast.LtE: ir.LE,
            ast.Gt: ir.GT,
            ast.GtE: ir.GE,
        }
        return ast2cinn[type(node.ops[0])].make(*args)

    def visit_If(self, node):
        condition_expr = self.visit(node.test)
        true_expr = self.visit_compound_statement(node.body)
        if len(node.orelse) == 0:
            return ir.IfThenElse.make(condition_expr, true_expr)
        false_expr = self.visit_compound_statement(node.orelse)
        return ir.IfThenElse.make(condition_expr, true_expr, false_expr)

    def set_value(self, name, value: Union[ir.Tensor, ir.Var]):
        if isinstance(value, ir.Tensor):
            value = value.Expr()
        self.variables_table[name] = value
