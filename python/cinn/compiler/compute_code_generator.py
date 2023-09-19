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

from .utils import node_is_schedule, VariableTable
from .expr_executor import ExprExecutor, exec_assign


class ComputeCodeGenerator(ast.NodeVisitor):
    """
    Convert python ast to CINN Lower Level IR,
    containing only the semantics of the compute part
    """

    def __init__(self, fn, function_name, inputs_signature):
        self.fn = fn
        self.function_name = function_name
        self.inputs_signature = inputs_signature
        self.cinn_llir_func = None
        self.variables_table = VariableTable()
        self.extra_scope = {"range": ir.sequential}

    def parse(self):
        ast_node = self.fn.parse()
        with ir.IRBuilder() as builder, self.variables_table:
            for k, v in self.fn.scope.items():
                self.variables_table.add(k, v)
            for k, v in self.extra_scope.items():
                self.variables_table.add(k, v)
            self.visit(ast_node)
        return builder.get()

    def visit_FunctionDef(self, node) -> None:
        """
        Parse CINN Low Level IR FunctionDef.

        Args:
            node(ast.FunctionDef): The ast FunctionDef Node
        """
        with ir.LowerFuncContext(self.function_name) as func_ctx:
            arg_names = self.visit(node.args)

            assert len(node.args.defaults) == 0, "Not support default args"

            # 1. Construct args of function
            for i, arg_name in enumerate(arg_names):
                # Obj of Argument is ir::Buffer
                if hasattr(self.inputs_signature[i], "dtype"):
                    tensor_shape = [
                        ir.Expr(dim) for dim in self.inputs_signature[i].shape
                    ]
                    # TODO(6clc): unify Tensor and Buffer
                    llir_value = ir._Buffer_.make(
                        arg_name, self.inputs_signature[i].dtype
                    )
                    ir.Arg(arg_name, llir_value)
                    llir_value = ir._Tensor_.make(
                        arg_name,
                        self.inputs_signature[i].dtype,
                        tensor_shape,
                        tensor_shape,
                    )
                    self.variables_table.add(arg_name, llir_value)
                # Obj of Argument is ir::Var
                else:
                    llir_value = ir.Var(arg_name)
                    ir.Arg(arg_name, llir_value)
                    llir_value = ir.Expr(llir_value)
                    self.variables_table.add(arg_name, llir_value)

            # 2. Construct body of function
            body = self.visit_compound_statement(node.body)

    def visit_compound_statement(self, stmts):
        for stmt in stmts:
            self.visit(stmt)

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
        """
        for_ctx = ExprExecutor(self.variables_table.get()).exec(node.iter)
        with self.variables_table:
            with for_ctx as loop_var:
                local_var_table = exec_assign(
                    target=node.target, source=loop_var)
                for k, v in local_var_table.items():
                    loop_var.rename(k)
                    self.variables_table.add(k, ir.Expr(v))
                self.visit_compound_statement(node.body)

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
        rhs_expr = ExprExecutor(self.variables_table.get()).exec(node.value)

        # 2 parse LHS
        # 2.1 Tensor
        if isinstance(lhs, ast.Subscript):
            expr_tensor = ExprExecutor(
                self.variables_table.get()).exec(lhs.value)
            if isinstance(lhs.slice, ast.Tuple):
                expr_indices = []
                for idx in lhs.slice.elts:
                    expr_indices.append(ExprExecutor(
                        self.variables_table.get()).exec(idx))
            else:
                expr_indices = [ExprExecutor(
                    self.variables_table.get()).exec(lhs.slice)]
            ir.TensorStore(expr_tensor.Expr(), rhs_expr, expr_indices)
        # 2.2 Attribute of Var
        elif isinstance(lhs, ast.Attribute):
            iter_var = self.visit(lhs.value)
            setattr(iter_var.as_var_mutable(), lhs.attr, self.eval(node.value))
            return "no compute"
        # 2.3 Var
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

    def visit_Call(self, node):
        func_name = node.func.attr
        if node_is_schedule(node) is not None:
            return "no compute"

    def visit_If(self, node):
        condition_expr = self.visit(node.test)
        true_expr = self.visit_compound_statement(node.body)
        if len(node.orelse) == 0:
            return ir.IfThenElse.make(condition_expr, true_expr)
        false_expr = self.visit_compound_statement(node.orelse)
        return ir.IfThenElse.make(condition_expr, true_expr, false_expr)

    def visit_With(self, node):
        for item in node.items:
            frame = sel.eval_expr(item.context_expr)
            rhs = frame
            if item.optional_vars is not None:
                self.eval_assign(target=item.optional_vars, source=rhs)
        body = self.visit_compound_statement(node.body)
        return body
