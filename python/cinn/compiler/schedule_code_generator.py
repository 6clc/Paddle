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

from cinn.schedule import IRSchedule

from .utils import node_is_schedule


class ScheduleCodeGenerator(ast.NodeVisitor):
    """
    Convert python ast to CINN Lower Level IR,
    containing only the semantics of the schedule part
    """

    def __init__(self, cinn_llir_func):
        self.cinn_llir_func = cinn_llir_func
        self.scheduler = IRSchedule.make(self.cinn_llir_func)
        self.sch_seq = []
        self.name2loops = {}

    def visit_Subscript(self, node):
        """
        save block information
        """
        if type(node.ctx) != ast.Store:
            return

        for sch_node in self.sch_seq:
            block_name2loops = self.scheduler.get_name2loops_dict(node.value.id)
            for k, v in block_name2loops.items():
                self.name2loops[k] = v

            # schedule node is ast.Call or ast.Assign
            sch_call_node = (
                sch_node if isinstance(sch_node, ast.Call) else sch_node.value
            )

            sch_name = (
                sch_call_node.func.id
                if isinstance(sch_call_node.func, ast.Name)
                else sch_call_node.func.attr
            )
            sch_args = [self.eval(item) for item in sch_call_node.args]

            sch_keywords = {
                kw.arg: self.eval(kw.value) for kw in sch_call_node.keywords
            }

            ret = getattr(self.scheduler, sch_name)(*sch_args, **sch_keywords)

            if isinstance(sch_node, ast.Assign):
                assert (
                    len(sch_node.targets) == 1
                ), "Unsupport targets is a \
               list of nodes, like 'a = b = c'"
                var_name = self.visit(sch_node.targets[0])
                if not isinstance(var_name, list):
                    var_name = [var_name]
                for k, v in zip(var_name, ret):
                    self.name2loops[k] = v

        self.sch_seq = []
        self.name2loops = {}

    def visit_Assign(self, node):
        if isinstance(node.value, ast.Call) and node_is_schedule(node.value):
            self.sch_seq.append(node)
            return
        self.generic_visit(node)

    def visit_Call(self, node):
        if not node_is_schedule(node):
            return
        self.sch_seq.append(node)

    def visit_Tuple(self, node):
        elts = [self.visit(x) for x in node.elts]
        return elts

    def visit_Name(self, node):
        return node.id

    def eval(self, node):
        return getattr(self, f'eval_{type(node).__name__}')(node)

    def eval_List(self, node):
        return [self.eval(item) for item in node.elts]

    def eval_Tuple(self, node):
        return [self.eval(item) for item in node.elts]

    def eval_Constant(self, node):
        return node.value

    def eval_UnaryOp(self, node):
        return eval(
            compile(ast.Expression(body=node), filename='', mode='eval')
        )

    def eval_Name(self, node):
        try:
            if node.id in self.name2loops:
                return self.name2loops[node.id]
            else:
                return self.scheduler.get_block(node.id)
        except:
            raise Exception(
                f'No matching block and loop was found for {node.id}. \
                 Current loops are {self.name2loops.keys()}. \
                 Current lower ir is {self.cinn_llir_func}.'
            )
