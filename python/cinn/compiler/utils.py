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
try:
    from _collections import defaultdict
except ImportError:
    pass


from cinn.schedule import IRSchedule


def node_is_schedule(node: ast.Call):
    func_name = ""
    if isinstance(node.func, ast.Name):
        func_name = node.func.id
    elif isinstance(node.func, ast.Attribute):
        func_name = node.func.attr

    return getattr(IRSchedule, func_name, None)


class VariableTable:
    def __init__(self):
        self.var_name_list = []
        self.name2value = defaultdict(list)

    def __enter__(self):
        self.var_name_list.append([])
        return self

    def __exit__(self, ptype, value, trace) -> None:
        if ptype is None and value is None:
            var_names = self.var_name_list.pop()
            for var_name in var_names:
                self.name2value.pop(var_name)

    def add(self, name, value):
        # TODO(6clc): to check value is equal
        self.var_name_list[-1].append(name)
        self.name2value[name].append(value)

    def get(self):
        return {k: v[-1] for k, v in self.name2value.items()}
