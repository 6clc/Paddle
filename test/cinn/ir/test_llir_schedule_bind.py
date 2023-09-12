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


import sys
import cinn
import numpy as np
from cinn import to_cinn_llir
from cinn.runtime.data_array import DataArray
import cinn.schedule as sch


def test_bind_reduce():

    @to_cinn_llir
    def reduce_sum(A: DataArray((1, 4, 256, 512)), B: DataArray((1, 4, 256))):

        for i1 in range(1):
            for j1 in range(4):
                for k1 in range(256):
                    with ir.ScheduleBlockRealize("init") as init:
                        vi, vj, vk = i1, j1, k1
                        B[vi, vj, vk] = 0.0
                    for l1 in range(512):
                        sch.bind(i1, "blockIdx.x")
                        sch.bind(j1, "threadIdx.z")
                        sch.bind(k1, "threadIdx.x")
                        vi1, vj1, vk1, vl1 = i1, j1, k1, l1
                        vl1.is_reduce_axis = True
                        B[vi1, vj1, vk1] = B[vi1, vj1, vk1] + \
                            A[vi1, vj1, vk1, vl1]

    print(reduce_sum)


if __name__ == "__main__":
    test_bind_reduce()
