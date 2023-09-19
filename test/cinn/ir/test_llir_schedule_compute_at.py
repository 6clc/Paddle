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

from test.cinn.utils.testing import assert_llir_equal

import cinn.schedule as sch
from cinn import to_cinn_llir, ir
from cinn.runtime.data_array import DataArray


def test_compute_at_elementwise():
    @to_cinn_llir
    def elementwise_add(
        X: DataArray((128, 128)),
        Y: DataArray((128, 128)),
        A: DataArray((128, 128)),
    ):
        for i in range(128):
            for j in range(128):
                with ir.ScheduleBlockContext("A"):
                    i1, j1 = ir.AxisMap("SS", [i, j])
                    A[i1, j1] = X[i1, j1] * 2.0
        for i3 in range(128):
            for j3 in range(128):
                with ir.ScheduleBlockContext("Y"):
                    i1, j1 = ir.AxisMap("SS", [i3, j3])
                    sch.compute_at(A, i3, False)
                    Y[i1, j1] = A[i1, j1] + 2.0

    @to_cinn_llir
    def elementwise_add_gt(
        X: DataArray((128, 128)),
        Y: DataArray((128, 128)),
        A: DataArray((128, 128)),
    ):
        for i in range(128):
            for j in range(128):
                with ir.ScheduleBlockContext("A"):
                    i1, j1 = ir.AxisMap("SS", [i, 0+j])
                    A[i1, j1] = X[i1, j1] * 2.0
            for k in range(128):
                with ir.ScheduleBlockContext("Y"):
                    i2, k1 = ir.AxisMap("SS", [i, k])
                    Y[i2, k1] = A[i2, k1] + 2.0

    assert_llir_equal(elementwise_add, elementwise_add_gt)


if __name__ == '__main__':
    test_compute_at_elementwise()
