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
from cinn import to_cinn_llir
from cinn.runtime.data_array import DataArray


def test_compute_inline_elementwise():
    @to_cinn_llir
    def elementwise_add_inline(
        X: DataArray((128, 128)),
        Y: DataArray((128, 128)),
        A: DataArray((128, 128)),
    ):
        for i in range(128):
            for j in range(128):
                i1 = i
                j1 = j
                A[i1, j1] = X[i1, j1] * 2.0
                sch.compute_inline(A)
        for i3 in range(128):
            for j3 in range(128):
                i1 = i3
                j1 = j3
                Y[i1, j1] = -A[i1, j1] + 3.0

    @to_cinn_llir
    def elementwise_add_inline_gt(
        X: DataArray((128, 128)),
        Y: DataArray((128, 128)),
        A: DataArray((128, 128)),
    ):
        for i in range(128):
            for j in range(128):
                i1 = i
                j1 = j
                Y[i1, j1] = -(X[i1, j1] * 2.0) + 3.0

    assert_llir_equal(elementwise_add_inline, elementwise_add_inline_gt)


if __name__ == "__main__":
    test_compute_inline_elementwise()
