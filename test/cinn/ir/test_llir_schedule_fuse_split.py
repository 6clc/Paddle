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


def test_fuse():
    @to_cinn_llir
    def elementwise_fuse_assign_loop(
        X: DataArray((128, 128, 128)), Y: DataArray((128, 128, 128))
    ):
        for i in range(128):
            for j in range(128):
                for k in range(128):
                    sch.fuse([i, j, k])
                    i1 = i
                    j1 = j
                    k1 = k
                    Y[i1, j1, k1] = X[i1, j1, k1] * 2.0

    @to_cinn_llir
    def elementwise_fuse_assign_loop_gt(
        X: DataArray((128, 128, 128)), Y: DataArray((128, 128, 128))
    ):
        for i in range(2097152):
            i1_1 = (i / 128) / 128
            j1_1 = (i / 128) % 128
            k1_1 = i % 128
            Y[i1_1, j1_1, k1_1] = X[i1_1, j1_1, k1_1] * 2.0

    assert_llir_equal(
        elementwise_fuse_assign_loop, elementwise_fuse_assign_loop_gt
    )


def test_split():
    @to_cinn_llir
    def elementwise_split(
        X: DataArray((128, 128, 128)), Y: DataArray((128, 128, 128))
    ):
        for i in range(128):
            for j in range(128):
                for k in range(128):
                    sch.split(i, factors=[2, 1, 64])
                    sch.split(j, factors=[4, 32])
                    sch.split(k, factors=[16, 8])
                    i1 = i
                    j1 = j
                    k1 = k
                    Y[i1, j1, k1] = X[i1, j1, k1] * 2.0

    @to_cinn_llir
    def elementwise_split_inferred_factor(
        X: DataArray((128, 128, 128)), Y: DataArray((128, 128, 128))
    ):
        for i in range(128):
            for j in range(128):
                for k in range(128):
                    sch.split(i, factors=[-1, 1, 64])
                    sch.split(j, factors=[4, -1])
                    sch.split(k, factors=[-1, 8])
                    i1 = i
                    j1 = j
                    k1 = k
                    Y[i1, j1, k1] = X[i1, j1, k1] * 2.0

    assert_llir_equal(elementwise_split, elementwise_split_inferred_factor)


if __name__ == "__main__":
    test_fuse()
    test_split()
