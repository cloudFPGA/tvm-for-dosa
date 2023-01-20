# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""MultiThreshold operator"""
import typing

from tvm import te
from tvm import topi


def get_transpose_dims(data):
    dims = list(range(len(data.shape)))
    dims[0] = 1
    dims[1] = 0
    return dims


def reshape_data_for_broadcast(data, thresholds):
    assert data.shape[1] == thresholds.shape[0], "shapes not compatible for broadcast"

    single_elem_in_batch = data.shape[0].value == 1
    if single_elem_in_batch:
        data = topi.squeeze(data, 0)
    else :
        data = topi.transpose(data, get_transpose_dims(data))

    data = topi.expand_dims(data, -1)
    data = topi.repeat(data, thresholds.shape[1].value, -1)
    return data, single_elem_in_batch


def restore_result_shape(res, single_elem_in_batch):
    if single_elem_in_batch:
        res = topi.expand_dims(res, -1)
    return topi.transpose(res, get_transpose_dims(res))

def multi_threshold(
    data: te.Tensor,
    thresholds: te.Tensor,
    bit_width: int,
    signed: bool,
    out_bias: int
) -> typing.List[te.Tensor]:
    """Batch normalization layer (Ioffe and Szegedy, 2014).

    This operator takes data as input from a given domain (floating point or integer) and maps it to another domain (necessarily integer). That is, for a value x in the input, the output integer corresponds to the number of thresholds that x is greater or equal to.

    Parameters
    ----------
    data : tvm.te.Tensor
        Input to the multi-threshold operator.

    thresholds : tvm.te.Tensor
        The threshold values.

    bit_width: int
        The bit-width of the integer domain.

    signed : bool
        Whether the integer domain is signed or not.

    out_bias : float
        The biass added to the output

    Returns
    -------
    output : tvm.te.Tensor
        The computed result of same shape as of the input but now in the target integer domain.
    """
    assert thresholds.shape[-1].value < 2**bit_width
    assert not signed or out_bias < 0

    data_reshaped, single_elem_in_batch = reshape_data_for_broadcast(data, thresholds)
    compare = topi.greater_equal(data_reshaped, thresholds).astype('float32')
    compare_sum = topi.sum(compare, -1)

    # restore correct shape
    res = restore_result_shape(compare_sum, single_elem_in_batch).astype('float32')
    return res
