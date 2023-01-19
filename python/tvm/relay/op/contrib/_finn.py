"""Backend compiler related feature registration"""
from __future__ import absolute_import

from python.tvm import topi
from .. import op as _reg

_reg.register_broadcast_schedule("MultiThreshold")

def elemwise_shape_func(attrs, inputs, _):
    """
    Shape function for elemwise op.
    """
    return [topi.math.identity(inputs[0])]

def MultiThreshold_compute(attrs, inputs, out_type):
    """compute definition for MultiThreshold"""
    data = inputs[0]
    thresholds = inputs[1]
    out_dtype = attrs.out_dtype
    bias = attrs.out_bias

    return data

_reg.register_shape_func("MultiThreshold", False, elemwise_shape_func)
_reg.register_compute("MultiThreshold", MultiThreshold_compute)