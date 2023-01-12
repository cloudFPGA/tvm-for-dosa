from . import _make

def MultiThreshold(
    data,
    param,
    out_dtype,
    out_bias,
    tvm_custom,
):
    out_dtype = out_dtype.decode('utf-8')
    return _make.MultiThreshold(data, param, out_dtype, out_bias)