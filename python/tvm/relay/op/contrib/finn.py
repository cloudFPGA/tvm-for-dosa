from . import _make

def MultiThreshold(
    data,
    param,
    out_dtype,
    out_bias,
    tvm_custom,
):
    """FINN MultiThreshold operator (see arXiv:1709.04060).

        This operator takes data as input from a given domain (floating point or integer) and maps it to another domain (necessarily integer). That is, for a value x in the input, the output integer corresponds to the number of thresholds that x is greater or equal to.

        Parameters
        ----------
        data : relay.Expr
            The input data to the operator.

        param: relay.Expr
            The thresholds values.

        out_dtype: str
            Type to return.

        out_bias : float
            bias added to the output integer

        Returns
        -------
        result: relay.Expr
            The integer result of the new domain.
        """

    out_dtype = out_dtype.decode('utf-8')
    return _make.MultiThreshold(data, param, out_dtype, out_bias)