/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file src/relay/op/contrib/ethosu/pooling.cc
 * \brief Pooling operators definitions for the Arm(R) Ethos(TM)-U NPU.
 */

#include <cstring>
#include <cmath>

#include <tvm/relay/attrs/finn.h>
#include <tvm/relay/op.h>

#include "../make_op.h"
#include "../op_common.h"
#include "../type_relations.h"

namespace tvm {
namespace relay {
namespace op {
namespace contrib {

TVM_REGISTER_NODE_TYPE(MultiThresholdAttrs);

bool try_process_out_dtype(std::string out_dtype, bool& is_signed, int& bit_width) {
    std::size_t found = out_dtype.find("UINT");
    std::size_t bit_width_index = 4;
    is_signed = found == std::string::npos;

    if (found == std::string::npos) {
        found = out_dtype.find("INT");
        bit_width_index = 3;
        if (found == std::string::npos) return false;
    }

    std::string bit_width_str = out_dtype.substr(bit_width_index);
    if (bit_width_str.size() == 0 || bit_width_str.size() > 2) return false;

    try {
        bit_width = std::stoi(bit_width_str);
        if (bit_width > 64) return false;
        return true;
    }
    catch (...) {
        return false;
    }
}

bool MultiThresholdRel(const Array<Type>& types, int num_inputs, const Attrs& attrs, const TypeReporter& reporter) {
    ICHECK_EQ(types.size(), 3);
    const auto* data = types[0].as<TensorTypeNode>();
    if (data == nullptr) return false;

    const auto* thresholds = types[0].as<TensorTypeNode>();
    if (thresholds == nullptr) return false;

    const MultiThresholdAttrs* params = attrs.as<MultiThresholdAttrs>();
    ICHECK(params != nullptr);
    String out_dtype = params->out_dtype;
    double out_bias = params->out_bias;

    bool out_dtype_signed(true);
    int bit_width(0);
    if (!try_process_out_dtype(static_cast<std::string>(out_dtype), out_dtype_signed, bit_width)) {
        reporter->GetDiagCtx().EmitFatal(Diagnostic::Error(reporter->GetSpan())
                                     << "MultiThreshold out_dtype bad format.");
        return false;
    }

    int bit_width_pow = std::pow(2, bit_width);
    std::size_t thresh_last_index = thresholds->shape.size();
    reporter->AssertEQ(thresholds->shape[thresh_last_index - 1], bit_width_pow);

    if (out_dtype_signed && out_bias != -bit_width_pow / 2) {
        reporter->GetDiagCtx().EmitFatal(Diagnostic::Error(reporter->GetSpan())
                                     << "For a signed out_dtype, out_bias must correspond to 2**(bit_width)/2");
        return false;
    }

    if (!out_dtype_signed && out_bias != 0) {
        reporter->GetDiagCtx().EmitFatal(Diagnostic::Error(reporter->GetSpan())
                                     << "For an unsigned out_dtype, the out_bias must correspond to zero.");
        return false;
    }

    reporter->Assign(types[2], types[0]);
    return true;
}

Expr MakeMultiThreshold(Expr data, Expr thresholds, String out_dtype, double out_bias) {
    auto attrs = make_object<MultiThresholdAttrs>();
    attrs->out_dtype = std::move(out_dtype);
    attrs->out_bias = out_bias;

    static const Op& op = Op::Get("MultiThreshold");
    return Call(op, {data, thresholds}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op.contrib._make.MultiThreshold").set_body_typed(MakeMultiThreshold);

RELAY_REGISTER_OP("MultiThreshold")
    .describe(R"code(Threshold the input data to map it from one domain to another.
)code" TVM_ADD_FILELINE)
    .set_attrs_type<MultiThresholdAttrs>()
    .set_num_inputs(2)
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("thresholds", "Tensor", "The thresholds for thresholding.")
    .set_support_level(9)
    .add_type_rel("MultiThreshold", MultiThresholdRel)
    .set_attr<TOpPattern>("TOpPattern", kBroadcast);

}  // namespace contrib
}  // namespace op
}  // namespace relay
}  // namespace tvm
