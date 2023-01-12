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

bool MultiThresholdRel(const Array<Type>& types, int num_inputs, const Attrs& attrs, const TypeReporter& reporter) {
    ICHECK_EQ(types.size(), 3);
    const auto* data = types[0].as<TensorTypeNode>();
    if (data == nullptr) return false;

    const MultiThresholdAttrs* params = attrs.as<MultiThresholdAttrs>();
    ICHECK(params != nullptr);
    String out_dtype = params->out_dtype;

    if (out_dtype == "") {
        reporter->GetDiagCtx().EmitFatal(Diagnostic::Error(reporter->GetSpan())
                                     << "The output dtype must be defined;");
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
