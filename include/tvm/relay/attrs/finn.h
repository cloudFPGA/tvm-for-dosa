#ifndef TVM_RELAY_ATTRS_FINN_
#define TVM_RELAY_ATTRS_FINN_

#include <tvm/ir/attrs.h>
#include <tvm/relay/base.h>

#include <string>

namespace tvm{
namespace relay{

/*! \brief Attributes used in FINN MultiThreshold operator */
struct MultiThresholdAttrs: public tvm::AttrsNode<MultiThresholdAttrs> {
    tvm::String out_dtype;
    double out_bias;

    TVM_DECLARE_ATTRS(MultiThresholdAttrs, "relay.attrs.MultiThresholdAttrs") {
        TVM_ATTR_FIELD(out_dtype).describe("The output dtype of the data");
        TVM_ATTR_FIELD(out_bias).describe("The bias added to the data (typically for unsigned integer)");
    }
};

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_ATTRS_FINN_