module OnnxSimplifier

using LinearAlgebra
using Flux

using ..OnnxParser 
const OXP = OnnxParser


batched_linearization(node::OXP.Node) = true

# TODO: can we do better here? E.g. only transpose the non-batch dimensions?
# transpose requires transposing the input to the linear layer, so we would also transpose the batch dim!
batched_linearization(node::OXP.ONNXLinear) = !node.transpose

# We just call reshape(x, shape), so if we put the identity matrix for x, it does not fit the shape!
batched_linearization(node::OXP.ONNXReshape) = false

# linearization for nodes with multiple inputs runs into problems when only one input has a batch_dim!
batched_linearization(node::OXP.ONNXConcat) = false  # can't concat tensors, when one has batch dim and the others don't
batched_linearization(node::OXP.ONNXGather) = false  # same problem as with Concat 

linear_special_case(node::OXP.Node) = false
linear_special_case(node::OXP.ONNXLinear) = node.transpose  # if transposed, we can't do the linearization as usual


include("affine2matrix.jl")
include("convert2dense.jl")

export net2dense

end