module OnnxSimplifier

using LinearAlgebra
using Flux

using ..OnnxParser 
const OXP = OnnxParser


batched_linearization(node::OXP.Node) = true

# linearization for nodes with multiple inputs runs into problems when only one input has a batch_dim!
batched_linearization(node::OXP.ONNXConcat) = false  # can't concat tensors, when one has batch dim and the others don't
batched_linearization(node::OXP.ONNXGather) = false  # same problem as with Concat 


include("affine2matrix.jl")
include("convert2dense.jl")

export net2dense

end