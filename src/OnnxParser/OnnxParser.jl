module OnnxParser

using LinearAlgebra, DataStructures
using Flux 

import ..NNLoader 
const NNL = NNLoader

const VERBOSE_ONNX = Ref{Int}(0)
set_onnx_verbosity(v::Int) = (VERBOSE_ONNX[] = v)

# whether to convert all weights to Float64
const DOUBLE_PRECISION = Ref(true)
set_double_precision(v::Bool) = (DOUBLE_PRECISION[] = v)


include("util.jl")

# some definitions need to be included before the nodes
include("onnx_net_def.jl")

include("nodes/base.jl")
include("nodes/linear.jl")
include("nodes/nonlinear.jl")
include("nodes/indexing.jl")

include("onnx_net.jl")

load_onnx_model(model_path::String) = NNL.load_network_dict(OnnxType, model_path)

export OnnxType, OnnxNet, set_onnx_verbosity, set_double_precision, get_input_names, get_output_names,
       compute_all_outputs, compute_outputs, compute_output, load_onnx_model
end
