using Test
using VNNLib

include("src/Parser/Main.jl")
include("src/AST/Main.jl")
include("src/OnnxParser/onnx_tests.jl")

detect_ambiguities(
	VNNLib;
	recursive = true
)

detect_unbound_args(
	VNNLib;
	recursive = true
)