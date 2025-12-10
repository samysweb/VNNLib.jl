using Test
using VNNLib

include("src/Parser/Main.jl")
include("src/AST/Main.jl")

if Sys.WORD_SIZE == 64
	include("src/onnx_utils.jl")
	include("src/OnnxParser/onnx_tests.jl")
	include("src/OnnxSimplifier/simplifier_test.jl")
end

detect_ambiguities(
	VNNLib;
	recursive = true
)

detect_unbound_args(
	VNNLib;
	recursive = true
)