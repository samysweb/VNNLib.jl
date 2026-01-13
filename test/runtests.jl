using Test
using VNNLib

include("src/Parser/Main.jl")
include("src/AST/Main.jl")

if Sys.WORD_SIZE == 64
	include("src/OnnxParser/onnx_tests.jl")
end

detect_ambiguities(
	VNNLib;
	recursive = true
)

detect_unbound_args(
	VNNLib;
	recursive = true
)