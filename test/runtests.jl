using Test
using VNNLib

include("src/Parser/Main.jl")
include("src/AST/Main.jl")

detect_ambiguities(
	VNNLib;
	recursive = true
)

detect_unbound_args(
	VNNLib;
	recursive = true
)