module Linearization

using ..AST
using ..Simplifier

using Base.Rounding

function __init__()
	Rounding.setrounding(BigFloat,Rounding.RoundDown)
end

include("Util.jl")
include("ToLP.jl")

export ast_to_lp

end