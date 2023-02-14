module Simplifier

    using ..AST

    using MLStyle
    using SymbolicUtils

    include("TermInterface.jl")
    include("SymbolicUtils.jl")

    export prepare_linearization, to_dnf
end