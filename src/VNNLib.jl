module VNNLib

include("Parser/Main.jl")

using .Parser

include("AST/Main.jl")

using .AST

include("Simplifier/Main.jl")

using .Simplifier

include("Linearization/Main.jl")

using .Linearization

include("Iterator/Main.jl")

using .Iterator

include("NNLoader/Main.jl")

using .NNLoader

function default_labeler(name)
    parts = split(name,"_")
    if length(parts) == 2
        if parts[1] == "x" || parts[1] == "X"
            return AST.Input, (parse(Int,parts[2]),)
        elseif parts[1] == "y" || parts[1] == "Y"
            return AST.Output, (parse(Int,parts[2]),)
        else
            error("Unknown variable $name")
        end
    else
        error("Unknown variable $name")
    end
end

function get_ast(filename :: String,variable_labeler=default_labeler)
    parser_result = parse_file(filename)
    return process_parser_output(parser_result,variable_labeler)
end

export get_ast, iterate, AST, ast_to_lp

export Network, Layer, Dense, ReLU
export load_network



end # module VNNLib
