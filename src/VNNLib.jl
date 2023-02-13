module VNNLib

include("Parser/Main.jl")

using .Parser

include("AST/Main.jl")

using .AST

include("Simplifier/Main.jl")

using .Simplifier

function default_labeler(name)
    parts = split(name,"_")
    if length(parts) == 2
        if lowercase(parts[1]) == "x"
            return AST.Input, (parse(Int,parts[2]),)
        elseif lowercase(parts[1]) == "y"
            return AST.Output, (parse(Int,parts[2]),)
        else
            error("Unknown variable $name")
        end
    else
        error("Unknown variable $name")
    end
end

function get_ast(filename :: String,variable_labeler=default_labeler)
    parser_result = Parser.parse_file(filename)
    return AST.process_parser_output(parser_result,variable_labeler)
end

export get_ast, ast_to_lp



end # module VNNLib
