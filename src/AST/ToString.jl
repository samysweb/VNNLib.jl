import Base.show

ast_to_string(f :: CompositeFormula{And,<:Formula}) = "(and" * join(map(ast_to_string,f.args), " ") * " )"
ast_to_string(f :: CompositeFormula{Or,<:Formula}) = "(or" * join(map(ast_to_string,f.args), " ") * " )"
ast_to_string(f :: CompositeFormula{Not,<:Formula}) = "(not" * ast_to_string(f.args[1]) * " )"
ast_to_string(f :: CompositeFormula{Implies,<:Formula}) = "(implies " * ast_to_string(f.args[1]) * " " * ast_to_string(f.args[2]) * " )"
ast_to_string(f :: CompositeFormula{Iff,<:Formula}) = "(iff " * ast_to_string(f.args[1]) * " " * ast_to_string(f.args[2]) * " )"

ast_to_string(f :: True) = "true"
ast_to_string(f :: False) = "false"

function ast_to_string(f :: ComparisonFormula)
    @match f.head begin
        Less => "(< " * ast_to_string(f.left) * " " * ast_to_string(f.right) * " )"
        LessEqual => "(<= " * ast_to_string(f.left) * " " * ast_to_string(f.right) * " )"
        Equal => "(= " * ast_to_string(f.left) * " " * ast_to_string(f.right) * " )"
    end
end

function ast_to_string(f :: LinearConstraint)
    return "(linear constraint)"
end

ast_to_string(f :: ArithmeticTerm{<:Term}) = "(" * ast_to_string(f.head) * " " * join(map(ast_to_string,f.args), " ") * " )"

ast_to_string(f :: Variable) = string(f.name)

ast_to_string(f :: Constant) = string(convert(Float64,f.value))

function ast_to_string(op :: Arithmetic)
    return @match op begin
        Addition => "+"
        Subtraction => "-"
        Multiplication => "*"
        Division => "/"
        Exponentiation => "^"
    end
end

function show(io::IO, p :: ASTNode)
	print(io, ast_to_string(p))
end