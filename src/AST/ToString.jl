import Base.show
ast_to_string(f :: CompositeFormula) = @match f.head begin
    And => "(and " * join(map(ast_to_string,f.args), " ") * " )"
    Or => "(or " * join(map(ast_to_string,f.args), " ") * " )"
    Not => "(not " * ast_to_string(f.args[1]) * " )"
    Implies => "(implies " * ast_to_string(f.args[1]) * " " * ast_to_string(f.args[2]) * " )"
    Iff => "(iff " * ast_to_string(f.args[1]) * " " * ast_to_string(f.args[2]) * " )"
end

ast_to_string(f :: True) = "true"
ast_to_string(f :: False) = "false"

function ast_to_string(f :: ComparisonFormula)
    @match f.head begin
        Less => "(< " * ast_to_string(f.left) * " " * ast_to_string(f.right) * " )"
        LessEqual => "(<= " * ast_to_string(f.left) * " " * ast_to_string(f.right) * " )"
        Equal => "(= " * ast_to_string(f.left) * " " * ast_to_string(f.right) * " )"
    end
end

ast_to_string(f :: ArithmeticTerm) = "(" * ast_to_string(f.head) * " " * join(map(ast_to_string,f.args), " ") * " )"

ast_to_string(f :: Variable) = string(f.name)

ast_to_string(f :: Constant) = string(convert(Float64,f.value))

function ast_to_string(f :: BoundConstraint)
    op = @match f.head begin
        Less => "<" 
        LessEqual => "<="
    end
    if f.var_index > 0
        return "(" * op * " v$(abs(f.var_index)) " * string(convert(Float64,f.bound)) * ")"
    else
        return "(" * op * " -v$(abs(f.var_index)) " * string(convert(Float64,f.bound)) * ")"
    end
end

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