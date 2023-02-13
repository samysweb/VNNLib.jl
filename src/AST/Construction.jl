using ..Parser

using Tokenize

function generate_variable_dict(output :: ParsingResult, variable_labeler) :: Dict{String, Variable}
    variables = Dict{String, Variable}()
    num_inputs = 0
    for variable in output.variables
        var_type = Tokenize.untokenize(variable.type)
        @assert var_type == "Real" "Currently only real variables are supported"
        name = Tokenize.untokenize(variable.name)
        sort, net_position = variable_labeler(name)
        if sort == Input
            num_inputs += 1
        end
        variables[name] = Variable(name, sort, Ref{Int}(0), net_position)
    end
    i = 0
    j = 0
    for variable in output.variables
        var = variables[Tokenize.untokenize(variable.name)]
        if var.sort == Input
            i += 1
            var.index[] = i
        else
            j += 1
            var.index[] = num_inputs+j
        end
    end
    return variables
end

function process_vnn_expression(expression :: VnnExpression, variables::Dict{String, Variable})
    if expression isa VnnIdentifier
        name = Tokenize.untokenize(expression.name)
        if haskey(variables,name)
            return variables[name]
        else
            error("Unknown variable $name at $(expression.position)")
        end
    elseif expression isa VnnNumber
        return Constant(expression.value)
    elseif expression isa CompositeVnnExpression
        head = Tokenize.untokenize(expression.head)
        arguments = map(x -> process_vnn_expression(x, variables), expression.args)
        return @match head begin
            "and" => CompositeFormula(And, arguments)
            "or" => CompositeFormula(Or, arguments)
            "implies" => CompositeFormula(Implies, arguments)
            "not" => CompositeFormula(Not, arguments)
            "<" => ComparisonFormula(Less, arguments[1], arguments[2])
            "<=" => ComparisonFormula(LessEqual, arguments[1], arguments[2])
            "=" => ComparisonFormula(Equal, arguments[1], arguments[2])
            ">=" => ComparisonFormula(LessEqual, arguments[2], arguments[1])
            ">" => ComparisonFormula(Less, arguments[2], arguments[1])
            "+" => ArithmeticTerm(Addition, arguments)
            "-" => ArithmeticTerm(Subtraction, arguments)
            "*" => ArithmeticTerm(Multiplication, arguments)
            "/" => ArithmeticTerm(Division, arguments)
            "^" => ArithmeticTerm(Exponentiation, arguments)
            _ => error("Unknown expression $head at $(expression.position)")
        end
    else
        error("Unknown expression $(expression)")
    end
end

function process_parser_output(
    output :: ParsingResult,
    variable_labeler)
    variables = generate_variable_dict(output, variable_labeler)
    assertions = Vector{Formula}()
    for assertion in output.assertions
        formula = process_vnn_expression(assertion, variables)
        push!(assertions, formula)
    end
    return CompositeFormula(And, assertions)
end