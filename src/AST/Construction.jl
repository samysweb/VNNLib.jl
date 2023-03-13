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

function process_vnn_term(var :: VnnIdentifier, variables) :: Variable
    name = Tokenize.untokenize(var.name)
    result = get(variables,name,nothing)
    if !isnothing(result)
        return result
    else
        error("Unknown variable $name at $(var.position)")
    end
end

function process_vnn_term(var :: VnnNumber, variables) :: Constant
    return Constant(var.value)
end

function process_vnn_term(expression :: CompositeVnnExpression, variables) :: ArithmeticTerm
    head = Tokenize.untokenize(expression.head)
    if !(head ∈ ["+", "-", "*", "/", "^"])
        error("Expected arithmetic term, got comparison at $(expression.position)")
    end
    arguments = map(x -> process_vnn_term(x, variables), expression.args)
    return @match head begin
        "+" => ArithmeticTerm(Addition, arguments)
        "-" => ArithmeticTerm(Subtraction, arguments)
        "*" => ArithmeticTerm(Multiplication, arguments)
        "/" => ArithmeticTerm(Division, arguments)
        "^" => ArithmeticTerm(Exponentiation, arguments)
        _ => error("Unknown expression $head at $(expression.position)")
    end
end

function process_vnn_comparison(head :: String, assertion :: CompositeVnnExpression, variables) :: ComparisonFormula
    left :: Term = process_vnn_term(assertion.args[1], variables)
    right :: Term = process_vnn_term(assertion.args[2], variables)
    return @match head begin
        "<" => ComparisonFormula(Less, left, right)
        "<=" => ComparisonFormula(LessEqual, left, right)
        "=" => ComparisonFormula(Equal, left, right)
        ">=" => ComparisonFormula(LessEqual, right, left)
        ">" => ComparisonFormula(Less, right, left)
        _ => error("Unknown comparison $head at $(assertion.position)")
    end
end

function process_vnn_composite(head :: String, assertion :: CompositeVnnExpression, variables) :: CompositeFormula
    if head == "not"
        @assert length(assertion.args) == 1
        argument = process_vnn_formula(assertion.args[1], variables)
        return CompositeFormula(Not, [argument])
    else
        arguments = Vector{Formula}(undef, length(assertion.args))
        for (i,arg) in enumerate(assertion.args)
            arguments[i] = process_vnn_formula(arg, variables)
        end
        return @match head begin
            "and" => CompositeFormula(And, arguments)
            "or" => CompositeFormula(Or, arguments)
            "implies" => CompositeFormula(Implies, arguments)
            "iff" => CompositeFormula(Iff, arguments)
            _ => error("Unknown composite $head at $(assertion.position)")
        end
    end
end

function process_vnn_formula(assertion :: CompositeVnnExpression, variables) :: Formula
    head = Tokenize.untokenize(assertion.head)
    if head=="<" || head=="<=" || head=="=" || head==">=" || head==">"
        return process_vnn_comparison(head, assertion, variables)
    else
        @assert head == "and" || head == "or" || head=="not" || head == "implies" || head == "iff" #∈ ["and", "or", "implies", "not"]
        return process_vnn_composite(head, assertion, variables)
    end
end

function process_parser_output(
    output :: ParsingResult,
    variable_labeler)
    variables = generate_variable_dict(output, variable_labeler)
    n_input = 0
    n_output = 0
    for variable in values(variables)
        if variable.sort == Input
            n_input += 1
        else
            n_output += 1
        end
    end
    assertions = Vector{Formula}()
    for assertion in output.assertions
        formula = process_vnn_formula(assertion, variables)
        push!(assertions, formula)
    end
    return CompositeFormula(And, assertions), n_input, n_output
end