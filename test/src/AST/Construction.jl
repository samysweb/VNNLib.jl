
using VNNLib.Parser: ParsingResult
using VNNLib.AST: Input, Output, ComparisonFormula, Variable, Constant, CompositeFormula, VnnExpression,ArithmeticTerm
using VNNLib.AST: And, Or, Implies, Iff, Equal, LessEqual, Less
using VNNLib.AST: Addition, Subtraction, Multiplication, Division, Exponentiation
using VNNLib.AST: generate_variable_dict, process_vnn_formula

@testset "Construction" begin
    @testset "generate_variable_dict" begin
        parsed = ParsingResult()
        push!(parsed.variables, DeclaredVariable(
            generate_token("X_1", Tokens.IDENTIFIER),
            generate_token("Real", Tokens.IDENTIFIER),
            (1,1)
        ))
        push!(parsed.variables, DeclaredVariable(
            generate_token("X_2", Tokens.IDENTIFIER),
            generate_token("Real", Tokens.IDENTIFIER),
            (1,1)
        ))
        push!(parsed.variables, DeclaredVariable(
            generate_token("X_3", Tokens.IDENTIFIER),
            generate_token("Real", Tokens.IDENTIFIER),
            (1,1)
        ))
        push!(parsed.variables, DeclaredVariable(
            generate_token("Y_1", Tokens.IDENTIFIER),
            generate_token("Real", Tokens.IDENTIFIER),
            (1,1)
        ))
        push!(parsed.variables, DeclaredVariable(
            generate_token("Y_2", Tokens.IDENTIFIER),
            generate_token("Real", Tokens.IDENTIFIER),
            (1,1)
        ))
        function labeler(var_name)
            letter, number = split(var_name,"_")
            if letter == "X"
                return Input, (parse(Int,number),)
            elseif letter == "Y"
                return Output, (parse(Int,number),)
            else
                error("Unknown variable $var_name")
            end
        end
        variables = generate_variable_dict(parsed, labeler)
        @test variables["X_1"].sort == Input
        @test variables["X_1"].net_position == (1,)
        @test variables["X_1"].name == "X_1"
        @test variables["X_2"].sort == Input
        @test variables["X_2"].net_position == (2,)
        @test variables["X_2"].name == "X_2"
        @test variables["X_3"].sort == Input
        @test variables["X_3"].net_position == (3,)
        @test variables["X_3"].name == "X_3"
        # Same for Y_i
        @test variables["Y_1"].sort == Output
        @test variables["Y_1"].net_position == (1,)
        @test variables["Y_1"].name == "Y_1"
        @test variables["Y_2"].sort == Output
        @test variables["Y_2"].net_position == (2,)
        @test variables["Y_2"].name == "Y_2"
        # Check that all variables have unique index
        @test length(unique([var.index[] for var in values(variables)])) == length(values(variables))
        @test length(values(variables)) == 5
    end
    @testset "process_vnn_formula" begin
        variables = Dict{String, Variable}(
            "X_1" => Variable("X_1", Input, Ref{Int}(1), (1,)),
            "X_2" => Variable("X_2", Input, Ref{Int}(2), (2,)),
            "X_3" => Variable("X_3", Input, Ref{Int}(3), (3,)),
            "Y_1" => Variable("Y_1", Output, Ref{Int}(4), (1,)),
            "Y_2" => Variable("Y_2", Output, Ref{Int}(5), (2,)),
        )
        parsed1 = nothing
        parsed2 = nothing
        result1 = nothing
        result2 = nothing
        for (op_string, op_type) in [(Tokens.LESS,Less),(Tokens.LESS_EQ,LessEqual)]

            parsed1 = CompositeVnnExpression(
                generate_token("", op_string),
                [
                    VnnIdentifier(
                        generate_token("X_1", Tokens.IDENTIFIER),
                        (1,1)
                    ),
                    VnnNumber(
                        Rational{BigInt}(1//1),
                        (1,1)
                    )
                ],
                (1,1)
            )
            result1 = process_vnn_formula(
                parsed1,
                variables
            )
            @test result1 isa ComparisonFormula
            @test result1.head == op_type
            @test result1.left isa Variable
            @test result1.left.name == "X_1"
            @test result1.left.sort == Input
            @test result1.left.net_position == (1,)
            @test result1.left.index[] == 1
            @test result1.right isa Constant
            @test isone(result1.right.value)
        end

        for (op_string, op_type) in [(Tokens.PLUS,Addition),(Tokens.MINUS,Subtraction),(Tokens.STAR,Multiplication),(Tokens.FWD_SLASH,Division),(Tokens.CIRCUMFLEX_ACCENT,Exponentiation)]
            parsed2 = CompositeVnnExpression(
                generate_token("", Tokens.EQ),
                VnnExpression[
                    VnnIdentifier(
                        generate_token("Y_2", Tokens.IDENTIFIER),
                        (1,1)
                    ),
                    CompositeVnnExpression(
                        generate_token("", op_string),
                        VnnExpression[
                            VnnNumber(
                                Rational{BigInt}(1//1),
                                (1,1)
                            ),
                            VnnNumber(
                                Rational{BigInt}(1//1),
                                (1,1)
                            )
                        ],
                        (1,1)
                    )
                ],
                (1,1)
            )
            result2 = process_vnn_formula(
                parsed2,
                variables
            )
            @test result2 isa ComparisonFormula
            @test result2.head == Equal
            @test result2.left isa Variable
            @test result2.left.name == "Y_2"
            @test result2.left.sort == Output
            @test result2.left.net_position == (2,)
            @test result2.left.index[] == 5
            @test result2.right isa ArithmeticTerm
            @test result2.right.head == op_type
            @test result2.right.args[1] isa Constant
            @test isone(result2.right.args[1].value)
            @test result2.right.args[2] isa Constant
            @test isone(result2.right.args[2].value)
        end


        for (op_string,op_type) in [("and",And),("or",Or),("implies",Implies),("iff",Iff)]
            result3 = process_vnn_formula(
                CompositeVnnExpression(
                    generate_token(op_string, Tokens.IDENTIFIER),
                    VnnExpression[
                        parsed1,
                        parsed2
                    ],
                    (1,1)
                ),
                variables
            )
            @test result3 isa CompositeFormula
            @test result3.head == op_type
            @test isequal(result3.args[1], result1)
            @test isequal(result3.args[2], result2)
        end
    end
end