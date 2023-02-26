using VNNLib.Parser: parse_tokens
using VNNLib.Parser:
	DeclaredVariable,
	CompositeVnnExpression,
	VnnIdentifier,
	VnnNumber

function check_expression(requirement, assert_expression)
	if requirement[1] == :composite
		@test assert_expression isa CompositeVnnExpression
		@test Tokens.untokenize(assert_expression.head) == requirement[2]
		@test length(assert_expression.args) == length(requirement[3])
		for (i, arg) in enumerate(assert_expression.args)
			check_expression(requirement[3][i], arg)
		end
	elseif requirement[1] == :identifier
		@test assert_expression isa VnnIdentifier
		@test Tokens.untokenize(assert_expression.name) == requirement[2]
	elseif requirement[1] == :number
		@test assert_expression isa VnnNumber
		@test assert_expression.value == requirement[2]
	else
		error("Unknown requirement type $(requirement[1])")
	end
end

@testset "Parser" begin
	test_inputs = [
		(
			"""
			(declare-const X_0 Real)
			""",
			[("X_0","Real")],
			[],
			true
		)
		(
			"""
			(assert (<= X_0 1.0))
			""",
			[],
			[
				(:composite, "<=", [(:identifier, "X_0"), (:number, 1//1)])
			],
			true
		)
		(
			"""
			(assert (<= X_0 (+ 1.0 X_1)))
			""",
			[],
			[
				(:composite,
					"<=",
					[
						(:identifier, "X_0"), 
						(:composite, "+", [(:number, 1//1), (:identifier, "X_1")])
					]
				)
			],
			true
		)
		(
			"""
			(assert 
				(and
					(<= X_0 (+ 1.0 X_1))
					(> X_2 2.0)
				)
			)
			""",
			[],
			[
				(:composite, "and",
				[
					(:composite,
						"<=",
						[
							(:identifier, "X_0"), 
							(:composite, "+", [(:number, 1//1), (:identifier, "X_1")])
						]
					),
					(:composite,
						">",
						[
							(:identifier, "X_2"), 
							(:number, 2//1)
						]
					)
				]
				)
			],
			true
		)
		(
			"""
			(declare-const X_0 Real)
			(declare-const X_1 Real)

			(assert 
				(or
					(= X_0 1.0)
					(= X_1 2.0)
					(and
						(<= X_1 (+ 1.0 X_0))
						(> X_0 0.5)
					)
				)
			)
			""",
			[("X_0","Real"),("X_1","Real")],
			[
				(:composite, "or",
				[
					(:composite,
						"=",
						[
							(:identifier, "X_0"), 
							(:number, 1//1)
						]
					),
					(:composite,
						"=",
						[
							(:identifier, "X_1"), 
							(:number, 2//1)
						]
					),
					(:composite, "and",
					[
						(:composite,
							"<=",
							[
								(:identifier, "X_1"), 
								(:composite, "+", [(:number, 1//1), (:identifier, "X_0")])
							]
						),
						(:composite,
							">",
							[
								(:identifier, "X_0"), 
								(:number, 1//2)
							]
						)
					]
					)
				]
				)
			],
			true
		)
		(
			"""
			(declare-const X_0 Real)
			(declare-const X_1 Real)

			(assert (<= X_0 (+ 1.0 X_1)))
			(assert (<= X_1 (+ 1.0 X_0)))
			(assert (= X_0 X_1))
			""",
			[("X_0","Real"),("X_1","Real")],
			[
				(:composite,
					"<=",
					[
						(:identifier, "X_0"), 
						(:composite, "+", [(:number, 1//1), (:identifier, "X_1")])
					]
				),
				(:composite,
					"<=",
					[
						(:identifier, "X_1"), 
						(:composite, "+", [(:number, 1//1), (:identifier, "X_0")])
					]
				),
				(:composite,
					"=",
					[
						(:identifier, "X_0"), 
						(:identifier, "X_1")
					]
				)
			],
			true
		)
	]
	for (input, vars, asserts, run) in test_inputs
		if !run
			continue
		end
		t = create_tm(input)
		result = parse_tokens(t)
		@test length(result.variables) == length(vars)
		@test all(map(x->Tokens.kind(x.name)==Tokens.IDENTIFIER && Tokens.kind(x.type)==Tokens.IDENTIFIER, result.variables))
		for (name, type) in vars
			@test (name, type) ∈ map(x->(Tokens.untokenize(x.name),Tokens.untokenize(x.type)),result.variables)
		end
		@test length(result.assertions) == length(asserts)
		for (requirement, parsed_assert) in zip(asserts, result.assertions)
			check_expression(requirement, parsed_assert)
		end
	end
end