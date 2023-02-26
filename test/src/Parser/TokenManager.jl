using VNNLib.Parser: peek_token, next

@testset "TokenManager" begin
	test_inputs = [
		("1.0", Tokens.FLOAT, "1.0",true)
		("X_1", Tokens.IDENTIFIER, "X_1",true)
		(")", Tokens.RPAREN,")",true)
		("(", Tokens.LPAREN,"(",true)
		("	)", Tokens.RPAREN,")",true)
		("-1.0", Tokens.OP, "-",true)
		("	1.0", Tokens.FLOAT, "1.0",true)
		("	X_1", Tokens.IDENTIFIER, "X_1",true)
		("	-", Tokens.OP, "-",true)
		("""
		1.0
		""", Tokens.FLOAT, "1.0",true)
		(""";test 123 test my comment - + * / 1.0
		X_0
		""", Tokens.IDENTIFIER, "X_0",true)
		(""";;;;; test ;;;;			;
		; line two X_1 2.0		
		1.0""", Tokens.FLOAT, "1.0",true)
		("",Tokens.ENDMARKER,nothing,true)
		(";",Tokens.ENDMARKER,nothing,true)
		("""
		;
		""",Tokens.ENDMARKER,nothing,true)
		("""; test
		; test two 1.0
		; test three 2.0
		""",Tokens.ENDMARKER,nothing,true)
		("""; test
		""",Tokens.ENDMARKER,nothing,true)
	]
	for (input, kind, untokenize,run) in test_inputs
		if !run
			continue
		end
		t = peek_token(create_tm(input))
		@test Tokens.kind(t) == kind
		if !isnothing(untokenize)
			@test Tokens.untokenize(t) == untokenize
		end
		t = next(create_tm(input))
		@test Tokens.kind(t) == kind
		if !isnothing(untokenize)
			@test Tokens.untokenize(t) == untokenize
		end
		@inferred peek_token(create_tm(input))
		@inferred next(create_tm(input))
	end
end