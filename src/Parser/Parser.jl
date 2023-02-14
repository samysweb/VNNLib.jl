function parse_file(filename :: String)
    # Parse VNNLib File
    open(filename, "r") do file_io
        return parse_io(file_io)
    end
end

function CustomLexer(io::IO_t, T::Type{TT} = Token) where {IO_t,TT <: Tokens.AbstractToken}
    c1 = ' '
    p1 = position(io)
    if eof(io)
        c2, p2 = EOF_CHAR, p1
        c3, p3 = EOF_CHAR, p1
    else
        c2 = read(io, Char)
        p2 = position(io)
        if eof(io)
            c3, p3 = EOF_CHAR, p1
        else
            c3 = read(io, Char)
            p3 = position(io)
        end

    end
    return Tokenize.Lexers.Lexer{IO_t,T}(io, position(io), 1, 1, position(io), 1, 1, position(io), Tokens.ERROR, IOBuffer(;sizehint=64), (c1,c2,c3), (p1,p2,p3), false, false)
end

function parse_io(io :: IO)
    token_manager = TokenManager(tokenize(io))#CustomLexer(io,Tokens.Token))
    return parse_tokens(token_manager)
end

function parse_variable_declaration(token_manager :: TokenManager, result :: ParsingResult)
    token1 = next(token_manager)
    if Tokens.kind(token1) != Tokens.IDENTIFIER
        error("Expected (name) identifier but found '$(token1)' at ", Tokens.startpos(token))
    end
    name = token1
    token2 = next(token_manager)
    if Tokens.kind(token2) != Tokens.IDENTIFIER
        error("Expected (type) identifier but found '$(token2)' at ", Tokens.startpos(token))
    end
    type = token2
    token = next(token_manager)
    if Tokens.kind(token) != Tokens.RPAREN
        error("Expected ')' but found '$(token)' at ", Tokens.startpos(token))
    end
    push!(result.variables, DeclaredVariable(name, type, Tokens.startpos(token1)))
end

function parse_expression(token_manager :: TokenManager)
    token = peek_token(token_manager)
    if Tokens.kind(token) == Tokens.IDENTIFIER
        next(token_manager)
        return VnnIdentifier(token, Tokens.startpos(token))
    elseif Tokens.exactkind(token) == Tokens.FLOAT
        next(token_manager)
        return VnnNumber(Rational{BigInt}(parse(BigFloat, Tokens.untokenize(token))), Tokens.startpos(token))
    elseif Tokens.kind(token) == Tokens.LPAREN
        next(token_manager)
        token = next(token_manager)
        if Tokens.kind(token) != Tokens.IDENTIFIER && Tokens.kind(token) != Tokens.OP
            error("Expected (expression head) identifier but found '$(token)' at ", Tokens.startpos(token))
        end
        head = token
        args = Vector{VnnExpression}()
        token = peek_token(token_manager)
        while Tokens.kind(token) != Tokens.RPAREN && Tokens.kind(token) != Tokens.ENDMARKER
            push!(args, parse_expression(token_manager))
            token = peek_token(token_manager)
        end
        if Tokens.kind(token) != Tokens.RPAREN
            error("Expected ')' but found '$(token)' at ", Tokens.startpos(token))
        end
        next(token_manager)
        return CompositeVnnExpression(head, args, Tokens.startpos(token))
    elseif Tokens.exactkind(token) == Tokens.MINUS
        next(token_manager)
        token = next(token_manager)
        if Tokens.exactkind(token) == Tokens.FLOAT
            return VnnNumber(Rational{BigInt}(-parse(BigFloat, Tokens.untokenize(token))), Tokens.startpos(token))
        else
            error("Expected float but found '$(token)' at ", Tokens.startpos(token))
        end
    else
        error("Expected expression but found '$(token)' at ", Tokens.startpos(token))
    end
end

function parse_assertion(token_manager :: TokenManager, result :: ParsingResult)
    expression = parse_expression(token_manager)
    push!(result.assertions, expression)
end

function parse_tokens(token_manager :: TokenManager)
    result = ParsingResult()
    token = peek_token(token_manager)
    while Tokens.kind(token) != Tokens.ENDMARKER
        if Tokens.kind(token) != Tokens.LPAREN
            error("Expected '(' but found '$(token)' at ", Tokens.startpos(token))
        end
        next(token_manager)
        token = next(token_manager)
        if Tokens.kind(token) == Tokens.IDENTIFIER && Tokens.untokenize(token)=="declare"
            token1 = next(token_manager)
            token2 = next(token_manager)
            if Tokens.exactkind(token1) != Tokens.MINUS || Tokens.kind(token2) != Tokens.KEYWORD || Tokens.untokenize(token2) != "const"
                error("Expected 'declare-const' but found '$(token)$(token1)$(token2)' at ", Tokens.startpos(token))
            end
            parse_variable_declaration(token_manager, result)
        elseif Tokens.kind(token) == Tokens.IDENTIFIER && Tokens.untokenize(token)=="assert"
            parse_assertion(token_manager, result)
            token = next(token_manager)
            if Tokens.kind(token) != Tokens.RPAREN
                error("Expected ')' but found '$(token)' at ", Tokens.startpos(token))
            end
        else
            error("Expected 'declare' or 'assert' but found '$(token)' at ", Tokens.startpos(token))
        end
        token = peek_token(token_manager)
    end
    return result
end