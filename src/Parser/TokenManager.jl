function peek_token(tokenmanager :: TokenManager) :: Tokens.Token
	found_token = false
	current_token :: Tokens.Token = peek(tokenmanager.tokens,Tokens.ENDMARKER)
    comment = false
	while !found_token
        current_token = peek(tokenmanager.tokens,Tokens.ENDMARKER)
        if Tokens.kind(current_token) == Tokens.ENDMARKER
            return current_token
        end
        if comment
            if Tokens.kind(current_token) == Tokens.WHITESPACE && !isnothing(findfirst('\n',Tokens.untokenize(current_token)))
                comment = false
            end
            popfirst!(tokenmanager.tokens)
            continue
        end
		if Tokens.kind(current_token) == Tokens.SEMICOLON
			comment = true
            popfirst!(tokenmanager.tokens)
            continue
        end
        if Tokens.kind(current_token) == Tokens.WHITESPACE
            popfirst!(tokenmanager.tokens)
            continue
        end
		found_token = true
	end
	if isnothing(current_token)
        return Tokens.ENDMARKER
    else
        return current_token
    end
end

function next(tokenmanager :: TokenManager) :: Tokens.Token
	found_token = false
	current_token :: Tokens.Token = peek(tokenmanager.tokens,Tokens.ENDMARKER)
    comment = false
	while !found_token
		current_token = popfirst!(tokenmanager.tokens)
        if Tokens.kind(current_token) == Tokens.ENDMARKER
            return current_token
        end
        if comment
            if Tokens.kind(current_token) == Tokens.WHITESPACE && !isnothing(findfirst('\n',Tokens.untokenize(current_token)))
                comment = false
            end
            continue
        end
		if Tokens.kind(current_token) == Tokens.SEMICOLON
			comment = true
            continue
        end
        if Tokens.kind(current_token) == Tokens.WHITESPACE
            continue
        end
		found_token = true
	end
	if isnothing(current_token)
        return Tokens.ENDMARKER
    else
        return current_token
    end
end