import Base.Iterators.approx_iter_type

Base.Iterators.approx_iter_type(::Type{Tokenize.Lexers.Lexer{IO_t, Tokens.Token}}) where IO_t<:IO = Tuple{Tokens.Token,Bool}

mutable struct TokenManager{IO_t <: IO}
	tokens :: Iterators.Stateful{Tokenize.Lexers.Lexer{IO_t, Tokens.Token},Tuple{Tokens.Token,Bool}}
	function TokenManager(tokens::Tokenize.Lexers.Lexer{IO_t, Tokens.Token}) where IO_t <: IO 
        return new{IO_t}(Iterators.Stateful(tokens))
    end
end

struct DeclaredVariable
    name :: Tokens.Token
    type :: Tokens.Token
    position :: Tuple{Int,Int}
    DeclaredVariable(name :: Tokens.Token, type :: Tokens.Token, position :: Tuple{Int,Int}) = new(name, type, position)
end

abstract type VnnExpression end

struct CompositeVnnExpression <: VnnExpression
    head :: Tokens.Token
    args :: Vector{VnnExpression}
    position :: Tuple{Int,Int}
    CompositeVnnExpression(head :: Tokens.Token, args :: Vector{VnnExpression}, position :: Tuple{Int,Int}) = new(head, args, position)
end

struct VnnIdentifier <: VnnExpression
    name :: Tokens.Token
    position :: Tuple{Int64,Int64}
    VnnIdentifier(name :: Tokens.Token, position :: Tuple{Int64,Int64}) = new(name, position)
end

struct VnnNumber <: VnnExpression
    value :: Rational{BigInt}
    position :: Tuple{Int,Int}
    VnnNumber(value :: Rational{BigInt}, position :: Tuple{Int,Int}) = new(value, position)
end

mutable struct ParsingResult
    variables :: Vector{DeclaredVariable}
    assertions :: Vector{VnnExpression}
    ParsingResult() = new(Vector{DeclaredVariable}(), Vector{VnnExpression}())
end