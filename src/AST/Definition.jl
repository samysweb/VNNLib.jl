using MLStyle.AbstractPatterns: literal

abstract type ASTNode end

abstract type Formula <: ASTNode end
abstract type Atom <: Formula end
abstract type Term <: ASTNode end

abstract type Connective end
abstract type And <: Connective end
abstract type Or <: Connective end
abstract type Not <: Connective end
abstract type Implies <: Connective end
abstract type Iff <: Connective end

@as_record struct CompositeFormula{C <: Connective, F <: Formula} <: Formula
    head :: Type{C}
    args :: Vector{F}
end


@as_record struct True <: Atom end
@as_record struct False <: Atom end

@enum Comparison begin
    Less
    LessEqual
    Equal
end 
MLStyle.is_enum(::Comparison) = true
MLStyle.pattern_uncall(a::Comparison, _, _, _, _) = literal(a)

@as_record struct ComparisonFormula <: Atom
    head :: Comparison
    left :: Term
    right :: Term
end

@as_record struct BoundConstraint <: Atom
    head :: Comparison
    # abs(var_index) is index of the Variable v
    # If index is positive, then v <= bound
    # If index is negative, then -v <= bound
    var_index :: Int64
    bound :: Rational{BigInt}
end

@enum Arithmetic begin
    Addition
    Subtraction
    Multiplication
    Division
    Exponentiation
end 
MLStyle.is_enum(::Arithmetic) = true
MLStyle.pattern_uncall(a::Arithmetic, _, _, _, _) = literal(a)

@as_record struct ArithmeticTerm{T <: Term} <: Term
    head :: Arithmetic
    args :: Vector{T}
end

@enum VariableSort begin
    Input
    Output
end
MLStyle.is_enum(::VariableSort) = true
MLStyle.pattern_uncall(a::VariableSort, _, _, _, _) = literal(a)

@as_record struct Variable <: Term
    name :: String
    sort :: VariableSort
    index :: Ref{Int}
    net_position :: Tuple{Vararg{Int}}
end

@as_record struct Constant <: Term
    value :: Rational{BigInt}
end
