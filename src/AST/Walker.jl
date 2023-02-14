@inline function MapReduce(fun, fold_fun, init :: T, f :: CompositeFormula{<:Connective,<:Formula}) :: T where T<:Any
    return fold_fun(mapreduce(arg -> MapReduce(fun, fold_fun, init, arg), fold_fun, f.args; init=init), fun(f))
end

@inline function MapReduce(fun, fold_fun, init :: T, f :: True) :: T where T<:Any
    return fold_fun(init, fun(f))
end

@inline function MapReduce(fun, fold_fun, init :: T, f :: False) :: T where T<:Any
    return fold_fun(init, fun(f))
end

@inline function MapReduce(fun, fold_fun, init :: T, f :: ComparisonFormula) :: T where T<:Any
    return fold_fun(
        fold_fun(MapReduce(fun, fold_fun, init, f.left),MapReduce(fun, fold_fun, init, f.right)),
        fun(f))
end

@inline function MapReduce(fun, fold_fun, init :: T, f :: ArithmeticTerm) :: T where T<:Any
    return fold_fun(mapreduce(arg -> MapReduce(fun, fold_fun, init, arg), fold_fun, f.args; init=init), fun(f))
end

@inline function MapReduce(fun, fold_fun, init :: T, f :: Variable) :: T where T<:Any
    return fold_fun(init, fun(f))
end

@inline function MapReduce(fun, fold_fun, init :: T, f::BoundConstraint) :: T where T<:Any
    return fold_fun(init, fun(f))
end

@inline function MapReduce(fun, fold_fun, init :: T, f :: Constant) :: T where T<:Any
    return fold_fun(init, fun(f))
end

@inline function Map(fun, f :: CompositeFormula{<:Connective,<:Formula})
    return (typeof(f))(f.head, map(arg -> Map(fun, arg), f.args))
end

@inline function Map(fun, f :: True)
    return fun(f)
end

@inline function Map(fun, f :: False)
    return fun(f)
end

@inline function Map(fun, f :: ComparisonFormula)
    return (typeof(f))(f.head, Map(fun, f.left), Map(fun, f.right))
end

@inline function Map(fun, f :: ArithmeticTerm)
    return (typeof(f))(f.head, map(arg -> Map(fun, arg), f.args))
end

@inline function Map(fun, f :: Variable)
    return fun(f)
end

@inline function Map(fun, f::BoundConstraint)
    return fun(f)
end

@inline function Map(fun, f :: Constant)
    return fun(f)
end