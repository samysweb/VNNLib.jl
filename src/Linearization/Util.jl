@inline function round_minimize(x :: Rational{BigInt}) :: Float32
	return Float32(BigFloat(x))
end

@inline function round_maximize(x :: Rational{BigInt}) :: Float32
	return -Float32(BigFloat(-x))
end