import Base.+
import Base.-
import Base.*
import Base./
import Base.^
import Base.convert


not(f :: T1) where {T1 <: Formula} =  CompositeFormula{Not, T1}(Not, f)

function and(fs :: T1...) where {T1 <: Formula}
	return and_construction(fs)
end

function and_construction(fs)
	if fs isa Formula
		# In case there is only one element in and
		return fs
	end
	if length(fs) == 0
		return TrueAtom()
	elseif length(fs) == 1
		return fs[1]
	else
        return CompositeFormula{And, Formula}(And, fs)
	end
end
function or(fs :: T1...) where {T1 <: Formula}
	return or_construction(fs)
end

function or_construction(fs)
	if fs isa Formula
		# In case there is only one element in and
		return fs
	end
	if length(fs) == 0
		return TrueAtom()
	elseif length(fs) == 1
		return fs[1]
	else
        return CompositeFormula{Or, Formula}(Or, fs)
	end
end

implies(f :: T1, g :: T2) where {T1 <: Formula, T2 <: Formula} = CompositeFormula{Implies,T1}(Implies,Formula[f,g])
iff(f :: T1, g :: T2) where {T1 <: Formula, T2 <: Formula} = CompositeFormula{Iff,T1}(Iff,Formula[f,g])

leq(f :: T1, g :: T2) where {T1 <: Term, T2 <: Term} = ComparisonFormula(LessEqual, f, g)
eq(f :: T1, g :: T2) where {T1 <: Term, T2 <: Term} = ComparisonFormula(Equal, f, g)
less(f :: T1, g :: T2) where {T1 <: Term, T2 <: Term} = ComparisonFormula(Less, f, g)

+(t :: T1, u :: T2) where {T1 <: Union{Term,Number}, T2 <: Union{Term,Number}} = ArithmeticTerm(Addition, Term[t,u])
-(t :: T1, u :: T2) where {T1 <: Union{Term,Number}, T2 <: Union{Term,Number}} = ArithmeticTerm(Subtraction, Term[t,u])
*(t :: T1, u :: T2) where {T1 <: Union{Term,Number}, T2 <: Union{Term,Number}} = ArithmeticTerm(Multiplication, Term[t,u])
/(t :: T1, u :: T2) where {T1 <: Union{Term,Number}, T2 <: Union{Term,Number}} = ArithmeticTerm(Division, Term[t,u])
^(t :: T1, u :: T2) where {T1 <: Union{Term,Number}, T2 <: Union{Term,Number}} = ArithmeticTerm(Exponentiation, Term[t,u])

+(t :: Constant, u :: Constant) = Constant(t.value + u.value)
-(t :: Constant, u :: Constant) = Constant(t.value - u.value)
*(t :: Constant, u :: Constant) = Constant(t.value * u.value)
/(t :: Constant, u :: Constant) = Constant(t.value / u.value)
^(t :: Constant, u :: Constant) = Constant(t.value ^ u.value)



convert(::Type{Term}, x :: Number) = Constant(Rational{BigInt}(x))