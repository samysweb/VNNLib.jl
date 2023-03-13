import Base.+
import Base.-
import Base.*
import Base./
import Base.^
import Base.convert


not(f :: Formula) =  CompositeFormula(Not, f)

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
        return CompositeFormula(And, fs)
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
        return CompositeFormula(Or, fs)
	end
end

implies(f :: Formula, g :: Formula) = CompositeFormula(Implies,Formula[f,g])
iff(f :: Formula, g :: Formula) = CompositeFormula(Iff,Formula[f,g])

leq(f :: Term, g :: Term) = ComparisonFormula(LessEqual, f, g)
eq(f :: Term, g :: Term) = ComparisonFormula(Equal, f, g)
less(f :: Term, g :: Term) = ComparisonFormula(Less, f, g)

+(t :: Term, u :: Term) = ArithmeticTerm(Addition, Term[t,u])
-(t :: Term, u :: Term) = ArithmeticTerm(Subtraction, Term[t,u])
*(t :: Term, u :: Term) = ArithmeticTerm(Multiplication, Term[t,u])
/(t :: Term, u :: Term) = ArithmeticTerm(Division, Term[t,u])
^(t :: Term, u :: Term) = ArithmeticTerm(Exponentiation, Term[t,u])

+(t :: Constant, u :: Constant) = Constant(t.value + u.value)
-(t :: Constant, u :: Constant) = Constant(t.value - u.value)
*(t :: Constant, u :: Constant) = Constant(t.value * u.value)
/(t :: Constant, u :: Constant) = Constant(t.value / u.value)
^(t :: Constant, u :: Constant) = Constant(t.value ^ u.value)



convert(::Type{Term}, x :: Number) = Constant(Rational{BigInt}(x))