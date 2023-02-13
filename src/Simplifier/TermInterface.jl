import SymbolicUtils.istree
import SymbolicUtils.operation
import SymbolicUtils.arguments
import SymbolicUtils.similarterm
import SymbolicUtils.symtype
import SymbolicUtils.promote_symtype
import SymbolicUtils.is_literal_number

istree(f::CompositeFormula{<:Connective,<:Formula}) = true
istree(f::ComparisonFormula) = true
istree(f::ArithmeticTerm) = true

istree(f::True) = false
istree(f::False) = false
istree(f::LinearConstraint) = false
istree(f::Variable) = false
istree(f::Constant) = false


operation(f::CompositeFormula{And,<:Formula}) = (and)
operation(f::CompositeFormula{Or,<:Formula}) = (or)
operation(f::CompositeFormula{Not,<:Formula}) = (not)
operation(f::CompositeFormula{Implies,<:Formula}) = (implies)
operation(f::CompositeFormula{Iff,<:Formula}) = (iff)
function operation(f::ComparisonFormula)
    return @match f.head begin
        Less => (<)
        LessEqual => (<=)
        Equal => (==)
    end
end
function operation(f::ArithmeticTerm{<:Term})
    return @match f.head begin
        Addition => (+)
        Subtraction => (-)
        Multiplication => (*)
        Division => (/)
        Exponentiation => (^)
        _ => error("Unknown arithmetic term $f")
    end
end

arguments(f :: CompositeFormula{<:Connective,<:Formula}) = f.args
arguments(f :: ComparisonFormula) = [f.left, f.right]
arguments(f :: ArithmeticTerm) = f.args

function similarterm(f :: Type{ASTNode}, head, args)
    print("similarterm type")
    return f
end

function similarterm(f :: ASTNode, head, args)
    print("similarterm")
    return f
end

function similarterm(f :: CompositeFormula{<:Connective,<:Formula}, head, args)
    return @match head begin
        (and) => CompositeFormula(And, args)
        (or) => CompositeFormula(Or, args)
        (not) => CompositeFormula(Not, args)
        (implies) => CompositeFormula(Implies, args)
        (iff) => CompositeFormula(Iff, args)
    end
end

function similarterm(f :: ComparisonFormula, head, args)
    return @match head begin
        (<) => ComparisonFormula(Less, args[1], args[2])
        (<=) => ComparisonFormula(LessEqual, args[1], args[2])
        (==) => ComparisonFormula(Equal, args[1], args[2])
    end
end

function similarterm(f :: ArithmeticTerm, head, args)
    if head == (+)
        return ArithmeticTerm(Addition, args)
    elseif head == (-)
        return ArithmeticTerm(Subtraction, args)
    elseif head == (*)
        return ArithmeticTerm(Multiplication, args)
    elseif head == (/)
        return ArithmeticTerm(Division, args)
    elseif head == (^)
        return ArithmeticTerm(Exponentiation, args)
    else
        error("Unknown arithmetic term $f")
    end
end


function not_division(x :: Term)
	return !(x isa ArithmeticTerm) || operation(x) != (/) && (!istree(x) || all(y->not_division(y), arguments(x)))
end

function _isone(x :: Term)
	return x isa Constant && isone(x.value)
end

function _iszero(x :: Term)
	return x isa Constant && iszero(x.value)
end

function _istwo(x :: Term)
	return x isa Constant && x.value == 2
end

function _isnotzero(x :: Term)
	return !_iszero(x)
end

function _istrue(x :: Formula)
	return x isa True
end

function _isfalse(x :: Formula)
	return x isa False
end

function is_literal_number(x :: Term)
	return x isa Constant
end

#exprhead(f :: ASTNode) = :call