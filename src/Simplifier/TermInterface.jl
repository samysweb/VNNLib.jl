import SymbolicUtils.istree
import SymbolicUtils.operation
import SymbolicUtils.arguments
import SymbolicUtils.similarterm
import SymbolicUtils.symtype
import SymbolicUtils.promote_symtype
import SymbolicUtils.is_literal_number

istree(f::CompositeFormula) = true
istree(f::ComparisonFormula) = true
istree(f::ArithmeticTerm) = true

istree(f::True) = false
istree(f::False) = false
istree(f::Variable) = false
istree(f::Constant) = false
istree(f::BoundConstraint) = false

operation(f::CompositeFormula) = @match f.head begin
    And => (and)
    Or => (or)
    Not => (not)
    Implies => (implies)
    Iff => (iff)
    _ => error("Unknown composite formula $f")
end

function operation(f::ComparisonFormula)
    return @match f.head begin
        Less => (<)
        LessEqual => (<=)
        Equal => (==)
    end
end
function operation(f::ArithmeticTerm)
    return @match f.head begin
        Addition => (+)
        Subtraction => (-)
        Multiplication => (*)
        Division => (/)
        Exponentiation => (^)
        _ => error("Unknown arithmetic term $f")
    end
end

arguments(f :: CompositeFormula) = f.args
arguments(f :: ComparisonFormula) = [f.left, f.right]
arguments(f :: ArithmeticTerm) = f.args

function similarterm(f :: ASTNode, head, args)
    error("Missing function similarterm for $(typeof(f)))")
end

function similarterm(f :: CompositeFormula, head, args)
    if head == (and)
        return CompositeFormula(And, args)
    elseif head == (or)
        return CompositeFormula(Or, args)
    elseif head == (not)
        return CompositeFormula(Not, args)
    elseif head == (implies)
        return CompositeFormula(Implies, args)
    elseif head == (iff)
        return CompositeFormula(Iff, args)
    else
        error("Unknown composite formula $f")
    end
end

similarterm(f :: CompositeFormula, head, args,_) = similarterm(f, head, args)

function similarterm(f :: ComparisonFormula, head, args)
    if head == (==)
        return ComparisonFormula(Equal, args[1], args[2])
    elseif head == (<=)
        return ComparisonFormula(LessEqual, args[1], args[2])
    elseif head == (<)
        return ComparisonFormula(Less, args[1], args[2])
    else
        error("Unknown comparison formula $f")
    end
end

similarterm(f :: ComparisonFormula, head, args,_) = similarterm(f, head, args)

function similarterm(f :: ArithmeticTerm, head, args)
    args = convert(Vector{Term}, args)
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

similarterm(f :: ArithmeticTerm, head, args, _) = similarterm(f, head, args)


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

function _isvar(x :: Term)
    return x isa Variable
end

#exprhead(f :: ASTNode) = :call