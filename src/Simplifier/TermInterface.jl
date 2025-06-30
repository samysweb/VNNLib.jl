import SymbolicUtils.istree
import TermInterface.operation
import TermInterface.arguments
import TermInterface.maketerm
import TermInterface.head
import TermInterface.iscall
import SymbolicUtils.symtype
import SymbolicUtils.promote_symtype
import SymbolicUtils.is_literal_number


#head(f::CompositeFormula) = :call
#children(f::CompositeFormula) = [operation(f); f.args]
iscall(f::CompositeFormula) = true
iscall(f::ComparisonFormula) = true
iscall(f::ArithmeticTerm) = true

istree(f::CompositeFormula) = true
istree(f::ComparisonFormula) = true
istree(f::ArithmeticTerm) = true

istree(f::True) = false
istree(f::False) = false
istree(f::Variable) = false
istree(f::Constant) = false
istree(f::BoundConstraint) = false

function operation(f::CompositeFormula)
    return @match f.head begin
        And => (and)
        Or => (or)
        Not => (not)
        Implies => (implies)
        Iff => (iff)
        _ => error("Unknown composite formula $f")
    end
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

function maketerm(T::Type{ASTNode}, callhead, args, type=nothing, metadata=nothing)
    error("Missing function maketerm for type $T and callhead $(callhead))")
end

function maketerm(::Type{CompositeFormula}, head, args, type=nothing, metadata=nothing)
    #@assert callhead == :call "Expected a call expression! Got $callhead, args: $args"
    #head = args[1]
    #args = args[2:end]
    
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


function maketerm(T::Type{ComparisonFormula}, head, args, type=nothing, metadata=nothing)
   
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


function maketerm(::Type{ArithmeticTerm}, head, args, type=nothing, metadata=nothing)
   
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

function _isvar(x :: Term)
    return x isa Variable
end

#exprhead(f :: ASTNode) = :call