import Base.isequal

isequal(f1 :: CompositeFormula{T,<:Formula}, f2 :: CompositeFormula{T,<:Formula}) where T<:Connective = f1.head == f2.head && isequal(f1.args, f2.args)
isequal(::True, ::True) = true
isequal(::False, ::False) = true
isequal(f1 :: ComparisonFormula, f2 :: ComparisonFormula) = f1.head == f2.head && isequal(f1.left, f2.left) && isequal(f1.right, f2.right)
isequal(f1::LinearConstraint, f2::LinearConstraint) = f1.head == f2.head && isequal(f1.term, f2.term) && isequal(f1.constant, f2.constant)
isequal(f1::ArithmeticTerm{T}, f2::ArithmeticTerm{T}) where T<:Term = f1.head == f2.head && isequal(f1.args, f2.args)
isequal(f1::Variable, f2::Variable) = f1.index[] == f2.index[]
isequal(f1::Constant, f2::Constant) = f1.value == f2.value