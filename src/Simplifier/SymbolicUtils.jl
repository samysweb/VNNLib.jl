import SymbolicUtils.simplify
import SymbolicUtils.<ₑ
import SymbolicUtils.Chain
import SymbolicUtils.Fixpoint
import SymbolicUtils.If
import SymbolicUtils.Postwalk
import SymbolicUtils.Prewalk
import SymbolicUtils.PassThrough

# Many of the rules are based on the rules provided by SymbolicUtils.simplify with adjustments for our AST types.

PLUS_RULES = [
	@rule(~x::SymbolicUtils.isnotflat(+) => SymbolicUtils.flatten_term(+, ~x))
	#@rule(~x::SymbolicUtils.needs_sorting(+) => SymbolicUtils.sort_args(+, ~x))
	@SymbolicUtils.ordered_acrule(~a::is_literal_number + ~b::is_literal_number => ~a + ~b)

	#@acrule(*(~~x) + *(~β, ~~x) => *(1 + ~β, (~~x)...))
	#@acrule(*(~α, ~~x) + *(~β, ~~x) => *(~α + ~β, (~~x)...))
	#@acrule(*(~~x, ~α) + *(~~x, ~β) => *(~α + ~β, (~~x)...))

	#@acrule(~x + *(~β, ~x) => *(1 + ~β, ~x))
	@acrule(*(~α::is_literal_number, ~x) + ~x => *(~α + 1, ~x))
	@rule(+(~~x::SymbolicUtils.hasrepeats) => +(SymbolicUtils.merge_repeats(*, ~~x)...))

	@SymbolicUtils.ordered_acrule((~z::_iszero + ~x) => ~x)
	@rule(+(~x) => ~x)
]

# Push divisions to the front of multiplication so they can be removed...
function mul_comp(a,b)
	if istree(a) && operation(a) == (/) && (!istree(b) || operation(b) != /)
		return true
	elseif (!istree(a) || operation(a) != (/)) && istree(b) && operation(b) == (/)
		return false
	else
		return a <ₑ b
	end
end

mul_needs_sorting(f) = x -> SymbolicUtils.is_operation(f)(x) && !issorted(arguments(x), lt=mul_comp)

function sort_mul_args(f, t)
    args = arguments(t)
    if length(args) < 2
		return maketerm(typeof(t), :call, [f, args...])
        #return similarterm(t, f, args)
    elseif length(args) == 2
        x, y = args
		args1 = mul_comp(x,y) ? [x,y] : [y,x]
		return maketerm(typeof(t), :call, [f, args1...])
        #return similarterm(t, f, mul_comp(x,y) ? [x,y] : [y,x])
    end
    args = args isa Tuple ? [args...] : args
	args1 = sort(args, lt=mul_comp)
	return maketerm(typeof(t), :call, [f, args1...])
    #similarterm(t, f, sort(args, lt=mul_comp))
end

TIMES_RULES = [
	@rule(~x::SymbolicUtils.isnotflat(*) => SymbolicUtils.flatten_term(*, ~x))
	#@rule(~x::mul_needs_sorting(*) => sort_mul_args(*, ~x))

	@SymbolicUtils.ordered_acrule(~a::is_literal_number * ~b::is_literal_number => ~a * ~b)
	@rule(*(~~x::SymbolicUtils.hasrepeats) => *(SymbolicUtils.merge_repeats(^, ~~x)...))

	@acrule((~y)^(~n) * ~y => (~y)^(~n+1))
	@SymbolicUtils.ordered_acrule((~x)^(~n) * (~x)^(~m) => (~x)^(~n + ~m))

	@SymbolicUtils.ordered_acrule((~z::_isone  * ~x) => ~x)
	@SymbolicUtils.ordered_acrule((~z::_iszero *  ~x) => ~z)
	@rule(*(~x) => ~x)
]


POW_RULES = [
	#@rule(^(*(~~x), ~y::SymbolicUtils._isinteger) => *(map(a->pow(a, ~y), ~~x)...))
	#@rule((((~x)^(~p::SymbolicUtils._isinteger))^(~q::SymbolicUtils._isinteger)) => (~x)^((~p)*(~q)))
	@rule(^(~x, ~z::_iszero) => Constant(1))
	@rule(^(~x, ~z::_isone) => ~x)
	@rule(^(~x::_isone, ~y) => Constant(1))
	@rule (^(~a::is_literal_number, ~b::is_literal_number) => ^(~a, ~b))
	@rule(^(+(~x,~y), ~z::_istwo) => +(^(~x, ~z), *(~z, ~x, ~y), ^(~y, ~z)))
	@rule( ( (~x) / (~y)  ) ^ (~z) => ( ( (~x)^(~z) )/( (~y)^(~z) ) ) )
	@rule(^(*(~~x),~y) => *(map(a->^(a,~y), ~~x)...))
	@rule(^(^(~x,~y::is_literal_number), ~z::is_literal_number) => ^(~x, ~y*~z))
	@rule(inv(~x) => 1/(~x))
]

ASSORTED_RULES = [
	#@rule(identity(~x) => ~x)
	@rule(-(~x) => Constant(-1.0) * ~x)
	@rule(-(~x, ~y) => ~x + Constant(-1.0) * (~y))
	#@rule(~x::_isone \ ~y => ~y)
	#@rule(~x \ ~y => ~y / (~x))
	#@rule(one(~x) => one(symtype(~x)))
	#@rule(zero(~x) => zero(symtype(~x)))
	#@rule(ifelse(~x::is_literal_number, ~y, ~z) => ~x ? ~y : ~z)
	# DIV Rules
	@rule (~x / ~x => Constant(1.0))
	@rule(~x / (~y) => (~x) * (Constant(1.0) / ~y))
	@acrule +((~z / ~y)*~x,~~xs) => (~z / ~y)*+(~x, map(a->a*(~y/~z), ~~xs)...)
	@acrule +(~x*(~z / ~y),~~xs) => (~z / ~y)*+(~x, map(a->a*(~y/~z), ~~xs)...)
	@acrule (~a/~b)*(~c/~d) => (~a*~c)/(~b*~d)
	@acrule (~a/~b::_isone) => ~a
	@rule (~a::is_literal_number / ~b::is_literal_number => ~a / ~b)
	@acrule ( *((~z / ~y), ~y) => ~z )
	# TODO(steuber): Push even further outwards by multiplying other parts...
]
MINMAX_RULES = [
	@rule ( (min(~x::is_literal_number, ~y::is_literal_number)) => (min((~x).value, (~y).value)) )
	@rule ( (max(~x::is_literal_number, ~y::is_literal_number)) => (max((~x).value, (~y).value)) )
]

function number_simplifier()
	rule_tree = [If(istree, Chain(ASSORTED_RULES)),
				 If(SymbolicUtils.is_operation(+),
					Chain(PLUS_RULES)),
				 If(SymbolicUtils.is_operation(*),
					Chain(TIMES_RULES)),
				 If(SymbolicUtils.is_operation(^),
					Chain(POW_RULES)),
				 If( x-> istree(x) && (operation(x) == min || operation(x) == max),
				 	Chain(MINMAX_RULES))]

	return Fixpoint(Chain(rule_tree))
end

function composite_formula_simplifier()
	Postwalk(
		Chain(
			[
				@acrule ((and(~x::_istrue, ~~y)) => (and_construction(~~y)))
				@acrule ((and(~x::_isfalse, ~~y)) => (FalseAtom()))
				@acrule ((or(~x::_istrue, ~~y)) => (TrueAtom()))
				@acrule ((or(~x::_isfalse, ~~y)) => (or_construction(~~y)))
				@rule   ((not(~x::_istrue)) => (FalseAtom()))
				@rule   ((not(~x::_isfalse)) => (TrueAtom()))
				@rule   (not(implies(~x,~y)) => (and_construction(Formula[~x, not(~y)])))
				@acrule ((and(and(~~x), ~~y)) => (and_construction(Formula[~~x; ~~y])))
				@acrule ((or(or(~~x), ~~y)) => (or_construction(Formula[~~x; ~~y])))
			]
		)
	)
end

function dnf_simplifier()
	Chain(
		[
			If(x->x.head==Or, Chain([
				@acrule ((or(or(~~x), ~~y)) => (or_construction([~~x; ~~y])))
			]))
			If(x->x.head==And, Chain([
				@acrule ((and(and(~~x), ~~y)) => (and_construction([~~x; ~~y])))
				@acrule ((and(or(~~x), ~~y)) => (distribute_or(~~x, ~~y)))
			]))
			If(x->x.head==Not,Chain([
				@rule   ((not(~x::_istrue)) => (FalseAtom()))
				@rule   ((not(~x::_isfalse)) => (TrueAtom()))
				@rule   (not(implies(~x,~y)) => (and_construction([~x, not(~y)])))
			]))
		]
	)
end

distribute_or(x,y) = or_construction([and_construction(Formula[x1;y]) for x1 in x])

function solve_concrete_atom(f, a :: Constant, b :: Constant)
	if f(a.value,b.value)
		return TrueAtom()
	else
		return FalseAtom()
	end
end

function atom_to_bounds()
	Chain(
		[
			@rule (~a::_isvar <= ~b::is_literal_number => BoundConstraint(LessEqual, (~a).index[], (~b).value))
			@rule (~a::_isvar < ~b::is_literal_number => BoundConstraint(Less, (~a).index[], (~b).value))
			@rule (~a::is_literal_number <= ~b::_isvar => BoundConstraint(LessEqual, -((~b).index[]), -((~a).value)))
			@rule (~a::is_literal_number < ~b::_isvar => BoundConstraint(Less, -((~b).index[]), -((~a).value)))
		]
	)
end

function atom_simplifier()
	Chain(
		[
			atom_to_bounds(),
			If(x -> x isa Atom, Chain(
				[
					@rule (~a <= ~b::_isnotzero => leq(~a - ~b, Constant(0.0)))
					@rule (~a < ~b::_isnotzero => less(~a - ~b, Constant(0.0)))
					@rule (~a == ~b::_isnotzero => eq(~a - ~b, Constant(0.0)))
				]
			)),
		]
	)
end

function formula_simplifier()
	Postwalk(
		Chain([
			If(x -> x isa Atom, atom_simplifier()),
		])
	)
end

function distribute_factor(x,ys)
	# @debug "Distributing factor: ", x, " over ", ys
	return map(y->x*y,ys)
end

function distribution_rules()
	# TODO(steuber): Not if DIV
	Fixpoint(Chain([
			@acrule( (~x::not_division) * (+(~~y)) => +(distribute_factor(x,y)...) ),
			@acrule( (+(~~y)) * (~x::not_division) => +(distribute_factor(x,y)...) )
		]))
end

function term_simplifier()
	Postwalk(
		Chain(
			[
			distribution_rules(),
		 	number_simplifier()
			]
		)
	)
end

function simplify_node()
	Postwalk(
		Chain([
			If(x -> typeof(x) <: Term, term_simplifier()),
			If(x -> x isa Formula, formula_simplifier())
		])
	)
end

function simplify(n :: ASTNode)
	f = Fixpoint(simplify_node())
	return PassThrough(f)(n)
end


function prepare_linearization(n :: ASTNode)
	f = Prewalk(
			Chain([
				If(x -> x isa Atom, atom_simplifier())
				If(x -> x isa Term, term_simplifier())
			])
		)
	return PassThrough(f)(n)
end

function debug_wrapper(rule)
	return x -> begin
		result = rule(x) 
		println("applying rule: ", rule)
		println("input:  ", x)
		println("output: ", result)

		return result
	end
end

function to_dnf(n :: ASTNode)
	f = Fixpoint(Postwalk(
			Chain([
				If(x -> x isa Formula, dnf_simplifier())
			])
		))
	return PassThrough(f)(n)
end
