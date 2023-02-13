is_conjunction(f :: CompositeFormula) = f.head == And && all(map(typeof,f.args) .<: Atom)
to_atomic_composite(f :: CompositeFormula) = CompositeFormula(f.head, convert(Vector{Atom},f.args))

function contains_output(f :: Formula)
    MapReduce(((x,y)->x||y),false,f) do x
        return x isa Variable && x.sort == Output
    end
end