function ast_to_lp(conjunction :: Conjunction)
    num_vars = MapReduce(((x,y)->max(x,y)),0,conjunction.atoms) do x
        return x isa Variable ? x.index : 0
    end
    num_atoms = MapReduce(((x,y)->x+y),0,conjunction.atoms) do x
        if x isa Atom
            return (x.head == Equal ? 2 : 1)
        else
            return 0
        end
    end
    constraint_matrix = zeros(Float64, num_atoms, num_vars)
    constraint_vector = zeros(Float64, num_atoms)
    constraint_index = 1
    for atom in conjunction.atoms
        
    end
end