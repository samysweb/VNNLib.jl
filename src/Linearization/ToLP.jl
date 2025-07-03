function ast_to_lp(conjunction :: Formula)
    num_vars = MapReduce(((x,y)->max(x,y)),0,conjunction) do x
        return x isa Variable ? x.index[] : (x isa BoundConstraint ? abs(x.var_index) : 0)
    end
    num_atoms = MapReduce(((x,y)->x+y),0,conjunction) do x
        if x isa Atom && !(x isa BoundConstraint)
            return (x.head == Equal ? 2 : 1)
        else
            return 0
        end
    end
    constraint_bounds = zeros(Float64, num_vars, 2)
    constraint_bounds[:,1] .= -Inf
    constraint_bounds[:,2] .= Inf
    constraint_matrix = zeros(Float64, num_atoms, num_vars)
    constraint_vector = zeros(Float64, num_atoms)
    constraint_index = 0
    for atom in conjunction.args
        contains_output = MapReduce(((x,y)->x||y),false,atom) do x
            return x isa Variable && x.sort == Output
        end
        if contains_output
            continue
        end
        constraint_index = add_constraint(atom, constraint_index, constraint_bounds, constraint_matrix, constraint_vector)
    end
    num_input_constraints = constraint_index
    for atom in conjunction.args
        contains_output = MapReduce(((x,y)->x||y),false,atom) do x
            return x isa Variable && x.sort == Output
        end
        if !contains_output
            continue
        end
        constraint_index = add_constraint(atom, constraint_index, constraint_bounds, constraint_matrix, constraint_vector)
    end
    return constraint_bounds, constraint_matrix, constraint_vector, num_input_constraints
end

function add_constraint(atom :: BoundConstraint,  constraint_index, constraint_bounds, constraint_matrix, constraint_vector)
    if atom.var_index < 0
        # -v <= b <-> -b <= v
        constraint_bounds[-atom.var_index,1] = -atom.bound
    else
        constraint_bounds[atom.var_index,2] = atom.bound
    end
    return constraint_index
end

function add_constraint(atom :: ComparisonFormula,  constraint_index, constraint_bounds, constraint_matrix, constraint_vector)
    @assert atom.right isa Constant && iszero(atom.right.value)
    # TODO(steuber): take Less into account
    @assert atom.left isa ArithmeticTerm && atom.left.head == Addition
    constraint_index+=1
    for arg in atom.left.args
        if arg isa Variable
            constraint_matrix[constraint_index,arg.index[]] += 1.0
        elseif arg isa ArithmeticTerm && arg.head == Multiplication
            @assert arg.args[1] isa Constant && arg.args[2] isa Variable
            constraint_matrix[constraint_index,arg.args[2].index[]] += round_minimize(arg.args[1].value)
        elseif arg isa Constant
            constraint_vector[constraint_index] += round_maximize(-arg.value)
        else
            error("Unknown atom type: $(typeof(atom))")
        end
    end
    if atom.head == Equal
        constraint_index+=1
        constraint_matrix[constraint_index,:] .= -constraint_matrix[constraint_index-1,:]
        constraint_vector[constraint_index] .= -constraint_vector[constraint_index-1]
    end
    return constraint_index
end