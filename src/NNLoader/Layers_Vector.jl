function (N::VNNLibNetwork)(x :: Dict)
    values = Dict()
    for i in N.inputs
        values[i] = x[i]
    end
    to_compute = deepcopy(N.outputs)
    while length(to_compute) > 0
        L = N.nodes[pop!(to_compute)]
        if !all([haskey(values, i) for i in L.inputs])
            push!(to_compute, L.name)
            append!(to_compute, L.inputs)
            continue
        end
        values[L.name] = L([values[i] for i in L.inputs]...)
    end
    return [i => values[i] for i in N.outputs]
end
function (L::VNNLibAdd{T})(x) where T<:Real
    return x .+ L.b
end

function (L::VNNLibDense{T})(x) where T<:Real
    res = L.W * x
    if !isnothing(L.b)
        res .+= L.b
    end
    return res
end

function (L::VNNLibReLU{T})(x) where T<:Real
    return max.(x,0.0)
end

function (L::VNNLibFlatten{T})(x) where T<:Real
    @assert L.axis == 1
    return x[:]
end