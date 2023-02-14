function (N::Network)(x :: Vector{Float64})
    for L in N.layers
        x = L(x)
    end
    return x
end

function (L::Dense)(x :: Vector{Float64})
    return L.W * x .+ L.b
end

function (L::ReLU)(x :: Vector{Float64})
    return max.(x,0.0)
end