abstract type Layer end

struct Network
    layers::Vector{Layer}
end

struct Dense <: Layer
    W::Matrix{Float64}
    b::Vector{Float64}
end

struct ReLU <: Layer end