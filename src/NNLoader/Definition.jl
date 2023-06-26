abstract type NetworkType end

struct VNNLibNetworkConstructor <: NetworkType
end

abstract type VNNLibLayer{T<:Real} end

struct VNNLibNetwork{T<:Real}
    inputs :: Vector{String}
    outputs :: Vector{String}
    nodes :: Dict{String, VNNLibLayer{T}}
end

struct VNNLibAdd{T} <: VNNLibLayer{T}
    name::String
    inputs::Vector{String}
    outputs::Vector{String}
    b
    function VNNLibAdd{T}(name :: String, inputs :: Vector{String}, outputs :: Vector{String}, bias) where T<:Real
        return new{T}(name, inputs, outputs, bias)
    end
end

struct VNNLibDense{T} <: VNNLibLayer{T}
    name::String
    inputs::Vector{String}
    outputs::Vector{String}
    W
    b
    function VNNLibDense{T}(name :: String, inputs :: Vector{String}, outputs :: Vector{String}, W, b) where T<:Real
        return new{T}(name, inputs, outputs, W, b)
    end
end

struct VNNLibReLU{T} <: VNNLibLayer{T}
    name::String
    inputs::Vector{String}
    outputs::Vector{String}
    function VNNLibReLU{T}(name :: String, inputs :: Vector{String}, outputs :: Vector{String}) where T<:Real
        return new{T}(name, inputs, outputs)
    end
end

struct VNNLibFlatten{T} <: VNNLibLayer{T}
    name::String
    inputs::Vector{String}
    outputs::Vector{String}
    function VNNLibFlatten{T}(name :: String, inputs :: Vector{String}, outputs :: Vector{String}) where T<:Real
        return new{T}(name, inputs, outputs)
    end
end