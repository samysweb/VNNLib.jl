abstract type NetworkType end

abstract type DynamicInput end

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
    a
    b
end


struct VNNLibSub{T} <: VNNLibLayer{T}
    name::String
    inputs::Vector{String}
    outputs::Vector{String}
    a
    b
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
    axis
    function VNNLibFlatten{T}(name :: String, inputs :: Vector{String}, outputs :: Vector{String}, axis) where T<:Real
        return new{T}(name, inputs, outputs, axis)
    end
end

struct VNNLibConstant{T} <: VNNLibLayer{T}
    name::String
    inputs::Vector{String}
    outputs::Vector{String}
    constant
end

struct VNNLibReshape{T} <: VNNLibLayer{T}
    name::String
    inputs::Vector{String}
    outputs::Vector{String}
    shape
end

struct VNNLibSplit{T} <: VNNLibLayer{T}
    name::String
    inputs::Vector{String}
    outputs::Vector{String}
    split
    num_outputs
    axis
end

struct VNNLibSlice{T} <: VNNLibLayer{T}
    name::String
    inputs::Vector{String}
    outputs::Vector{String}
    starts
    ends
    axes
    steps
end

struct VNNLibGather{T} <: VNNLibLayer{T}
    name::String
    inputs::Vector{String}
    outputs::Vector{String}
    indices
    axes
end

struct VNNLibConv{T} <: VNNLibLayer{T}
    name::String
    inputs::Vector{String}
    outputs::Vector{String}
    W
    b
    auto_pad
    dilations
    group
    kernel_shape
    pads
    strides
end

struct VNNLibConcat{T} <: VNNLibLayer{T}
    name::String
    inputs::Vector{String}
    outputs::Vector{String}
    # Might contain DynamicInput at certain positions!
    data
    axis
end

struct VNNLibMul{T} <: VNNLibLayer{T}
    name::String
    inputs::Vector{String}
    outputs::Vector{String}
end

struct VNNLibDiv{T} <: VNNLibLayer{T}
    name::String
    inputs::Vector{String}
    outputs::Vector{String}
end

struct VNNLibReduceSum{T} <: VNNLibLayer{T}
    name::String
    inputs::Vector{String}
    outputs::Vector{String}
    axes
    keepdims
    noop_with_empty_axes
end