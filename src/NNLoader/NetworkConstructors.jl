# Abstract implementations

function construct_layer_add(::Type{<:NetworkType}, name,  inputs, outputs, bias :: Vector{Real})
    throw("Not implemented")
end

function construct_layer_sub(::Type{<:NetworkType}, name,  inputs, outputs, bias :: Vector{Real})
    throw("Not implemented")
end

function construct_layer_matmul(::Type{<:NetworkType}, name,  inputs, outputs, weight :: Matrix{Real})
    throw("Not implemented")
end

function construct_layer_gemm(::Type{<:NetworkType}, name,  inputs, outputs, weight :: Matrix{Real}, bias :: Vector{Real})
    throw("Not implemented")
end

function construct_layer_relu(::Type{<:NetworkType}, name,  inputs, outputs, data)
    @assert data == DynamicInput
    throw("Not implemented")
end

function construct_layer_flatten(::Type{<:NetworkType}, name, inputs, outputs, data;axis=1)
    @assert data == DynamicInput
    throw("Not implemented")
end

function construct_network(::Type{<:NetworkType}, inputs, outputs, nodes)
    throw("Not implemented")
end
function construct_layer_reshape(::Type{<:NetworkType}, name, inputs, outputs, data, shape)
    throw("Not implemented")
end
function construct_layer_split(::Type{<:NetworkType}, name, inputs, outputs, data, split; axis=0, num_outputs=nothing)
    throw("Not implemented")
end
function construct_layer_split(::Type{<:NetworkType}, name, inputs, outputs, data; split=nothing, axis=0, num_outputs=nothing)
    throw("Not implemented")
end
function construct_layer_gather(::Type{<:NetworkType}, name, inputs, outputs, data, indices; axis=0)
    throw("Not implemented")
end
function construct_layer_conv(::Type{<:NetworkType}, name, inputs, outputs, data, weights, bias;
    auto_pad="NOTSET", dilations=nothing, group=1, kernel_shape=nothing, pads=nothing, strides=nothing)
    throw("Not implemented")
end
function construct_layer_concat(::Type{<:NetworkType}, name, inputs, outputs, data...;axis=nothing)
    throw("Not implemented")
end
function construct_layer_mul(::Type{<:NetworkType}, name, inputs, outputs, A, B)
    throw("Not implemented")
end

function construct_layer_reducesum(::Type{<:NetworkType}, name, inputs, outputs, data; axes=nothing, keepdims=1, noop_with_empty_axes=0)
    throw("Not implemented")
end

function construct_layer_div(::Type{<:NetworkType}, name, inputs, outputs, A, B)
    throw("Not implemented")
end
# Implementations for VNNLibNetwork

function construct_layer_add(::Type{VNNLibNetworkConstructor}, name, inputs, outputs, a, b)
    return VNNLibAdd{Float64}(name, inputs, outputs, a, b)
end

function construct_layer_sub(::Type{VNNLibNetworkConstructor}, name, inputs, outputs, a, b)
    return VNNLibSub{Float64}(name, inputs, outputs, a, b)
end

function construct_layer_matmul(::Type{VNNLibNetworkConstructor}, name, inputs, outputs, weight, x)
    @assert x == DynamicInput
    return VNNLibDense{Float64}(name, inputs, outputs, weight, nothing)
end

function construct_layer_gemm(::Type{VNNLibNetworkConstructor}, name, inputs, outputs, A, B, C; alpha=1.0, beta=1.0, transA=0, transB=0)
    @assert (transA == 0 && A==DynamicInput) "General Gemm not supported"
    if transB == 0
        weight = alpha.*B
    else
        weight = alpha.*B'
    end
    bias = beta.*C

    return VNNLibDense{Float64}(name, inputs, outputs, weight, bias)
end

function construct_layer_relu(::Type{VNNLibNetworkConstructor}, name, inputs, outputs, data)
    @assert data == DynamicInput
    return VNNLibReLU{Float64}(name, inputs, outputs)
end

function construct_layer_flatten(::Type{VNNLibNetworkConstructor}, name, inputs, outputs, data;axis=1)
    @assert data == DynamicInput
    return VNNLibFlatten{Float64}(name, inputs, outputs, axis)
end

function construct_layer_constant(::Type{VNNLibNetworkConstructor}, name, inputs, outputs; value = nothing)
    @assert !isnothing(value) "Constant layer currently requires a value"
    return VNNLibConstant{Float64}(name, inputs, outputs, value)
end

function construct_layer_reshape(::Type{VNNLibNetworkConstructor}, name, inputs, outputs, data, shape)
    @assert data == DynamicInput
    return VNNLibReshape{Float64}(name, inputs, outputs, shape)
end

function construct_layer_split(net_type::Type{VNNLibNetworkConstructor}, name, inputs, outputs, data, split; axis=0, num_outputs=nothing)
    return construct_layer_split(net_type, name, inputs, outputs, data; split=split, axis=axis, num_outputs=num_outputs)
end

function construct_layer_split(::Type{VNNLibNetworkConstructor}, name, inputs, outputs, data; split=nothing, axis=0, num_outputs=nothing)
    @assert data == DynamicInput
    @assert !isnothing(split) || !isnothing(num_outputs) "Split layer requires either split or num_outputs"
    @assert isnothing(split) || isnothing(num_outputs) "Split layer requires either split or num_outputs, not both"
    # Split into equal parts if num_output is given, otherwise split according to split
    return VNNLibSplit{Float64}(name, inputs, outputs, split, num_outputs, axis)
end

function construct_layer_slice(::Type{VNNLibNetworkConstructor}, name, inputs, outputs, data, starts, ends, axes, steps)
    @assert data == DynamicInput
    return VNNLibSlice{Float64}(name, inputs, outputs, starts, ends, axes, steps)
end

function construct_layer_gather(::Type{VNNLibNetworkConstructor}, name, inputs, outputs, data, indices; axis=0)
    @assert data == DynamicInput
    return VNNLibGather{Float64}(name, inputs, outputs, indices, axis)
end

function construct_layer_conv(::Type{VNNLibNetworkConstructor}, name, inputs, outputs, data, weights, bias;
    auto_pad="NOTSET", dilations=nothing, group=1, kernel_shape=nothing, pads=nothing, strides=nothing)
    return VNNLibConv{Float64}(name, inputs, outputs, weights, bias, auto_pad, dilations, group, kernel_shape, pads, strides)
end

function construct_layer_concat(::Type{VNNLibNetworkConstructor}, name, inputs, outputs, data...;axis=nothing)
    @assert !isnothing(axis) "Concatenation layer requires axis"
    return VNNLibConcat{Float64}(name, inputs, outputs, data, axis)
end

function construct_layer_mul(::Type{VNNLibNetworkConstructor}, name, inputs, outputs, A, B)
    @assert A == DynamicInput
    @assert B == DynamicInput
    return VNNLibMul{Float64}(name, inputs, outputs)
end

function construct_layer_reducesum(::Type{VNNLibNetworkConstructor}, name, inputs, outputs, data; axes=nothing, keepdims=1, noop_with_empty_axes=0)
    @assert data == DynamicInput
    return VNNLibReduceSum{Float64}(name, inputs, outputs, axes, keepdims, noop_with_empty_axes)
end

function construct_layer_div(::Type{VNNLibNetworkConstructor}, name, inputs, outputs, A, B)
    @assert A == DynamicInput
    @assert B == DynamicInput
    return VNNLibDiv{Float64}(name, inputs, outputs)
end

function construct_network(::Type{VNNLibNetworkConstructor}, inputs, outputs, nodes)
    return VNNLibNetwork{Float64}(inputs, outputs, nodes)
end