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

function construct_layer_relu(::Type{<:NetworkType}, name,  inputs, outputs)
    throw("Not implemented")
end

function construct_layer_flatten(::Type{<:NetworkType}, name,  inputs, outputs)
    throw("Not implemented")
end

function construct_network(::Type{<:NetworkType}, inputs, outputs, nodes)
    throw("Not implemented")
end

# Implementations for VNNLibNetwork

function construct_layer_add(::Type{VNNLibNetworkConstructor}, name, inputs, outputs, bias)
    return VNNLibAdd{Float64}(name, inputs, outputs, bias)
end

function construct_layer_sub(::Type{VNNLibNetworkConstructor}, name, inputs, outputs, bias)
    return VNNLibAdd{Float64}(name, inputs, outputs, -bias)
end

function construct_layer_matmul(::Type{VNNLibNetworkConstructor}, name, inputs, outputs, weight)
    return VNNLibDense{Float64}(name, inputs, outputs, weight, zeros(size(weight, 1)))
end

function construct_layer_gemm(::Type{VNNLibNetworkConstructor}, name, inputs, outputs, weight, bias)
    return VNNLibDense{Float64}(name, inputs, outputs, weight, bias)
end

function construct_layer_relu(::Type{VNNLibNetworkConstructor}, name, inputs, outputs)
    return VNNLibReLU{Float64}(name, inputs, outputs)
end

function construct_layer_flatten(::Type{VNNLibNetworkConstructor}, name, inputs, outputs)
    return VNNLibFlatten{Float64}(name, inputs, outputs)
end

function construct_network(::Type{VNNLibNetworkConstructor}, inputs, outputs, nodes)
    return VNNLibNetwork{Float64}(inputs, outputs, nodes)
end