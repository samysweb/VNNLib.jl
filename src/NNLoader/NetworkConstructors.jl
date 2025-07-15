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

function construct_layer_leaky_relu(::Type{<:NetworkType}, name,  inputs, outputs, data; alpha=0.01)
    @assert data == DynamicInput
    throw("Not implemented")
end

function construct_layer_sigmoid(::Type{<:NetworkType}, name, inputs, outputs, data)
    @assert data == DynamicInput
    throw("Not implemented")    
end

function construct_layer_tanh(::Type{<:NetworkType}, name, inputs, outputs, data)
    @assert data == DynamicInput
    throw("Not implemented")
end

function construct_layer_sign(::Type{<:NetworkType}, name, inputs, outputs, data)
    @assert data == DynamicInput
    throw("Not implemented")    
end

function construct_layer_softmax(::Type{<:NetworkType}, name, inputs, outputs, data; axis=-1)
    @assert data == DynamicInput
    throw("not implemented")
end

function construct_layer_floor(::Type{<:NetworkType}, name, inputs, outputs, data)
    @assert data == DynamicInput
    throw("not implemented")
end

function construct_layer_cos(::Type{<:NetworkType}, name, inputs, outputs, data)
    @assert data == DynamicInput
    throw("not implemented")
end

function construct_layer_sin(::Type{<:NetworkType}, name, inputs, outputs, data)
    @assert data == DynamicInput
    throw("not implemented")
end

function construct_layer_sqrt(::Type{<:NetworkType}, name, inputs, outputs, data)
    @assert data == DynamicInput
    throw("not implemented")
end

function construct_layer_exp(::Type{<:NetworkType}, name, inputs, outputs, data)
    @assert data == DynamicInput
    throw("not implemented")
end

function construct_layer_elu(::Type{<:NetworkType}, name, inputs, outputs, data; alpha=1.0)
    @assert data == DynamicInput
    throw("not implemented")
end

function construct_layer_gelu(::Type{<:NetworkType}, name, inputs, outputs, data; approximate=nothing)
    @assert data == DynamicInput
    throw("not implemented")
end

function construct_layer_abs(::Type{<:NetworkType}, name, inputs, outputs, data)
    @assert data == DynamicInput
    throw("not implemented")
end

function construct_layer_acos(::Type{<:NetworkType}, name, inputs, outputs, data)
    @assert data == DynamicInput
    throw("not implemented")
end

function construct_layer_hard_sigmoid(::Type{<:NetworkType}, name, inputs, outputs, data; alpha=0.2, beta=0.5)
    @assert data == DynamicInput
    throw("not implemented")
end

function construct_layer_hard_swish(::Type{<:NetworkType}, name, inputs, outputs, data; alpha=1/6, beta=0.5)
    @assert data == DynamicInput
    throw("not implemented")
end

function construct_layer_flatten(::Type{<:NetworkType}, name, inputs, outputs, data;axis=1)
    @assert data == DynamicInput
    throw("Not implemented")
end

function construct_network(::Type{<:NetworkType}, inputs, outputs, nodes, input_shape, output_shape)
    throw("Not implemented")
end
function construct_layer_reshape(::Type{<:NetworkType}, name, inputs, outputs, data, shape)
    throw("Not implemented")
end
function construct_layer_transpose(::Type{<:NetworkType}, name, inputs, outputs, data; perm=nothing)
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
function construct_layer_squeeze(::Type{<:NetworkType}, name, inputs, outputs, data, axes)
    throw("Not implemented")
end
function construct_layer_unsqueeze(::Type{<:NetworkType}, name, inputs, outputs, data, axes)
    throw("not implemented")
end
function construct_layer_pad(::Type{<:NetworkType}, name, inputs, outputs, data; mode="constant", pads=nothing, value=0.)
    @assert !isnothing(pads) "pads is a required argument!"
    throw("Not implemented")
end
function construct_layer_conv(::Type{<:NetworkType}, name, inputs, outputs, data, weights, bias;
    auto_pad="NOTSET", dilations=nothing, group=1, kernel_shape=nothing, pads=nothing, strides=nothing)
    throw("Not implemented")
end
function construct_layer_average_pool(::Type{<:NetworkType}, name, inputs, outputs, data;
    auto_pad="NOTSET", ceil_mode=0, count_include_pad=0, dilations=nothing, kernel_shape=nothing, pads=nothing, strides=nothing)
    throw("Not implemented")
end
function construct_layer_max_pool(::Type{<:NetworkType}, name, inputs, outputs, data;
    auto_pad="NOTSET", ceil_mode=0, dilations=nothing, kernel_shape=nothing, pads=nothing, storage_order=0, strides=nothing)
    throw("Not implemented")
end
function construct_layer_concat(::Type{<:NetworkType}, name, inputs, outputs, data...;axis=nothing)
    throw("Not implemented")
end
function construct_layer_mul(::Type{<:NetworkType}, name, inputs, outputs, A, B)
    throw("Not implemented")
end
function construct_layer_neg(::Type{<:NetworkType}, name, inputs, outputs, data)
    throw("Not implemented")
end
function construct_layer_reducesum(::Type{<:NetworkType}, name, inputs, outputs, data; axes=nothing, keepdims=1, noop_with_empty_axes=0)
    throw("Not implemented")
end

function construct_layer_div(::Type{<:NetworkType}, name, inputs, outputs, A, B)
    throw("Not implemented")
end

function construct_layer_pow(::Type{<:NetworkType}, name, inputs, outputs, A, B)
    throw("Not implemented")
end

function construct_layer_batch_normalization(::Type{<:NetworkType}, name, inputs, outputs, X, scale, B, input_mean, input_var; epsilon=1e-5, momentum=0.9, training_mode=0)
    throw("Not implemented")
end

function construct_layer_conv_transpose(::Type{<:NetworkType}, name, inputs, outputs, data, weights, bias;
    auto_pad="NOTSET", dilations=nothing, group=1, kernel_shape=nothing, output_padding=nothing, output_shape=nothing, pads=nothing, strides=nothing)
    throw("Not implemented")
end

function construct_layer_lstm(::Type{<:NetworkType}, name, inputs, outputs, data, W_ih, W_hh, bias=nothing, sequence_lens=nothing, 
    initial_h=nothing, initial_c=nothing, P=nothing; activation_alpha=nothing, activation_beta=nothing, activations=nothing, clip=nothing, 
    direction="forward", hidden_size=-1, input_forget=0, layout=0)
    throw("Not implemented")
end

"""
We use version 13 of the ONNX standard, since this is the vgg16 benchmark is the only one with dropout layer and is uses this version.
"""
function construct_layer_dropout(::Type{<:NetworkType}, name, inputs, outputs, data, ratio=0.5, training_mode=false)
    @assert data == DynamicInput "Expected DynamicInput for data, but got $data"
    throw("Not implemented")
end

function construct_layer_upsample(::Type{<:NetworkType}, name, inputs, outputs, data, scales; mode="nearest")
    @assert data == DynamicInput "Expected DynamicInput for data, but got $data"
    println("Constructing Upsample: data = $data, scales = $scales")
    throw("Not implemented")
end

function construct_layer_resize(::Type{<:NetworkType}, name, inputs, outputs, data, roi, scales, sizes; antialias=0, axes=nothing, 
    coordinate_transformation_mode="half_pixel", cubic_coeff_a=-0.75, exclude_outside=0, extrapolation_value=0., keep_aspect_ratio_policy="stretch",
    mode="nearest", nearest_mode="round_prefer_floor")
    @assert data == DynamicInput "Expected Dynamic input for data, but got $data"
    throw("Not implemented")
end

# Implementations for VNNLibNetwork

function construct_layer_add(::Type{VNNLibNetworkConstructor}, name, inputs, outputs, a, b)
    return VNNLibAdd{Float64}(name, inputs, outputs, a, b)
end

function construct_layer_sub(::Type{VNNLibNetworkConstructor}, name, inputs, outputs, a, b)
    return VNNLibSub{Float64}(name, inputs, outputs, a, b)
end

function construct_layer_matmul(::Type{VNNLibNetworkConstructor}, name, inputs, outputs, weight, x::Type{DynamicInput})
    @assert x == DynamicInput
    println("typeof(weight) = ", typeof(weight))
    println("weight <: Array{N} ? ", typeof(weight) <: AbstractArray{<:Number})
    return VNNLibDense{Float64}(name, inputs, outputs, weight, nothing)
end

function construct_layer_matmul(::Type{VNNLibNetworkConstructor}, name, inputs, outputs, x::Type{DynamicInput}, weight)
    @assert x == DynamicInput
    println("typeof(weight) = ", typeof(weight))
    println("weight <: Array{N} ? ", typeof(weight) <: AbstractArray{<:Number})
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

function construct_layer_leaky_relu(::Type{VNNLibNetworkConstructor}, name,  inputs, outputs, data; alpha=0.01)
    @assert data == DynamicInput
    return VNNLibLeakyReLU{Float64}(name, inputs, outputs, alpha)
end

function construct_layer_sigmoid(::Type{VNNLibNetworkConstructor}, name, inputs, outputs, data)
    @assert data == DynamicInput
    return VNNLibSigmoid{Float64}(name, inputs, outputs)  
end

function construct_layer_tanh(::Type{VNNLibNetworkConstructor}, name, inputs, outputs, data)
    @assert data == DynamicInput
    return VNNLibTanh{Float64}(name, inputs, outputs)  
end

function construct_layer_softmax(::Type{VNNLibNetworkConstructor}, name, inputs, outputs, data; axis=-1)
    @assert data == DynamicInput
    return VNNLibSoftmax{Float64}(name, inputs, outputs, axis=axis)
end

function construct_layer_floor(::Type{VNNLibNetworkConstructor}, name, inputs, outputs, data)
    @assert data == DynamicInput
    return VNNLibFloor{Float64}(name, inputs, outputs)
end

function construct_layer_cos(::Type{VNNLibNetworkConstructor}, name, inputs, outputs, data)
    @assert data == DynamicInput
    return VNNLibCos{Float64}(name, inputs, outputs)
end

function construct_layer_sin(::Type{VNNLibNetworkConstructor}, name, inputs, outputs, data)
    @assert data == DynamicInput
    return VNNLibSin{Float64}(name, inputs, outputs)
end

function construct_layer_sign(::Type{VNNLibNetworkConstructor}, name, inputs, outputs, data)
    @assert data == DynamicInput
    return VNNLibSign{Float64}(name, inputs, outputs)
end

function construct_layer_sqrt(::Type{VNNLibNetworkConstructor}, name, inputs, outputs, data)
    @assert data == DynamicInput
    return VNNLibSqrt{Float64}(name, inputs, outputs)
end

function construct_layer_exp(::Type{VNNLibNetworkConstructor}, name, inputs, outputs, data)
    @assert data == DynamicInput
    return VNNLibExp{Float64}(name, inputs, outputs)
end

function construct_layer_elu(::Type{VNNLibNetworkConstructor}, name, inputs, outputs, data; alpha=1.0)
    @assert data == DynamicInput
    return VNNLibELU{Float64}(name, inputs, outputs, alpha)
end

function construct_layer_gelu(::Type{VNNLibNetworkConstructor}, name, inputs, outputs, data; approximate=nothing)
    @assert data == DynamicInput
    return VNNLibGELU{Float64}(name, inputs, outputs, approximate)
end

function construct_layer_abs(::Type{VNNLibNetworkConstructor}, name, inputs, outputs, data)
    @assert data == DynamicInput
    return VNNLibAbs{Float64}(name, inputs, outputs)
end

function construct_layer_acos(::Type{VNNLibNetworkConstructor}, name, inputs, outputs, data)
    @assert data == DynamicInput
    return VNNLibAcos{Float64}(name, inputs, outputs)
end

function construct_layer_hard_sigmoid(::Type{VNNLibNetworkConstructor}, name, inputs, outputs, data; alpha=0.2, beta=0.5)
    @assert data == DynamicInput
    return VNNLibHardSigmoid{Float64}(name, inputs, outputs, alpha, beta)
end

function construct_layer_hard_swish(::Type{VNNLibNetworkConstructor}, name, inputs, outputs, data; alpha=1/6, beta=0.5)
    @assert data == DynamicInput
    return VNNLibHardSwish{Float64}(name, inputs, outputs, alpha, beta)
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

function construct_layer_transpose(::Type{VNNLibNetworkConstructor}, name, inputs, outputs, data; perm=nothing)
    @assert data == DynamicInput
    return VNNLibTranspose{Float64}(name, inputs, outputs, perm)
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

function construct_layer_squeeze(::Type{VNNLibNetworkConstructor}, name, inputs, outputs, data, axes)
    @assert data == DynamicInput
    return VNNLibSqueeze{Float64}(name, inputs, outputs, axes)
end

function construct_layer_unsqueeze(::Type{VNNLibNetworkConstructor}, name, inputs, outputs, data, axes)
    @assert data == DynamicInput
    return VNNLibUnsqueeze{Float64}(name, inputs, outputs, axes)    
end

function construct_layer_pad(::Type{VNNLibNetworkConstructor}, name, inputs, outputs, data; mode="constant", pads=nothing, value=0.)
    @assert data == DynamicInput
    @assert !isnothing(pads) "pads required!"
    return VNNLibPad{Float64}(name, inputs, outputs, mode, pads, value)
end

function construct_layer_conv(::Type{VNNLibNetworkConstructor}, name, inputs, outputs, data, weights, bias=0.;
    auto_pad="NOTSET", dilations=nothing, group=1, kernel_shape=nothing, pads=nothing, strides=nothing)
    return VNNLibConv{Float64}(name, inputs, outputs, weights, bias, auto_pad, dilations, group, kernel_shape, pads, strides)
end

function construct_layer_average_pool(::Type{VNNLibNetworkConstructor}, name, inputs, outputs, data;
    auto_pad="NOTSET", ceil_mode=0, count_include_pad=0, dilations=nothing, kernel_shape=nothing, pads=nothing, strides=nothing)
    return VNNLibAveragePool{Float64}(name, inputs, outputs, auto_pad, ceil_mode, count_include_pad, dilations, kernel_shape, pads, strides)
end

function construct_layer_max_pool(::Type{VNNLibNetworkConstructor}, name, inputs, outputs, data;
    auto_pad="NOTSET", ceil_mode=0, dilations=nothing, kernel_shape=nothing, pads=nothing, storage_order=0, strides=nothing)
    return VNNLibMaxPool{Float64}(name, inputs, outputs, auto_pad, ceil_mode, dilations, kernel_shape, pads, storage_order, strides)
end

function construct_layer_concat(::Type{VNNLibNetworkConstructor}, name, inputs, outputs, data...;axis=nothing)
    @assert !isnothing(axis) "Concatenation layer requires axis"
    return VNNLibConcat{Float64}(name, inputs, outputs, data, axis)
end

function construct_layer_mul(::Type{VNNLibNetworkConstructor}, name, inputs, outputs, A::Type{DynamicInput}, B::Type{DynamicInput})
    @assert A == DynamicInput
    @assert B == DynamicInput
    return VNNLibMul{Float64}(name, inputs, outputs)
end

function construct_layer_mul(::Type{VNNLibNetworkConstructor}, name, inputs, outputs, A::Type{DynamicInput}, B)
    println("node $name: mul constant")
    return VNNLibMulConst{Float64}(name, inputs, outputs, B)
end

function construct_layer_mul(::Type{VNNLibNetworkConstructor}, name, inputs, outputs, A, B::Type{DynamicInput})
    println("node $name: mul constant")
    return VNNLibMulConst{Float64}(name, inputs, outputs, A)
end

function construct_layer_neg(::Type{VNNLibNetworkConstructor}, name, inputs, outputs, data)
    return VNNLibNeg{Float64}(name, inputs, outputs)
end

function construct_layer_reducesum(::Type{VNNLibNetworkConstructor}, name, inputs, outputs, data; axes=nothing, keepdims=1, noop_with_empty_axes=0)
    @assert data == DynamicInput
    return VNNLibReduceSum{Float64}(name, inputs, outputs, axes, keepdims, noop_with_empty_axes)
end

function construct_layer_div(::Type{VNNLibNetworkConstructor}, name, inputs, outputs, A::Type{DynamicInput}, B::Type{DynamicInput})
    @assert A == DynamicInput
    @assert B == DynamicInput
    return VNNLibDiv{Float64}(name, inputs, outputs)
end

function construct_layer_div(::Type{VNNLibNetworkConstructor}, name, inputs, outputs, A::Type{DynamicInput}, B)
    println("node $name: div const")
    return VNNLibDivConst{Float64}(name, inputs, outputs, B)
end

function construct_layer_div(::Type{VNNLibNetworkConstructor}, name, inputs, outputs, A, B::Type{DynamicInput})
    println("node $name: div const")
    return VNNLibDivConst{Float64}(name, inputs, outputs, A)
end

function construct_layer_pow(::Type{VNNLibNetworkConstructor}, name, inputs, outputs, A, B)
    @assert A == DynamicInput
    return VNNLibPow{Float64}(name, inputs, outputs, B)
end


function construct_layer_batch_normalization(::Type{VNNLibNetworkConstructor}, name, inputs, outputs, X, scale, B, input_mean, input_var; epsilon=1e-5, momentum=0.9, training_mode=0)
    @assert X == DynamicInput 
    return VNNLibBatchNorm{Float64}(name, inputs, outputs, X, scale, B, input_mean, input_var, epsilon, momentum, training_mode)
end

function construct_layer_conv_transpose(::Type{VNNLibNetworkConstructor}, name, inputs, outputs, data, weights, bias;
    auto_pad="NOTSET", dilations=nothing, group=1, kernel_shape=nothing, output_padding=nothing, output_shape=nothing, pads=nothing, strides=nothing)
    println("constructing convT!")
    @assert data == DynamicInput "Expected DynamicInput for data, but got $data"
    return VNNLibConvTranspose{Float64}(name, inputs, outputs, weights, bias, auto_pad, dilations, group, kernel_shape, output_padding, output_shape, pads, strides)
end

function construct_layer_lstm(::Type{VNNLibNetworkConstructor}, name, inputs, outputs, data, W_ih, W_hh, bias=nothing, sequence_lens=nothing, 
    initial_h=nothing, initial_c=nothing, P=nothing; activation_alpha=nothing, activation_beta=nothing, activations=nothing, clip=nothing, 
    direction="forward", hidden_size=-1, input_forget=0, layout=0)
    @assert data == DynamicInput "Expected DynamicInput for data, but got $data"
    # although hidden_size is a keyword argument, it has no default value and therefore must be set!
    @assert hidden_size >= 0 "hidden_size must be set!"
    return VNNLibLSTM{Float64}(name, inputs, outputs, W_ih, W_hh, bias, sequence_lens, initial_h, initial_c, P, activation_alpha, activation_beta, activations, 
                        clip, direction, hidden_size, input_forget, layout)
end

function construct_layer_dropout(::Type{VNNLibNetworkConstructor}, name, inputs, outputs, data, ratio=0.5, training_mode=false)
    @assert data == DynamicInput "Expected DynamicInput for data, but got $data"
    return VNNLibDropout{Float64}(name, inputs, outputs, ratio, training_mode)
end

function construct_layer_upsample(::Type{VNNLibNetworkConstructor}, name, inputs, outputs, data, scales; mode="nearest")
    @assert data == DynamicInput "Expected DynamicInput for data, but got $data"
    println("Constructing Upsample: data = $data, scales = $scales")
    return VNNLibUpsample{Float64}(name, inputs, outputs, scales, mode)
end

function construct_layer_resize(::Type{VNNLibNetworkConstructor}, name, inputs, outputs, data, roi=nothing, scales=nothing, sizes=nothing; antialias=0, axes=nothing, 
    coordinate_transformation_mode="half_pixel", cubic_coeff_a=-0.75, exclude_outside=0, extrapolation_value=0., keep_aspect_ratio_policy="stretch",
    mode="nearest", nearest_mode="round_prefer_floor")
    @assert data == DynamicInput "Expected Dynamic input for data, but got $data"
    @assert !isnothing(scales) || !isnothing(sizes) "Resize layer requires either scales or sizes"
    @assert isnothing(scales) || isnothing(sizes) "Resize layer requires either scales or sizes, not both"
    println("Constructing Resize: data = $data, roi = $roi, scales = $scales, sizes = $sizes")
    return VNNLibResize{Float64}(name, inputs, outputs, antialias, axes, coordinate_transformation_mode, cubic_coeff_a, exclude_outside, 
                                 extrapolation_value, keep_aspect_ratio_policy, mode, nearest_mode, roi, scales, sizes)
end

function construct_network(::Type{VNNLibNetworkConstructor}, inputs, outputs, nodes, input_shape, output_shape)
    return VNNLibNetwork{Float64}(inputs, outputs, nodes, input_shape, output_shape)
end