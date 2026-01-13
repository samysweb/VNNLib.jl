abstract type NetworkType end

abstract type DynamicInput end

struct VNNLibNetworkConstructor <: NetworkType
end

abstract type VNNLibLayer{T<:Real} end

struct VNNLibNetwork{T<:Real}
    inputs :: Vector{String}
    outputs :: Vector{String}
    nodes :: Dict{String, VNNLibLayer{T}}
    input_shapes :: Dict{String, AbstractVector}
    output_shapes :: Dict{String, AbstractVector}
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

struct VNNLibLeakyReLU{T} <: VNNLibLayer{T}
    name::String
    inputs::Vector{String}
    outputs::Vector{String}
    α
    function VNNLibLeakyReLU{T}(name :: String, inputs :: Vector{String}, outputs :: Vector{String}, α) where T<:Real
        return new{T}(name, inputs, outputs, α)
    end
end

struct VNNLibSigmoid{T} <: VNNLibLayer{T}
    name::String
    inputs::Vector{String}
    outputs::Vector{String}
    function VNNLibSigmoid{T}(name :: String, inputs :: Vector{String}, outputs :: Vector{String}) where T<:Real
        return new{T}(name, inputs, outputs)
    end
end

struct VNNLibTanh{T} <: VNNLibLayer{T}
    name::String
    inputs::Vector{String}
    outputs::Vector{String}
    function VNNLibTanh{T}(name :: String, inputs :: Vector{String}, outputs :: Vector{String}) where T<:Real
        return new{T}(name, inputs, outputs)
    end
end

struct VNNLibSoftmax{T} <: VNNLibLayer{T}
    name::String
    inputs::Vector{String}
    outputs::Vector{String}
    axis::Int
    function VNNLibSoftmax{T}(name::String, inputs::Vector{String}, outputs::Vector{String}; axis=-1) where T<:Real
        return new{T}(name, inputs, outputs, axis)
    end
end

struct VNNLibFloor{T} <: VNNLibLayer{T} 
    name::String
    inputs::Vector{String}
    outputs::Vector{String}
    function VNNLibFloor{T}(name::String, inputs::Vector{String}, outputs::Vector{String}) where T<:Real
        return new{T}(name, inputs, outputs)
    end
end

struct VNNLibCos{T} <: VNNLibLayer{T} 
    name::String
    inputs::Vector{String}
    outputs::Vector{String}
    function VNNLibCos{T}(name::String, inputs::Vector{String}, outputs::Vector{String}) where T<:Real
        return new{T}(name, inputs, outputs)
    end
end

struct VNNLibSin{T} <: VNNLibLayer{T} 
    name::String
    inputs::Vector{String}
    outputs::Vector{String}
    function VNNLibSin{T}(name::String, inputs::Vector{String}, outputs::Vector{String}) where T<:Real
        return new{T}(name, inputs, outputs)
    end
end

struct VNNLibSign{T} <: VNNLibLayer{T} 
    name::String
    inputs::Vector{String}
    outputs::Vector{String}
    function VNNLibSign{T}(name::String, inputs::Vector{String}, outputs::Vector{String}) where T<:Real
        return new{T}(name, inputs, outputs)
    end
end

struct VNNLibExp{T} <: VNNLibLayer{T} 
    name::String
    inputs::Vector{String}
    outputs::Vector{String}
    function VNNLibExp{T}(name::String, inputs::Vector{String}, outputs::Vector{String}) where T<:Real
        return new{T}(name, inputs, outputs)
    end
end

struct VNNLibSqrt{T} <: VNNLibLayer{T} 
    name::String
    inputs::Vector{String}
    outputs::Vector{String}
    function VNNLibSqrt{T}(name::String, inputs::Vector{String}, outputs::Vector{String}) where T<:Real
        return new{T}(name, inputs, outputs)
    end
end

struct VNNLibAbs{T} <: VNNLibLayer{T} 
    name::String
    inputs::Vector{String}
    outputs::Vector{String}
    function VNNLibAbs{T}(name::String, inputs::Vector{String}, outputs::Vector{String}) where T<:Real
        return new{T}(name, inputs, outputs)
    end
end

struct VNNLibAcos{T} <: VNNLibLayer{T} 
    name::String
    inputs::Vector{String}
    outputs::Vector{String}
    function VNNLibAcos{T}(name::String, inputs::Vector{String}, outputs::Vector{String}) where T<:Real
        return new{T}(name, inputs, outputs)
    end
end

struct VNNLibHardSigmoid{T} <: VNNLibLayer{T}
    name::String
    inputs::Vector{String}
    outputs::Vector{String}
    alpha
    beta
    function VNNLibHardSigmoid{T}(name :: String, inputs :: Vector{String}, outputs :: Vector{String}, alpha, beta) where T<:Real
        return new{T}(name, inputs, outputs, alpha, beta)
    end
end

struct VNNLibHardSwish{T} <: VNNLibLayer{T}
    name::String
    inputs::Vector{String}
    outputs::Vector{String}
    alpha
    beta
    function VNNLibHardSwish{T}(name :: String, inputs :: Vector{String}, outputs :: Vector{String}, alpha, beta) where T<:Real
        return new{T}(name, inputs, outputs, alpha, beta)
    end
end

struct VNNLibELU{T} <: VNNLibLayer{T}
    name::String
    inputs::Vector{String}
    outputs::Vector{String}
    alpha
    function VNNLibELU{T}(name :: String, inputs :: Vector{String}, outputs :: Vector{String}, alpha) where T<:Real
        return new{T}(name, inputs, outputs, alpha)
    end
end

struct VNNLibGELU{T} <: VNNLibLayer{T}
    name::String
    inputs::Vector{String}
    outputs::Vector{String}
    approximate
    function VNNLibGELU{T}(name :: String, inputs :: Vector{String}, outputs :: Vector{String}, approximate) where T<:Real
        return new{T}(name, inputs, outputs, approximate)
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

struct VNNLibTranspose{T} <: VNNLibLayer{T} 
    name::String
    inputs::Vector{String}
    outputs::Vector{String}
    perm  # permutation of dimensions
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

struct VNNLibSqueeze{T} <: VNNLibLayer{T}
    name::String
    inputs::Vector{String}
    outputs::Vector{String}
    axes
end

struct VNNLibUnsqueeze{T} <: VNNLibLayer{T} 
    name::String
    inputs::Vector{String}
    outputs::Vector{String}
    axes
end

struct VNNLibPad{T} <: VNNLibLayer{T}
    name::String
    inputs::Vector{String}
    outputs::Vector{String}
    mode
    pads
    value
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

struct VNNLibAveragePool{T} <: VNNLibLayer{T}
    name::String
    inputs::Vector{String}
    outputs::Vector{String}
    auto_pad
    ceil_mode
    count_include_pad
    dilations
    kernel_shape
    pads
    strides
end

struct VNNLibMaxPool{T} <: VNNLibLayer{T}
    name::String
    inputs::Vector{String}
    outputs::Vector{String}
    auto_pad
    ceil_mode
    dilations
    kernel_shape
    pads
    storage_order
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

struct VNNLibMulConst{T} <: VNNLibLayer{T} 
    name::String
    inputs::Vector{String}
    outputs::Vector{String}
    c
end

struct VNNLibDiv{T} <: VNNLibLayer{T}
    name::String
    inputs::Vector{String}
    outputs::Vector{String}
end

struct VNNLibDivConst{T} <: VNNLibLayer{T} 
    name::String
    inputs::Vector{String}
    outputs::Vector{String}
    c
end

struct VNNLibNeg{T} <: VNNLibLayer{T} 
    name::String
    inputs::Vector{String}
    outputs::Vector{String}
end

struct VNNLibPow{T} <: VNNLibLayer{T}
    name::String
    inputs::Vector{String}
    outputs::Vector{String}
    exponent
end

struct VNNLibReduceSum{T} <: VNNLibLayer{T}
    name::String
    inputs::Vector{String}
    outputs::Vector{String}
    axes
    keepdims
    noop_with_empty_axes
end

struct VNNLibBatchNorm{T} <: VNNLibLayer{T}
    name::String
    inputs::Vector{String}
    outputs::Vector{String}
    data
    scale
    bias
    input_mean
    input_var
    epsilon
    momentum
    training_mode
end

struct VNNLibConvTranspose{T} <: VNNLibLayer{T}
    name::String
    inputs::Vector{String}
    outputs::Vector{String}
    W
    b
    auto_pad
    dilations
    group
    kernel_shape
    output_padding
    ouput_shape
    pads
    strides
end

struct VNNLibLSTM{T} <: VNNLibLayer{T}
    # has possibility to represent bi-directional LSTM via num_directions=2
    name::String
    inputs::Vector{String}
    outputs::Vector{String}
    # ONNX Inputs
    # TODO: are shapes reversed because of Julia?
    W  # weight tensor for gates num_directions × 4*hidden_size × input_size
    R  # weight tensor for recurrent connections num_directions × 4*hidden_size × hidden_size
    b  # bias tensor for input gate *and* recurrent connections num_directions × 8*hidden_size
    sequence_lens 
    initial_h
    initial_c
    P  # weight tensor for peepholes num_directions × 3*hidden_size
    # ONNX attributes
    activation_alpha
    activation_beta 
    activations
    clip
    direction
    hidden_size
    input_forget
    layout
end

struct VNNLibDropout{T} <: VNNLibLayer{T}
    name::String
    inputs::Vector{String}
    outputs::Vector{String}
    ratio
    traning_mode
end

struct VNNLibUpsample{T} <: VNNLibLayer{T}
    name::String
    inputs::Vector{String}
    outputs::Vector{String}
    scales
    mode
end

struct VNNLibResize{T} <: VNNLibLayer{T}
    name::String
    inputs::Vector{String}
    outputs::Vector{String}
    antialias
    axes
    coordinate_transformation_mode
    cubic_coeff_a
    exclude_outside
    extrapolate_value
    keep_aspect_ratio_policy
    mode
    nearest_mode
    roi
    scales
    sizes
end