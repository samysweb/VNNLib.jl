
##############################################################
## Dense layer and corresponding ONNX nodes Matmul and Gemm ##
##############################################################

struct ONNXLinear{S,VS<:AbstractVector{S},D<:Dense} <: Node{S}
    # vector of identifiers for each input
    inputs::VS
    # vector of identifiers for each output
    outputs::VS
    name::S
    dense::D
    transpose::Bool
end

# TODO(steuber): Transpose for single dimensional input shape seems to be wrong here?
# TODO(pkern): We should only translate to Dense, when a linear layer is really intended.
#              For Linear layers, we want the last dimension to be the batch dimension, so transpose doesn't make sense.
#              If W has shape (m, n) and x has shape (n, b), then we want Wx with shape (m, b) (and crucially, (m,n) * (n,1) for every dim)
#              For multiplication of two matrices (which MatMul and Gemm can also represent), we need to transpose because 
#              Flux uses WHCN format.
onnx_node_to_flux_layer(node::ONNXLinear) = x -> (node.transpose && length(size(x)) > 1 ) ? node.dense(x')' : node.dense(x)

islinear(node::ONNXLinear) = true

function ONNXLinear(inputs::AbstractVector{S}, outputs::AbstractVector{S}, name::S, W::AbstractMatrix{WN}, b::AbstractVector{BN}; transpose=false, double_precision=false) where {S,WN<:Number,BN<:Number}
    if double_precision
        W = Float64.(W)
        b = Float64.(b)
    end

    dense = Dense(W, b)

    return ONNXLinear(inputs, outputs, name, dense, transpose)
end

function NNL.construct_layer_matmul(::Type{OnnxType}, name, inputs, outputs, weight, x::Type{NNL.DynamicInput})
    VERBOSE_ONNX[] > 0 && println("Constructing Matmul node: $name with inputs $(inputs) and outputs $(outputs)")

    # since Flux is WHCN, we need to transpose the weight (and then also transpose the input vector and transpose the result of Wx' back 
    # to be consistent.
    return ONNXLinear(inputs, outputs, name, weight', zero(weight[1,:]), transpose=true, double_precision=DOUBLE_PRECISION[])
end

# TODO: is Type{NNL.DynamicInput} what we really want here?
function NNL.construct_layer_matmul(::Type{OnnxType}, name, inputs, outputs, x::Type{NNL.DynamicInput}, weight)
    VERBOSE_ONNX[] > 0 && println("Constructing Matmul node: $name with inputs $(inputs) and outputs $(outputs)")
    return ONNXLinear(inputs, outputs, name, weight, zero(weight[:,1]), transpose=false, double_precision=DOUBLE_PRECISION[])
end

function NNL.construct_layer_gemm(::Type{OnnxType}, name, inputs, outputs, A, B, C; alpha=1., beta=1., transA=0, transB=0)
    @assert (transA == 0 && A == NNL.DynamicInput) "General Gemm not supported"
    VERBOSE_ONNX[] > 0 && println("Constructing Gemm node: $name with inputs $(inputs) and outputs $(outputs) ($(typeof(A)), $(typeof(B)), $(typeof(C)))")
    if transB == 0
        W = alpha .* B
    else
        W = alpha .* B'
    end

    b = beta .* C

    return ONNXLinear(inputs, outputs, name, W, b, double_precision=DOUBLE_PRECISION[])
end



#######################################################################
## Add with variable and different configurations of constant inputs ##
#######################################################################


struct ONNXAddConst{S,VS<:AbstractVector{S},VN} <: Node{S}
    inputs::VS
    outputs::VS
    name::S
    c::VN
end

onnx_node_to_flux_layer(node::ONNXAddConst) = x -> x .+ node.c

islinear(node::ONNXAddConst) = true

function NNL.construct_layer_add(::Type{OnnxType}, name, inputs, outputs, a::Type{NNL.DynamicInput}, b)
    VERBOSE_ONNX[] > 0 && println("Constructing ONNXAddConst node: $name with inputs $(inputs) and outputs $(outputs)")
    b = DOUBLE_PRECISION[] ? Float64.(b) : b
    ONNXAddConst(inputs, outputs, name, b)
end

function NNL.construct_layer_add(::Type{OnnxType}, name, inputs, outputs, a, b::Type{NNL.DynamicInput})
    VERBOSE_ONNX[] > 0 && println("Constructing ONNXAddConst node: $name with inputs $(inputs) and outputs $(outputs)")
    a = DOUBLE_PRECISION[] ? Float64.(a) : a
    ONNXAddConst(inputs, outputs, name, a)
end

struct ONNXAdd{S,VS<:AbstractVector{S}} <: Node{S}
    inputs::VS
    outputs::VS
    name::S
end

onnx_node_to_flux_layer(node::ONNXAdd) = (x, y) -> x .+ y

islinear(node::ONNXAdd) = true

function NNL.construct_layer_add(::Type{OnnxType}, name, inputs, outputs, a::Type{NNL.DynamicInput}, b::Type{NNL.DynamicInput})
    # need to specify that both inputs are dynamic inputs to avoid ambiguity with ONNXAddConst constructors
    VERBOSE_ONNX[] > 0 && println("Constructing ONNXAdd node: $name with inputs $(inputs) and outputs $(outputs)")
    ONNXAdd(inputs, outputs, name)
end



############################################################################
## Subtract with variable and different configurations of constant inputs ##
############################################################################


struct ONNXSubConst{S,VS<:AbstractVector{S},C} <: Node{S}
    inputs::VS
    outputs::VS
    name::S
    c::C
    left::Bool  # if variable is on the left side of the subtraction
end

onnx_node_to_flux_layer(node::ONNXSubConst) = x -> ifelse(node.left, x .- node.c, node.c .- x)

islinear(node::ONNXSubConst) = true

function NNL.construct_layer_sub(::Type{OnnxType}, name, inputs, outputs, a::Type{NNL.DynamicInput}, b)
    VERBOSE_ONNX[] > 0 && println("Constructing ONNXSubConst node: $name with inputs $(inputs) and outputs $(outputs)")
    b = DOUBLE_PRECISION[] ? Float64.(b) : b
    ONNXSubConst(inputs, outputs, name, b, true)
end

function NNL.construct_layer_sub(::Type{OnnxType}, name, inputs, outputs, a, b::Type{NNL.DynamicInput})
    VERBOSE_ONNX[] > 0 && println("Constructing ONNXSubConst node: $name with inputs $(inputs) and outputs $(outputs)")
    a = DOUBLE_PRECISION[] ? Float64.(a) : a
    ONNXSubConst(inputs, outputs, name, a, false)
end

struct ONNXSub{S,VS<:AbstractVector{S}} <: Node{S}
    inputs::VS
    outputs::VS
    name::S
end

onnx_node_to_flux_layer(node::ONNXSub) = (x, y) -> x .- y

islinear(node::ONNXSub) = true

function NNL.construct_layer_sub(::Type{OnnxType}, name, inputs, outputs, a::Type{NNL.DynamicInput}, b::Type{NNL.DynamicInput})
    VERBOSE_ONNX[] > 0 && println("Constructing ONNXSub node: $name with inputs $(inputs) and outputs $(outputs)")
    ONNXSub(inputs, outputs, name)
end


############################################################################
## Multiplication with different configurations of constant inputs        ##
############################################################################


struct ONNXMulConst{S,VS<:AbstractVector{S},C} <: Node{S}
    inputs::VS
    outputs::VS
    name::S
    c::C
end

onnx_node_to_flux_layer(node::ONNXMulConst) = x -> x .* node.c

islinear(node::ONNXMulConst) = true

function NNL.construct_layer_mul(::Type{OnnxType}, name, inputs, outputs, a::Type{NNL.DynamicInput}, b)
    VERBOSE_ONNX[] > 0 && println("Constructing ONNXSubConst node: $name with inputs $(inputs) and outputs $(outputs)")
    b = DOUBLE_PRECISION[] ? Float64.(b) : b
    ONNXMulConst(inputs, outputs, name, b)
end

function NNL.construct_layer_mul(::Type{OnnxType}, name, inputs, outputs, a, b::Type{NNL.DynamicInput})
    VERBOSE_ONNX[] > 0 && println("Constructing ONNXSubConst node: $name with inputs $(inputs) and outputs $(outputs)")
    a = DOUBLE_PRECISION[] ? Float64.(a) : a
    ONNXMulConst(inputs, outputs, name, a)
end


############################################################################
##                      Division by Constant                              ##
############################################################################

struct ONNXDivConst{S,VS<:AbstractVector{S},C} <: Node{S}
    inputs::VS
    outputs::VS
    name::S
    c::C
end

onnx_node_to_flux_layer(node::ONNXDivConst) = x -> x ./ node.c

islinear(node::ONNXDivConst) = true

function NNL.construct_layer_div(::Type{OnnxType}, name, inputs, outputs, a::Type{NNL.DynamicInput}, b)
    VERBOSE_ONNX[] > 0 && println("Constructing ONNXDivConst node: $name with inputs $(inputs) and outputs $(outputs)")
    b = DOUBLE_PRECISION[] ? Float64.(b) : b
    ONNXDivConst(inputs, outputs, name, b)
end


#############################################################################
##                      Convolution                                        ##
#############################################################################


struct ONNXConv{S,VS<:AbstractVector{S},C<:Conv} <: Node{S}
    inputs::VS
    outputs::VS
    name::S
    conv::C
end

onnx_node_to_flux_layer(node::ONNXConv) = node.conv

islinear(node::ONNXConv) = true


"""
Creates an instance of a Convolutional node.

For Conv((k₁, ..., kₙ), c_in => c_out), we have
- size(weight) = (k₁, ..., kₙ, c_in, c_out)
- size(bias) = (c_out,)

args:
    inputs
    outputs
    name
    weight
    bias

kwargs:
    stride - either an Integer or a TUPLE of integers, representing the stride in each dimension
    pad - either an Integer or a TUPLE of integers, representing the padding in each dimension
    dilation - eiher an Integer or a TUPLE of integers, representing the dilation in each dimension
    groups - Integer representing number of groups
    double_precision - whether to use double precision weights
"""
function ONNXConv(inputs::AbstractVector{S}, outputs::AbstractVector{S}, name::S, 
                    weight, bias=false; stride=1, pad=0, dilation=1, groups=1, double_precision=false) where S
    # TODO: is this correct?
    kernel_size = size(weight)[1:end-2]
    in_channels, out_channels = size(weight)[end-1:end]

    conv = Conv(kernel_size, in_channels => out_channels, bias=bias, stride=stride, pad=pad, dilation=dilation, groups=groups)

    if double_precision
        conv = conv |> f64
    end

    conv.weight .= weight

    return ONNXConv(inputs, outputs, name, conv)
end


function NNL.construct_layer_conv(::Type{OnnxType}, name, inputs, outputs, data, weights, bias=false;
                                  auto_pad="NOTSET", dilations=nothing, group=1, kernel_shape=nothing, pads=nothing, strides=nothing)
    @assert auto_pad == "NOTSET" "auto_pad currently not supported! (node $name)"
    VERBOSE_ONNX[] > 0 && println("Constructing ONNXConv node: $name with inputs $(inputs) and outputs $(outputs)")

    strides = isnothing(strides) ? 1 : convert2intOrTuple(strides)
    dilations = isnothing(dilations) ? 1 : convert2intOrTuple(dilations)
    pads = isnothing(pads) ? 0 : convert_onnx_pad(convert2intOrTuple(pads))

    # onnx really calculates CrossCorrelation, so need to flip weights for convolution
    weights = flipweights(weights)

    ONNXConv(inputs, outputs, name, weights, bias, stride=strides, pad=pads, dilation=dilations, groups=group, double_precision=DOUBLE_PRECISION[])
end


##################################################################################
##                          Transposed Convolution                              ##
##################################################################################


struct ONNXConvT{S,VS<:AbstractVector{S},C<:ConvTranspose} <: Node{S}
    inputs::VS
    outputs::VS
    name::S
    convt::C
end

onnx_node_to_flux_layer(node::ONNXConvT) = node.convt

islinear(node::ONNXConvT) = true


"""
Create a transposed convolution node.

For ConvTranspose((k₁, ..., kₙ), c_in => c_out), we have
- size(weight) = (k₁, ..., kₙ, c_out, c_in)
- size(bias) = (c_out,)

"""
function ONNXConvT(inputs::AbstractVector{S}, outputs::AbstractVector{S}, name::S, 
    weight, bias; stride=1, pad=0, dilation=1, groups=1, double_precision=false) where S
    # TODO: is this correct?
    kernel_size = size(weight)[1:end-2]
    # for ConvT order is swapped when compared to Conv
    out_channels, in_channels = size(weight)[end-1:end]

    convt = ConvTranspose(kernel_size, in_channels => out_channels, bias=bias, stride=stride, pad=pad, dilation=dilation, groups=groups)

    if double_precision
        convt = convt |> f64
    end

    convt.weight .= weight


    return ONNXConvT(inputs, outputs, name, convt)
end


function NNL.construct_layer_conv_transpose(::Type{OnnxType}, name, inputs, outputs, data, weights, bias;
                                            auto_pad="NOTSET", dilations=nothing, group=1, kernel_shape=nothing, output_padding=nothing, output_shape=nothing, pads=nothing, strides=nothing)
    VERBOSE_ONNX[] > 0 && println("Constructing ONNXConvT node: $name with inputs $(inputs) and outputs $(outputs)")
    @assert auto_pad == "NOTSET" "auto_pad currently not supported! (node $name)"
    @assert data == NNL.DynamicInput "Expected DynamicInput for data, but got $data"
    strides = isnothing(strides) ? 1 : convert2intOrTuple(strides)
    dilations = isnothing(dilations) ? 1 : convert2intOrTuple(dilations)
    pads = isnothing(pads) ? 0 : convert2intOrTuple(pads)
    
    !isnothing(output_padding) && @warn("Flux doesn't support output padding, got output_padding = $output_padding")
    !isnothing(output_shape) && @warn("Flux doesn't support output shape, got output_shape = $output_shape")

    weights = flipweights(weights)

    ONNXConvT(inputs, outputs, name, weights, bias, stride=strides, pad=pads, dilation=dilations, groups=group, double_precision=DOUBLE_PRECISION[])
end


struct ONNXAveragePool{S,VS<:AbstractVector{S},M<:MeanPool} <: Node{S}
    inputs::VS
    outputs::VS
    name::S
    avg::M
end

onnx_node_to_flux_layer(node::ONNXAveragePool) = node.avg

islinear(node::ONNXAveragePool) = true


function ONNXAveragePool(inputs, outputs, name, window::NTuple; pad=0, stride=window)
    avg = Flux.MeanPool(window, pad=pad, stride=stride)

    return ONNXAveragePool(inputs, outputs, name, avg)
end

function NNL.construct_layer_average_pool(::Type{OnnxType}, name, inputs, outputs, data; auto_pad="NOTSET", 
                                          ceil_mode=0, count_include_pad=0, dilations=nothing, kernel_shape=nothing, 
                                          pads=nothing, strides=nothing)
    @assert auto_pad == "NOTSET" "auto_pad currently not supported! (node $name)"
    @assert ceil_mode == 0 "only ceil_mode = 0 supported! (node $name)"
    @assert count_include_pad == 1 || all(pads .== 0) "only count_include_pad = 1 supported! (node $name) (exception, when pads = 0 in every entry)"
    @assert isnothing(dilations) "dilations not supported! (node $name)"
    VERBOSE_ONNX[] > 0 && println("Constructing ONNXAveragePool node: $name with inputs $(inputs) and outputs $(outputs)")

    strides = isnothing(strides) ? 1 : convert2intOrTuple(strides)
    dilations = isnothing(dilations) ? 1 : convert2intOrTuple(dilations)
    pads = isnothing(pads) ? 0 : convert_onnx_pad(convert2intOrTuple(pads))
    window = reverse(Tuple(kernel_shape))

    ONNXAveragePool(inputs, outputs, name, window, stride=strides, pad=pads)   
end


struct ONNXDropout{S,VS<:AbstractVector{S},R,M} <: Node{S}
    inputs::VS
    outputs::VS
    name::S
    ratio::R
    training_mode::M
end


function NNL.construct_layer_dropout(::Type{OnnxType}, name, inputs, outputs, data, ratio=0.5, training_mode=false)
    @assert data == NNL.DynamicInput
    VERBOSE_ONNX[] > 0 && println("Constructing Dropout layer: $name with inputs $(inputs) and outputs $(outputs) (ratio = $ratio, training_mode = $training_mode)")
    return ONNXDropout(inputs, outputs, name, ratio, training_mode)
end


struct ONNXBatchNorm{S,VS<:AbstractVector{S},B<:BatchNorm} <: Node{S}
    inputs::VS
    outputs::VS
    name::S
    batchnorm::B
end

onnx_node_to_flux_layer(node::ONNXBatchNorm) = node.batchnorm

islinear(node::ONNXBatchNorm) = true


function ONNXBatchNorm(inputs, outputs, name, μ, γ, β, σ²; λ=identity, ϵ=1e-5, double_precision=false)
    channels = length(γ)

    batchnorm = BatchNorm(channels)

    batchnorm.σ² .= σ²
    batchnorm.ϵ = ϵ
    batchnorm.μ .= μ
    batchnorm.γ .= γ
    batchnorm.β .= β
    batchnorm.λ = λ

    if double_precision
        batchnorm = batchnorm |> f64
    end

    # don't update parameters
    testmode!(batchnorm)

    return ONNXBatchNorm(inputs, outputs, name, batchnorm)
end


function NNL.construct_layer_batch_normalization(::Type{OnnxType}, name, inputs, outputs, X, scale, B, input_mean, input_var; 
                                                 epsilon=1e-5, momentum=0.9, training_mode=0)
    @assert X == NNL.DynamicInput
    VERBOSE_ONNX[] > 0 && println("Constructing BatchNormalization node: $name with inputs $(inputs) and outputs $(outputs)")
    return ONNXBatchNorm(inputs, outputs, name, input_mean, scale, B, input_var, ϵ=epsilon, double_precision=DOUBLE_PRECISION[])
end



struct ONNXReduceSum{S,VS<:AbstractVector{S},N<:Integer} <: Node{S}
    inputs::VS
    outputs::VS
    name::S
    axes::AbstractArray{N}
    keepdims::Bool
    noop_with_empty_axes::Bool
end

# TODO: have a factory method that constructs the function once 
#       and then the returned function doesn't have that many ifs.
onnx_node_to_flux_layer(node::ONNXReduceSum) = x -> begin
    if length(node.axes) == 0 && node.noop_with_empty_axes
        return x 
    elseif length(node.axes) == 0 && node.keepdims
        # reduce over all elements
        # if dims is an array, Julia keeps dim by default
        return sum(x, dims=1:ndims(x))
    elseif length(node.axes) == 0
        # reduce over all elements and don't keep dim
        return sum(x)
    else
        axes = ndims(x) .- node.axes  # NCHW -> WHCN       
        y = sum(x, dims=axes)

        if !node.keepdims
            idx = Tuple(i in axes ? 1 : Colon() for i in 1:ndims(x))
            y = y[idx...]
        end

        return y
    end
end

islinear(node::ONNXReduceSum) = true

function NNL.construct_layer_reducesum(::Type{OnnxType}, name, inputs, outputs, data, axes; keepdims=1, noop_with_empty_axes=0)
    # TODO: actually axes=[] is a default value, but if I add the default value here, then Julia cannot precompile the method 
    # as default values are not part of the method signature --> duplicate signature with the below method!
    @assert all(axes .>= 0) "Only non-negative axes are supported!"
    return ONNXReduceSum(inputs, outputs, name, axes, (keepdims == 1), (noop_with_empty_axes == 1))
end

function NNL.construct_layer_reducesum(node::Type{OnnxType}, name, inputs, outputs, data; axes=[], keepdims=1, noop_with_empty_axes=0)
    # in ONNX 11, axes is a kwarg, not an arg
    return NNL.construct_layer_reducesum(node, name, inputs, outputs, data, axes, keepdims=keepdims, noop_with_empty_axes=noop_with_empty_axes)
end


############################################################################
##                              Upsample                                  ##
############################################################################


struct ONNXUpsample{S,VS<:AbstractVector{S},U<:Upsample} <: Node{S}
    inputs::VS
    outputs::VS
    name::S
    upsampling::U
end

onnx_node_to_flux_layer(node::ONNXUpsample) = node.upsampling


islinear(node::ONNXUpsample) = true


function ONNXUpsample(inputs, outputs, name; mode=:nearest, scale=nothing, size=nothing)
    @assert ~isnothing(scale) || ~isnothing(size) "Either size or scale needs to be set! (constructor of $name)"
    upsampling = Upsample(mode, scale=scale, size=size)
    return ONNXUpsample(inputs, outputs, name, upsampling)
end

function NNL.construct_layer_upsample(::Type{OnnxType}, name, inputs, outputs, data, scales; mode="nearest")
    @assert data == NNL.DynamicInput "Expected DynamicInput for data, but got $data"
    # NCHW -> WHCN
    scales = reverse(tuple(Integer.(scales)...))
    return ONNXUpsample(inputs, outputs, name, scale=scales)
end