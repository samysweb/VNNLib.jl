

struct ONNXMul{S} <: Node{S}
    inputs::AbstractVector{S}
    outputs::AbstractVector{S}
    name::S
end

onnx_node_to_flux_layer(node::ONNXMul) = (x, y) -> x .* y

function NNL.construct_layer_mul(::Type{OnnxType}, name, inputs, outputs, a::Type{NNL.DynamicInput}, b::Type{NNL.DynamicInput})
    VERBOSE_ONNX[] > 0 && println("Constructing Mul layer: $name with inputs $inputs and outputs $outputs")
    return ONNXMul(inputs, outputs, name)
end


# Need to write it as Relu as ReLU is in NeuralVerification and relu is in Flux
struct ONNXRelu{S} <: Node{S}
    inputs::AbstractVector{S}
    outputs::AbstractVector{S}
    name::S
end

onnx_node_to_flux_layer(node::ONNXRelu) = Flux.relu

function NNL.construct_layer_relu(::Type{OnnxType}, name, inputs, outputs, data)
    VERBOSE_ONNX[] > 0 && println("Constructing ReLU layer: $name")
    return ONNXRelu(inputs, outputs, name)
end


struct ONNXSigmoid{S} <: Node{S}
    inputs::AbstractVector{S}
    outputs::AbstractVector{S}
    name::S
end

onnx_node_to_flux_layer(node::ONNXSigmoid) = Flux.sigmoid

function NNL.construct_layer_sigmoid(::Type{OnnxType}, name, inputs, outputs, data)
    VERBOSE_ONNX[] > 0 && println("Constructing Sigmoid layer: $name")
    return ONNXSigmoid(inputs, outputs, name)
end


struct ONNXTanh{S} <: Node{S}
    inputs::AbstractVector{S}
    outputs::AbstractVector{S}
    name::S
end

onnx_node_to_flux_layer(node::ONNXTanh) = Flux.tanh_fast

function NNL.construct_layer_tanh(::Type{OnnxType}, name, inputs, outputs, data)
    VERBOSE_ONNX[] > 0 && println("Constructing Tanh layer: $name")
    return ONNXTanh(inputs, outputs, name)
end


# need to implement sqrt, exp, abs, acos, hard_sigmoid, hard_swish, elu, gelu

struct ONNXLeakyRelu{S,F} <: Node{S}
    inputs::AbstractVector{S}
    outputs::AbstractVector{S}
    name::S
    alpha::F
end

onnx_node_to_flux_layer(node::ONNXLeakyRelu) = x -> Flux.leakyrelu(x, node.alpha)

function NNL.construct_layer_leaky_relu(::Type{OnnxType}, name, inputs, outputs, data; alpha=0.01)
    VERBOSE_ONNX[] > 0 && println("Constructing Leaky ReLU layer: $name with alpha = $alpha")
    return ONNXLeakyRelu(inputs, outputs, name, alpha) 
end

struct ONNXSign{S} <: Node{S}
    inputs::AbstractVector{S}
    outputs::AbstractVector{S}
    name::S
end

onnx_node_to_flux_layer(node::ONNXSign) = x -> sign.(x)

function NNL.construct_layer_sign(::Type{OnnxType}, name, inputs, outputs, data)
    VERBOSE_ONNX[] > 0 && println("Constructing Sign layer: $name")
    return ONNXSign(inputs, outputs, name)
end

struct ONNXSin{S} <: Node{S}
    inputs::AbstractVector{S}
    outputs::AbstractVector{S}
    name::S
end

onnx_node_to_flux_layer(node::ONNXSin) = x -> sin.(x)

function NNL.construct_layer_sin(::Type{OnnxType}, name, inputs, outputs, data)
    VERBOSE_ONNX[] > 0 && println("Constructing Sin layer: $name")
    return ONNXSin(inputs, outputs, name)
end

struct ONNXCos{S} <: Node{S}
    inputs::AbstractVector{S}
    outputs::AbstractVector{S}
    name::S
end

onnx_node_to_flux_layer(node::ONNXCos) = x -> cos.(x)

function NNL.construct_layer_cos(::Type{OnnxType}, name, inputs, outputs, data)
    VERBOSE_ONNX[] > 0 && println("Constructing Cos layer: $name")
    return ONNXCos(inputs, outputs, name)
end

struct ONNXExp{S} <: Node{S}
    inputs::AbstractVector{S}
    outputs::AbstractVector{S}
    name::S
end

onnx_node_to_flux_layer(node::ONNXExp) = x -> exp.(x)

function NNL.construct_layer_exp(::Type{OnnxType}, name, inputs, outputs, data)
    VERBOSE_ONNX[] > 0 && println("Constructing Exp layer: $name")
    return ONNXExp(inputs, outputs, name)
end

struct ONNXSqrt{S} <: Node{S}
    inputs::AbstractVector{S}
    outputs::AbstractVector{S}
    name::S
end

onnx_node_to_flux_layer(node::ONNXSqrt) = x -> sqrt.(x)

function NNL.construct_layer_sqrt(::Type{OnnxType}, name, inputs, outputs, data)
    VERBOSE_ONNX[] > 0 && println("Constructing Sqrt layer: $name")
    return ONNXSqrt(inputs, outputs, name)
end

struct ONNXAbs{S} <: Node{S}
    inputs::AbstractVector{S}
    outputs::AbstractVector{S}
    name::S
end

onnx_node_to_flux_layer(node::ONNXAbs) = x -> abs.(x)

function NNL.construct_layer_abs(::Type{OnnxType}, name, inputs, outputs, data)
    VERBOSE_ONNX[] > 0 && println("Constructing Abs layer: $name")
    return ONNXAbs(inputs, outputs, name)
end

struct ONNXAcos{S} <: Node{S}
    inputs::AbstractVector{S}
    outputs::AbstractVector{S}
    name::S
end

onnx_node_to_flux_layer(node::ONNXAcos) = x -> acos.(x)

function NNL.construct_layer_acos(::Type{OnnxType}, name, inputs, outputs, data)
    VERBOSE_ONNX[] > 0 && println("Constructing Acos layer: $name")
    return ONNXAcos(inputs, outputs, name)
end

struct ONNXHardSigmoid{S,F} <: Node{S}
    inputs::AbstractVector{S}
    outputs::AbstractVector{S}
    name::S
    alpha::F
    beta::F
end


function onnx_node_to_flux_layer(node::ONNXHardSigmoid) 
    if node.alpha == 1/6 && node.beta == 0.5
        return x -> Flux.hardsigmoid(x)
    else
        return x -> max.(0, min.(1, node.alpha * x .+ node.beta))
    end
end

function NNL.construct_layer_hard_sigmoid(::Type{OnnxType}, name, inputs, outputs, data; alpha=1/6, beta=0.5)
    VERBOSE_ONNX[] > 0 && println("Constructing Hard Sigmoid layer: $name with alpha = $alpha and beta = $beta")
    T = typeof(alpha)
    beta = convert(T, beta)  # force beta to be the same type as alpha
    return ONNXHardSigmoid(inputs, outputs, name, alpha, beta)
end

struct ONNXHardSwish{S,F} <: Node{S}
    inputs::AbstractVector{S}
    outputs::AbstractVector{S}
    name::S
    alpha::F
    beta::F
end

function onnx_node_to_flux_layer(node::ONNXHardSwish) 
    if node.alpha == 1/6 && node.beta == 0.5
        return x -> Flux.hardswish(x)
    else
        return x -> x .* max.(0, min.(1, node.alpha * x .+ node.beta))
    end
end

function NNL.construct_layer_hard_swish(::Type{OnnxType}, name, inputs, outputs, data; alpha=1/6, beta=0.5)
    VERBOSE_ONNX[] > 0 && println("Constructing Hard Swish layer: $name with alpha = $alpha and beta = $beta")
    T = typeof(alpha)
    beta = convert(T, beta)  # force beta to be the same type as alpha
    return ONNXHardSwish(inputs, outputs, name, alpha, beta)
end

struct ONNXElu{S,F} <: Node{S}
    inputs::AbstractVector{S}
    outputs::AbstractVector{S}
    name::S
    alpha::F
end

onnx_node_to_flux_layer(node::ONNXElu) = x -> Flux.elu(x, node.alpha)

function NNL.construct_layer_elu(::Type{OnnxType}, name, inputs, outputs, data; alpha=1.0)
    VERBOSE_ONNX[] > 0 && println("Constructing ELU layer: $name with alpha = $alpha")
    return ONNXElu(inputs, outputs, name, alpha)
end

struct ONNXGelu{S} <: Node{S}
    inputs::AbstractVector{S}
    outputs::AbstractVector{S}
    name::S
    approximate::S
end

function onnx_node_to_flux_layer(node::ONNXGelu)
    if node.approximate == "none"
        x -> Flux.gelu_erf(x)
    elseif node.approximate == "tanh"
        x -> Flux.gelu_tanh(x)
    else
        error("Unsupported GELU approximation: $(node.approximate)")
    end
end

function NNL.construct_layer_gelu(::Type{OnnxType}, name, inputs, outputs, data; approximate="none")
    VERBOSE_ONNX[] > 0 && println("Constructing GELU layer: $name with approximation = $approximate")
    return ONNXGelu(inputs, outputs, name, approximate)
end


struct ONNXSoftmax{S} <: Node{S}
    inputs::AbstractVector{S}
    outputs::AbstractVector{S}
    name::S
    axis::Int
end

onnx_node_to_flux_layer(node::ONNXSoftmax) = x -> Flux.softmax(x, dims=node.axis)

function NNL.construct_layer_softmax(::Type{OnnxType}, name, inputs, outputs, data; axis=-1)
    VERBOSE_ONNX[] > 0 && println("Constructing Softmax layer: $name with axis = $axis")
    return ONNXSoftmax(inputs, outputs, name, axis)
end


struct ONNXUpsample{S} <: Node{S}
    inputs::AbstractVector{S}
    outputs::AbstractVector{S}
    name::S
    upsampling::Upsample
end


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


# TODO: Maybe better to use Flux here?
struct ONNXLSTMCell{S} <: Node{S}
    inputs::AbstractVector{S}
    outputs::AbstractVector{S}
    name::S
    linear_ih::ONNXLinear
    linear_hh::ONNXLinear
    # (hidden_state, cell_state)
    state0::Tuple{AbstractArray, AbstractArray}
end


function ONNXLSTMCell(inputs, outputs, name, Wih::AbstractArray{<:N}, Whh::AbstractArray{<:N}, b::AbstractVector{<:N}; state0=nothing) where N<:Number
    hs4, n_in = size(Wih)
    hidden_size = floor(Integer, hs4 / 4)

    linear_ih = ONNXLinear([], [], name * "_linear_ih", Wih, b)
    linear_hh = ONNXLinear([], [], name * "_linear_hh", Whh, zeros(eltype(Whh), hs4))

    if isnothing(state0)
        # (hidden_state, cell_state)
        state0 = (zeros(hidden_size), zeros(hidden_size))
    end

    return ONNXLSTMCell(inputs, outputs, name, linear_ih, linear_hh, state0)
end

struct ONNXLSTMLayer{S,AN} <: Node{S}
    # only need LSTMLayer, passing the whole sequence through the unrolling is handled inside the propagation of LSTMLayer
    inputs::AbstractVector{S}
    outputs::AbstractVector{S}
    name::S
    cell::Flux.LSTMCell
    num_directions
    # initial state
    h0::AN
    c0::AN
end

onnx_node_to_flux_layer(node::ONNXLSTMLayer) = x -> begin
    # in ONNX, the input is of order (length, batch_size, input_size)
    # transformed to WHCN order, it is (input_size, batch_size, length),
    # but Flux expects (input_size, length, batch_size) for the input
    x = permutedims(x, (1, 3, 2)) 
    y, (hn, cn) = Flux.LSTM(node.cell, return_state=true)(x, (node.h0, node.c0))

    # Flux returns (output, (hidden_state, cell_state)) just as ONNX, but ONNX also 
    # adds a dimension for the number of directions.
    # In Flux, the output is of the following shapes:
    # - y: (output_size, length, batch_size)
    # - hn, cn: (output_size, batch_size)
    #
    # In ONNX, the output is of the following shapes:
    # - y: (length, num_directions, batch_size, output_size)
    # - hn, cn: (num_directions, batch_size, output_size)
    #
    # We keep WHCN order, so we permute that to 
    # - y: (output_size, batch_size, num_directions, length)
    # - hn, cn: (output_size, batch_size, num_directions)
    y = permutedims(y, (1, 3, 2))
    @assert node.num_directions == 1 "Bidirectional LSTM is not supported! Got num_directions = $(node.num_directions)"
    # in the bidirectional case, the new dimension would have size 2!
    y = unsqueeze(y, 3)
    hn = unsqueeze(hn, 3)
    cn = unsqueeze(cn, 3)
    y, hn, cn
end


function extract_cell(lstm::ONNXLSTMLayer)
    W_ih = lstm.cell.Wi
    W_hh = lstm.cell.Wh
    b = lstm.cell.b
    state0 = lstm.cell.state0 

    return ONNXLSTMCell(lstm.inputs, lstm.outputs, lstm.name * "_cell", W_ih, W_hh, b, state0=state0)
end


function NNL.construct_layer_lstm(::Type{OnnxType}, name, inputs, outputs, data, W_ih::AbstractArray{<:M}, W_hh::AbstractArray{<:M}, bias=nothing, sequence_lens=nothing, 
                                  initial_h=nothing, initial_c=nothing, P=nothing; activation_alpha=nothing, activation_beta=nothing, activations=nothing, clip=nothing,
                                  direction="forward", hidden_size=-1, input_forget=0, layout=0) where M<:Number
    @assert data == NNL.DynamicInput "Expected DynamicInput for data but got $data"
    @assert hidden_size >= 0 "hidden_size must be set!"
    # TODO: How can this be DynamicInput?
    #       Everything that has no initializer is a DynamicInput, but this has no initializer because it is not set ...
    @assert (isnothing(sequence_lens) || sequence_lens == NNL.DynamicInput) "implementation can't use sequence_lens, got sequence_lens = $(sequence_lens)"
    if sequence_lens == NNL.DynamicInput
        VERBOSE_ONNX[] > 0 && println("LSTM layer $name has DynamicInput for sequence_lens, but it is not used in the implementation")
        sequence_lens = nothing  # convert DynamicInput to nothing
        inputs = inputs[1:end-1]  # remove sequence_lens from inputs
    end

    @assert isnothing(P) "Peephole connections are not supported!"
    @assert isnothing(activation_alpha) && isnothing(activation_beta) && isnothing(activations) "Only standard activations are supported! Got activation_alpha = $(activation_alpha), 
                                                                                                 activation_beta = $(activation_beta), activations = $activations"
    @assert isnothing(clip) "implementation doesn't support clipping!"
    @assert direction == "forward" "reverse or bidirectional not supported! Got $direction"
    @assert input_forget == 0 "Coupling of input and forget gates is not supported!"
    # TODO: what is the supported layout???
    VERBOSE_ONNX[] > 0 && println("Constructing LSTM layer: $name (input size = $(size(W_ih, 1)), hidden size = $(size(W_hh, 1)), direction = $direction)")

    # | Param | ONNX input shape  | required Flux shape |
    # +-------+-------------------+---------------------+
    # | W_ih  |(n_i, 4*n_h, dirs) |  (4*n_h, n_i)       |
    # | W_hh  |(n_h, 4*n_h, dirs) |  (4*n_h, n_h)       |
    # |b      | (2*4*n_h, dirs)   |  (4*n_h)            |
  
    input_size = size(W_ih, 1)
    hidden_size = size(W_hh, 1)

    # need to reorder columns of W_ih and W_hh
    # Flux has [i f c o] order for input, forget, cell and output gate, while
    # ONNX has [i o f c] order
    reordering = (1:hidden_size) ∪ (2*hidden_size+1:3*hidden_size) ∪ (3*hidden_size+1:4*hidden_size) ∪ (hidden_size+1:2*hidden_size)
    
    num_directions = size(W_ih, 3)  # this should be 1 with asserting direction == "forward"
    W_ih = W_ih[:,reordering,1]'  # have no bidirectional in Flux, so assume weights for first direction are the ones for forward
    W_hh = W_hh[:,reordering,1]'  # transpose to get required shape

    # in ONNX biases for ih and hh are stacked, but only one bias is required, which is the result of the addition of both halves
    bias = isnothing(bias) ? zeros(M, 4*hidden_size) : bias[reordering, 1] .+ bias[reordering .+ 4*hidden_size, 1]
    initial_h = isnothing(initial_h) ? zeros(M, hidden_size, 1) : reshape(initial_h, hidden_size, 1)  # vec(initial_h)
    initial_c = isnothing(initial_c) ? zeros(M, hidden_size, 1) : reshape(initial_c, hidden_size, 1)  #  vec(initial_c)

    cell = Flux.LSTMCell(W_ih, W_hh, bias)
    if DOUBLE_PRECISION[]
        cell = cell |> f64
        initial_h = convert.(Float64, initial_h)  # ensure initial_h is of the same type as W_ih
        initial_c = convert.(Float64, initial_c)  # ensure initial_c is of the same type as
    end

    return ONNXLSTMLayer(inputs, outputs, name, cell, num_directions, initial_h, initial_c)
end