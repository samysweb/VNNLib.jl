
"""
    A node is guaranteed to include 
    - a list of identifiers of type S for inputs (parents) 
    - a list of identifiers of type S for outputs (children)
    - an identifier of type S (name)
    - the params of the node
"""
abstract type Node{S} end


struct DummyInputNode{S} <: Node{S}
    inputs::AbstractVector{S}
    outputs::AbstractVector{S}
    name::S
end


"""
Mirrors weights tensor along each dimension.

Flux convolution is true convolution, so we have to flip the weights from
onnx convolution, which is CrossCorrelation in reality.
"""
flipweights(w::AbstractArray{T,N}) where {T,N} = w[(size(w, i):-1:1 for i = 1:(N-2))..., :, :]

"""
Converts the argument to a tuple, if it is a one dimensional array, 
or does nothing, if the argument is a single integer.

(Flux convolution allows for integer parameters for padding, dilation, ... If 
the parameter is a single integer, it is the same for all dimensions, if there are different
values per dimension, it has to be passed as a TUPLE, not a array.)
"""
function convert2intOrTuple(v::AbstractArray{<:Integer, 1})
    # TOOD: do we need to revert all of these tuples?
    return reverse(Tuple(v))
end

function convert2intOrTuple(v::Integer)
    return v
end

"""
Padding in ONNX (that is already reverted) is defined as (pad_1_begin, pad_2_begin, ..., pad_1_end, pad_2_end, ...),
but Flux needs it in the form of (pad_1_begin, pad_1_end, pad_2_begin, pad_2_end, ...)
"""
function convert_onnx_pad(pad::NTuple{N, <:Integer}) where N   
    half = N ÷ 2
    return Tuple([ifelse(iseven(i), pad[half + (i ÷ 2)], pad[(i + 1) ÷ 2]) for i in 1:length(pad)])
end


function convert_negative_dim(dim::Integer, x::AbstractArray)
    # ONNX uses NCHW order, so if ONNX indexes a dimension with a positive index, it will be the same index counting from the **end** 
    # of the array in WHCN order.
    # It is impossible for us to know the index at construction time.
    # Therefore, we negate the index and compute the index at runtime with this function.
    convert_negative_dim(dim, ndims(x))
end


function convert_negative_dim(dim::Integer, data_dims::Integer)
    @assert dim <= 0 "Non-positive dimension required! (got $dim)"
    @assert data_dims > 0 "Data dimensions must be positive! (got $data_dims)"
    return data_dims + dim
end


function get_positive_index(idx, len::Integer)
    idx < 0 ? len + idx + 1 : idx 
end