

struct ONNXConcat{S} <: Node{S}
    inputs::AbstractVector{S}
    outputs::AbstractVector{S}
    name::S
    dim::Integer
end

onnx_node_to_flux_layer(node::ONNXConcat) = (xs...) -> begin 
    # as the node only stores the dimension along which to concatenate,
    # but not the shape of the input, we have to compute the axis 
    # after conversion from NCHW to WHCN at runtime
    axis = ndims(xs[1]) - node.dim
    cat(xs..., dims=axis)
end

function NNL.construct_layer_concat(::Type{OnnxType}, name, inputs, outputs, data...; axis=nothing)
    @assert !isnothing(axis) "Concatenation layer requires axis!"
    return ONNXConcat(inputs, outputs, name, axis)
end

struct ONNXReshape{S} <: Node{S}
    inputs::AbstractVector{S}
    outputs::AbstractVector{S}
    name::S
    shape
end

onnx_node_to_flux_layer(node::ONNXReshape) = x -> begin
    reshape(x, node.shape)
end

function NNL.construct_layer_reshape(::Type{OnnxType}, name, inputs, outputs, data, shape)
    # have assertion here instead of type annotation in argument, s.t. we get more meaningful error message
    @assert data == NNL.DynamicInput "Reshape layer requires dynamic input (@ node $(name))"
    VERBOSE_ONNX[] > 0 && println("Constructing Reshape layer: $name (shape = $shape)")

    # Flux needs WHCN instead of NCHW -> reverse
    # Julia needs : for calculate dim instead of -1 for python
    shape = reverse(tuple(map(x -> ifelse(x > 0, x, :), shape)...))
    
    return ONNXReshape(inputs, outputs, name, shape)
end

struct ONNXFlatten{S} <: Node{S}
    inputs::AbstractVector{S}
    outputs::AbstractVector{S}
    name::S
    axis::Int
end

onnx_node_to_flux_layer(node::ONNXFlatten) = x -> reshape(x, :, size(x)[end])

function NNL.construct_layer_flatten(::Type{OnnxType}, name, inputs, outputs, data; axis=1)
    VERBOSE_ONNX[] > 0 && println("Constructing Flatten layer: $name")
    @assert axis == 1 "Flattening only up to some arbitrary axis is not supported yet! Got axis = $axis at node $(name)"
    return ONNXFlatten(inputs, outputs, name, axis)
end


struct ONNXGather{S} <: Node{S}
    inputs::AbstractVector{S}
    outputs::AbstractVector{S}
    name::S
    inds::Union{Int,AbstractArray{Int}}
    axis::Int
end


function ONNXGather(inputs, outputs, name, inds; axis=1)
    return ONNXGather(inputs, outputs, name, inds, axis)
end


function my_gather(x::AbstractArray, inds::AbstractVector{<:Integer}, axis::Integer)
    # if axis > 0 we already converted it to 1-based indexing, if axis < 0, we have to convert it now
    axis = axis > 0 ? axis : convert_negative_dim(axis, x)
    inds = get_positive_index.(inds, size(x, axis))

    # gather along the specified axis
    idxs = Tuple(ifelse(i == axis, inds, :) for i in 1:ndims(x))
    return x[idxs...]
end


function my_gather(x::AbstractArray, ind::Integer, axis::Integer) 
    axis = axis > 0 ? axis : convert_negative_dim(axis, x)
    # TODO: this method and my_gather(x, inds::AbstractVector, axis) just differ in whether it is get_positive_index.() or get_positive_index (whithout dot)
    #       maybe we should just use the same method for both?
    ind = get_positive_index(ind, size(x, axis))
    idx = Tuple(ifelse(i == axis, ind, :) for i in 1:ndims(x))
    return x[idx...]
end


function my_gather(x::AbstractArray, inds::AbstractArray{N,0}, axis::Integer) where N<:Number 
    # special case for 0-dimensional indices (i.e. arrays holding scalars), which are not supported by the default gather
    # can happen when you have e.g. fill(4) in Julia
    my_gather(x, inds[], axis)
end


onnx_node_to_flux_layer(node::ONNXGather) = x -> my_gather(x, node.inds, node.axis)


function NNL.construct_layer_gather(::Type{OnnxType}, name, inputs, outputs, data, inds; axis=0)
    VERBOSE_ONNX[] > 0 && println("Constructing Gather layer: $name (inds = $inds, axis = $axis)")
    # convert to WHCN order, which is easy if axis is negative (because -1 is the last index, it is the first index in WHCN)
    # but if the index is positive, we cannot know the size of the input at construction time,
    # so we pass it along to convert it at runtime later (which we'll notice, because the axis will be negative then)
    axis = -axis 
    inds = ifelse.(inds .< 0, inds, inds .+ 1) # convert to 1-based indexing, but leave negative indices as they are (reverse problem to axis)
    return ONNXGather(inputs, outputs, name, inds, axis)
end
  

struct ONNXSlice{S} <: Node{S}
    inputs::AbstractVector{S}
    outputs::AbstractVector{S}
    name::S
    starts::AbstractArray{<:Integer}
    # writing ends in Julia is annoying!
    stops::AbstractArray{<:Integer}
    axes::AbstractArray{<:Integer}
    steps::AbstractArray{<:Integer}
end


function ONNXSlice(inputs, outputs, name, starts, stops, axes; steps=1)
    @assert all(starts .>= 0) && all(stops .>= 0) "Negative starts or ends are currently not supported! (@ $(Node.name))"
    return ONNXSlice(inputs, outputs, name, starts, stops, axes, steps)
end

function NNL.construct_layer_slice(::Type{OnnxType}, name, inputs, outputs, data, starts, ends, axes, steps)
    VERBOSE_ONNX[] > 0 && println("Constructing Slice layer: $name (starts = $starts, ends = $ends, axes = $axes, steps = $steps)")
    return ONNXSlice(inputs, outputs, name, starts, ends, axes, steps=steps)
end


struct ONNXSplit{S} <: Node{S}
    inputs::AbstractVector{S}
    outputs::AbstractVector{S}
    name::S
    axis::Int
    splits::Union{Vector{Int}, Nothing}
    num_outputs::Union{Int, Nothing}
end


function ONNXSplit(inputs, outputs, name; splits=nothing, num_outputs=nothing, axis=1)
    @assert ~isnothing(splits) || ~isnothing(num_outputs) "Either splits or num_outputs has to be set (@ node $(name))"
    if isnothing(num_outputs)
        num_outputs = length(splits)
    end
    # since we don't know the input dimensions, we can't set splits here 
    return ONNXSplit(inputs, outputs, name, axis, splits, num_outputs)
end


function onnx_split(node::ONNXSplit, x)
    # if axis > 0 we already converted it to 1-based indexing, if axis < 0, we have to convert it now
    axis = node.axis > 0 ? node.axis : convert_negative_dim(node.axis, x)

    if isnothing(node.splits)
        # if splits is not set, we assume that the number of outputs is set
        @assert !isnothing(node.num_outputs) "Either splits or num_outputs has to be set (@ node $(node.name))"
        # according to ONNX, we then split along node.axis into equally sized parts (if not evenly divisible, the last part will be smaller)
         ranges = Iterators.partition(1:size(x, axis), ceil(Integer, size(x, axis) / node.num_outputs))
    else
        @assert !isnothing(node.splits) "Either splits or num_outputs has to be set (@ node $(node.name))"
        starts = [0; cumsum(node.splits[1:end-1])]
        stops  = cumsum(node.splits)
        ranges = [start+1:stop for (start, stop) in zip(starts, stops)]
    end

    replace_ranges = inds -> begin
        idxs = [1:size(x,i) for i in 1:ndims(x)]
        idxs[axis] = inds
        idxs
    end

    # factor out replace_ranges s.t. we can use a comprehension and stay type stable
    [x[replace_ranges(inds)...] for inds in ranges]
end

onnx_node_to_flux_layer(node::ONNXSplit) = x -> onnx_split(node, x)


function NNL.construct_layer_split(::Type{OnnxType}, name, inputs, outputs, data, splits; num_outputs=nothing, axis=1)
    @assert data == NNL.DynamicInput "Split layer requires dynamic input (@ node $(name))"
    VERBOSE_ONNX[] > 0 && println("Constructing Split layer: $name (splits = $splits, num_outputs = $num_outputs, axis = $axis)")

    # convert to WHCN order, which is easy if axis is negative (because -1 is the last index, it is the first index in WHCN) 
    # but if the index is positive, we cannot know the size of the input at construction time,
    # so we pass it along to convert it at runtime later (which we'll notice, because the axis will be negative then)
    axis = -axis
    return ONNXSplit(inputs, outputs, name, axis, isnothing(splits) ? splits : Array(splits), num_outputs)
end


struct ONNXTranspose{S} <: Node{S}
    inputs::AbstractVector{S}
    outputs::AbstractVector{S}
    name::S
    perm
end

onnx_node_to_flux_layer(node::ONNXTranspose) = x -> permutedims(x, node.perm)

function NNL.construct_layer_transpose(::Type{OnnxType}, name, inputs, outputs, data; perm=nothing)
    @assert data == NNL.DynamicInput
    VERBOSE_ONNX[] > 0 && println("Constructing Transpose layer: $name (perm = $perm)")

    # Flux needs WHCN instead of NCHW -> reverse
    # Since dims are reversed, the index of the smallest dim needs to be largest and the index of the largest dim needs to be 1 (since Julia is 1-indexed)
    #   -> subract the current index from (max(perm) + 1)
    perm = Tuple(reverse((maximum(perm) + 1) .- perm))
    
    return ONNXTranspose(inputs, outputs, name, perm)
end

struct ONNXSqueeze{S} <: Node{S}
    inputs::AbstractVector{S}
    outputs::AbstractVector{S}
    name::S
    # if axes == nothing, then all singleton dimensions will be removed
    axes
end

onnx_node_to_flux_layer(node::ONNXSqueeze) = x -> begin
    axes = isnothing(node.axes) ? Tuple(findall(size(x) .== 1)) : ndims(x) .- node.axes
    @assert all(size(x)[axes] .== 1) "Can't squeeze non-singleton dimensions! Got size(x) = $(size(x)) for axes $axes !"

    dropdims(x, dims=Tuple(axes))
end

function NNL.construct_layer_squeeze(::Type{OnnxType}, name, inputs, outputs, data, axes)
    @assert data == NNL.DynamicInput
    VERBOSE_ONNX[] > 0 && println("Constructing Squeeze layer: $name (axes = $axes)")
    # Flux needs reversed dimensions, but ONNX Squeeze only stores the axes to be squeezed.
    # so we don't know how many axes there are, which makes it difficult to calculate the right indices.
    # We therefore subtract the respective index from the length of the ndims of the input array, which is known at runtime.
    # Since we subtract from the end, we also don't need to add 1 despite Julia being 1-indexed
    return ONNXSqueeze(inputs, outputs, name, axes)
end