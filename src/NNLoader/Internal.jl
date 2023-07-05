module Internal
    include("onnx/onnx.jl")
    using .onnx

    export tensor_to_array, extract_shape

    function tensor_to_array(tensor :: TensorProto)
        if !isnothing(tensor.segment)
            error("TensorProto.segment is not supported")
        end
        if tensor.data_type==Int(onnx.var"TensorProto.DataType".UNDEFINED)
            error("TensorProto.data_type is UNDEFINED")
        end
        
        tensor_dtype = tensor.data_type

        if isnothing(tensor.raw_data)
            error("Import without raw_data is not supported")
        end

        dims = tensor.dims
        # Types: https://github.com/onnx/onnx/blob/fb80e3ade84e9f406711aa41b9f3665753158371/onnx/mapping.py#L13
        # TODO(steuber): check if dimension direction correct
        if tensor_dtype == Int(onnx.var"TensorProto.DataType".FLOAT)
            return reshape(reinterpret(Float32, tensor.raw_data),Tuple(reverse(dims)))
        elseif tensor_dtype == Int(onnx.var"TensorProto.DataType".DOUBLE)
            return reshape(reinterpret(Float64, tensor.raw_data),Tuple(reverse(dims)))
        elseif tensor_dtype == Int(onnx.var"TensorProto.DataType".INT64)
            return reshape(reinterpret(Int64, tensor.raw_data),Tuple(reverse(dims)))
        else
            error("TensorProto.data_type $(tensor_dtype) is not yet supported")
        end
        
    end


    """
    Extracts the shape of a TensorShapeProto.

    Shapes are not always just integer vectors, e.g. ["batch_size", 1, 1] is also valid.
    """
    function extract_shape(tensor_shape_proto_dims::AbstractVector{<:var"TensorShapeProto.Dimension"})
        dims = []  # not only int, but can also be string e.g. "batch_size"
        for i in eachindex(tensor_shape_proto_dims)
            # TODO: there has to be a better way!
            push!(dims, tensor_shape_proto_dims[i].value.value)
        end

    return dims
end
end