module Internal
    include("onnx/onnx.jl")
    using .onnx

    export tensor_to_array, extract_shape


    """
    Extracts data of the correct data type from the TensorProto.

    The datatype of a tensor is specified in the tensor.data_type field.
    For each possible data type (float, int32, string, int64, double, uint64), there is a corresponding field 
    <dtype>_data in the TensorProto storing that information.

    However, sometimes the data is not stored in these specialized fields, but in raw format in the raw_data field!
    Then it has to be reinterpreted.

    See https://github.com/onnx/onnx/issues/1271 for why we need to support both representations.

    args:
        tensor - TensorProto instance
        tensor_dtype - **Julia** type of the tensor (not onnx enum)
        tensor_dtype_field - name of the corresponding specialized field for storing that data

    returns:
        reshaped array of correct type
    """
    function parse_data(tensor, dtype, tensor_dtype_field)
        if length(tensor_dtype_field) == 0
            # not data in designated field, need to look for data stored in raw format
            data = reinterpret(dtype, tensor.raw_data)
        else
            data = tensor_dtype_field
        end

        return reshape(data, Tuple(reverse(tensor.dims)))
    end


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
            return parse_data(tensor, Float32, tensor.float_data)
        elseif tensor_dtype == Int(onnx.var"TensorProto.DataType".DOUBLE)
            return parse_data(tensor, Float64, tensor.double_data)
        elseif tensor_dtype == Int(onnx.var"TensorProto.DataType".INT64)
            return parse_data(tensor, Int64, tensor.int64_data)
        elseif tensor_dtype == Int(onnx.var"TensorProto.DataType".BOOL)
            # can directly use reinterpret, as there is no separate bool field, so it has to be stored as raw data
            return reshape(reinterpret(Bool, tensor.raw_data), Tuple(reverse(tensor.dims)))
        else
            error("TensorProto.data_type $(tensor_dtype) is not yet supported")
        end
        
    end


    """
    Extracts the shape of a TensorShapeProto.

    Shapes are not always just integer vectors, e.g. ["batch_size", 1, 1] is also valid.
    """
    function extract_shape(tensor_shape_proto_dims::AbstractVector{<:var"TensorShapeProto.Dimension"})
        dims = []  # not only int, but can also be string e.g. "batch_size" or "unk_156" (which also allows for batch propagation)
        for i in eachindex(tensor_shape_proto_dims)
            # TODO: there has to be a better way!
            push!(dims, tensor_shape_proto_dims[i].value.value)
        end

    return dims
end
end