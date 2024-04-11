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

    if tensor_dtype == Int(onnx.var"TensorProto.DataType".FLOAT)
        # Float32 https://github.com/onnx/onnx/blob/fb80e3ade84e9f406711aa41b9f3665753158371/onnx/mapping.py#L13
        # TODO(steuber): check if dimension direction correct
        if length(tensor.float_data) > 0
            return reshape(tensor.float_data,Tuple(reverse(dims)))
        else
            return reshape(reinterpret(Float32, tensor.raw_data),Tuple(reverse(dims)))
        end
    elseif tensor_dtype == Int(onnx.var"TensorProto.DataType".DOUBLE)
        # Float64 https://github.com/onnx/onnx/blob/fb80e3ade84e9f406711aa41b9f3665753158371/onnx/mapping.py#L13
        # TODO(steuber): check if dimension direction correct
        if length(tensor.double_data) > 0
            return reshape(tensor.double_data,Tuple(reverse(dims)))
        else
            return reshape(reinterpret(Float64, tensor.raw_data),Tuple(reverse(dims)))
        end
    else
        error("TensorProto.data_type $(tensor_dtype) is not supported")
    end
    
end