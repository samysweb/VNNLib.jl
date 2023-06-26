module Internal
    include("onnx/onnx.jl")
    using .onnx

    export tensor_to_array

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
end