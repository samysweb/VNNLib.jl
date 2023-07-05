module NNLoader

    using LinearAlgebra
    using ProtoBuf
    using MLStyle

    include("Internal.jl")
    using .Internal
    using .Internal.onnx

    include("Definition.jl")
    include("Layers_Vector.jl")
    include("NetworkConstructors.jl")
            

    """
    Parses constant nodes and adds their values to the initializer_map.

    Caveats:
    - assumes that all constants have TENSOR type
    - assumes that all constants only have one attribute
    """
    function process_constants!(graph, initializer_map)
        constants = [n for n in graph.node if n.op_type == "Constant"]
        for c in constants 
            @assert length(c.attribute) == 1 "Expected only 1 attribute for constant $(c.name), got : $(c.attribute)"
            @assert c.attribute[1].var"#type" == onnx.var"AttributeProto.AttributeType".TENSOR "Expected constant of type TENSOR, got $(c.attribute[1].var"#type")"
            for o in c.output
                initializer_map[o] = c.attribute[1].t
            end
        end
    end


    function load_network_dict(net_type::Type{<:NetworkType},filename::String;return_graph=false)
        onnx_proto_model = open(filename,"r") do f
            input = ProtoDecoder(f)
            return decode(input, onnx.ModelProto)
        end
        graph = onnx_proto_model.graph
        if return_graph
            return graph
        end

        initializer_map = Dict(i.name => i for i in graph.initializer)

        process_constants!(graph, initializer_map)

        node_map = Dict()
        for node in graph.node
            if node.op_type == "Constant"
                println("Skipping constant")
                continue
            end
            print(keys(node_map))
            node_map[node.name] = process_graph_node(net_type, node, initializer_map)
        end
        all_inputs = [i.name for i in graph.input]
        inputs = [i.name for i in graph.input if !haskey(initializer_map,i.name)]
        outputs = [o.name for o in graph.output if !in(o.name, all_inputs)]

        print(inputs)
        print(outputs)
        return construct_network(net_type, inputs, outputs, node_map)
    end

    function process_graph_node(net_type::Type{<:NetworkType}, node :: NodeProto, initializer_map)
        layer_inputs = []
        for input in node.input
            if haskey(initializer_map,input)
                push!(layer_inputs,
                    tensor_to_array(initializer_map[input])
                )
            else
                push!(layer_inputs, DynamicInput)
            end
        end
        params = Dict()
        for attribute in node.attribute
            if attribute.var"#type"==onnx.var"AttributeProto.AttributeType".FLOAT
                value = attribute.f
            elseif attribute.var"#type"==onnx.var"AttributeProto.AttributeType".INT
                value = attribute.i
            elseif attribute.var"#type"==onnx.var"AttributeProto.AttributeType".STRING
                value = attribute.s
            elseif attribute.var"#type"==onnx.var"AttributeProto.AttributeType".TENSOR
                value = tensor_to_array(attribute.t)
            elseif attribute.var"#type"==onnx.var"AttributeProto.AttributeType".FLOATS
                value = attribute.floats
            elseif attribute.var"#type"==onnx.var"AttributeProto.AttributeType".INTS
                value = attribute.ints
            elseif attribute.var"#type"==onnx.var"AttributeProto.AttributeType".STRINGS
                value = attribute.strings
            elseif attribute.var"#type"==onnx.var"AttributeProto.AttributeType".TENSORS
                value = [tensor_to_array(t) for t in attribute.tensors]
            else
                error("Unknown attribute type $(attribute.var"#type")")
            end
            params[Symbol(attribute.name)] = value
        end
        inputs = [i for i in node.input if !haskey(initializer_map,i)]
        try
            return (@match node.op_type begin
                "Sub" => construct_layer_sub
                "Add" => construct_layer_add
                "MatMul" => construct_layer_matmul
                "Relu" => construct_layer_relu
                "Sigmoid" => construct_layer_sigmoid
                "Tanh" => construct_layer_tanh
                "Gemm" => construct_layer_gemm
                "Flatten" => construct_layer_flatten
                "Constant" => construct_layer_constant
                "Reshape" => construct_layer_reshape
                "Split" => construct_layer_split
                "Slice" => construct_layer_slice
                "Gather" => construct_layer_gather
                "Conv" => construct_layer_conv
                "Concat" => construct_layer_concat
                "Mul" => construct_layer_mul
                "ReduceSum" => construct_layer_reducesum
                "Div" => construct_layer_div
                "Pow" => construct_layer_pow
                "BatchNormalization" => construct_layer_batch_normalization
                "ConvTranspose" => construct_layer_conv_transpose
                "Dropout" => construct_layer_dropout
                "Upsample" => construct_layer_upsample
                _ => error("Unknown operation $(node.op_type)")
            end)(net_type, node.name, inputs, node.output, layer_inputs...;params...)
        catch e
            error("Error while processing node $(node.name) of type $(node.op_type): $(e)\nFull Node: $(node)")
        end
    end

    #export Network, Layer, Dense, ReLU
    export load_network_dict, VNNLibNetworkConstructor
end