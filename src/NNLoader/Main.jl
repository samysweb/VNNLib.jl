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


    function get_input_shape(input_node::ValueInfoProto)   
        tensor_shape_proto_dims = input_node.var"#type".value.value.shape.dim
        return extract_shape(tensor_shape_proto_dims)
    end


    function get_output_shape(output_node::ValueInfoProto)
        tensor_shape_proto_dims = output_node.var"#type".value.value.shape.dim
        return extract_shape(tensor_shape_proto_dims)
    end
            

    """
    Loads a network from an ONNX file. Parsing is done according to the provided `NetworkType`.

    The ONNX file is parsed according to `construct_layer_{layer_name}(::Type{NetworkType}, ...)` functions defined for this type.

    args:
    - `net_type::Type{<:NetworkType}`: Type of the network to construct, must be a subtype of `NetworkType`.
    - `filename::String`: Path to the ONNX file.

    kwargs:
    - `return_graph::Bool`: If true, returns the ONNX graph instead of the constructed network.
    - `verbosity::Int`: Level of verbosity for logging. Default is 0 (no logging).

    returns:
    - `Network`: A constructed network of type `net_type` with inputs, outputs, and nodes.
    """
    function load_network_dict(net_type::Type{<:NetworkType},filename::String;return_graph=false, verbosity=0)
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
                #println("Skipping constant")
                continue
            end
            # print(keys(node_map))
            name = node.name
            if isempty(name) || haskey(node_map, name)
                # if the node name is empty or already exists, we generate a new name
                name = "node_$(length(node_map)+1)"
            end
            node_map[name] = process_graph_node(net_type, node, name, initializer_map, verbosity=verbosity)
        end
        all_inputs = [i.name for i in graph.input]
        inputs = [i.name for i in graph.input if !haskey(initializer_map,i.name)]
        outputs = [o.name for o in graph.output if !in(o.name, all_inputs)]

        input_shapes = [get_input_shape(i) for i in graph.input if !haskey(initializer_map,i.name)]
        output_shapes = [get_output_shape(o) for o in graph.output if !in(o.name, all_inputs)]

        input_shapes_dict = Dict(inputs .=> input_shapes)
        output_shapes_dict = Dict(outputs .=> output_shapes)

        if verbosity > 0
            println("Network inputs: ", inputs, " with shapes: ", input_shapes)
            println("Network outputs: ", outputs, " with shapes: ", output_shapes)
        end

        return construct_network(net_type, inputs, outputs, node_map, input_shapes_dict, output_shapes_dict)
    end

    function process_graph_node(net_type::Type{<:NetworkType}, node :: NodeProto, name :: String, initializer_map; verbosity=0)
        layer_inputs = []
        for input in node.input
            if haskey(initializer_map,input)
                try
                    push!(layer_inputs,
                        tensor_to_array(initializer_map[input])
                    )
                catch e
                    display(stacktrace(catch_backtrace()))
                    error("Error while processing initializer $input:\n$e\nfull initializer: $(initializer_map[input])")
                end
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
                value = String(attribute.s)  # attribute.s is a byte array, convert to String
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
        # inputs should only represent the inputs that are fed through the layer at runtime!
        # ONNX also defines some weights as inputs, we store them in layer_inputs as they are fixed at runtime (their values are stored in the initializer_map)
        inputs = [i for i in node.input if !haskey(initializer_map,i)]
        verbosity > 0 && println("node.name: $(name), inputs: $inputs, node.output: $(node.output)\n\tlayer_inputs: (len $(length(layer_inputs))) $(layer_inputs)\n\tparams: $(params)")
        try
            return (@match node.op_type begin
                "Sub" => construct_layer_sub
                "Add" => construct_layer_add
                "MatMul" => construct_layer_matmul
                "Relu" => construct_layer_relu
                "LeakyRelu" => construct_layer_leaky_relu
                "Sigmoid" => construct_layer_sigmoid
                "Tanh" => construct_layer_tanh
                "Sign" => construct_layer_sign
                "Softmax" => construct_layer_softmax
                "Floor" => construct_layer_floor
                "Sin" => construct_layer_sin
                "Cos" => construct_layer_cos
                "Sqrt" => construct_layer_sqrt
                "Exp" => construct_layer_exp
                "Elu" => construct_layer_elu
                "Gelu" => construct_layer_gelu
                "Abs" => construct_layer_abs
                "Acos" => construct_layer_acos
                "HardSigmoid" => construct_layer_hard_sigmoid
                "HardSwish" => construct_layer_hard_swish
                "Gemm" => construct_layer_gemm
                "Flatten" => construct_layer_flatten
                "Constant" => construct_layer_constant
                "Reshape" => construct_layer_reshape
                "Transpose" => construct_layer_transpose
                "Split" => construct_layer_split
                "Slice" => construct_layer_slice
                "Gather" => construct_layer_gather
                "Squeeze" => construct_layer_squeeze
                "Unsqueeze" => construct_layer_unsqueeze
                "Pad" => construct_layer_pad
                "Conv" => construct_layer_conv
                "AveragePool" => construct_layer_average_pool
                "MaxPool" => construct_layer_max_pool
                "Concat" => construct_layer_concat
                "Mul" => construct_layer_mul
                "Neg" => construct_layer_neg
                "ReduceSum" => construct_layer_reducesum
                "Div" => construct_layer_div
                "Pow" => construct_layer_pow
                "BatchNormalization" => construct_layer_batch_normalization
                "ConvTranspose" => construct_layer_conv_transpose
                "LSTM" => construct_layer_lstm
                "Dropout" => construct_layer_dropout
                "Upsample" => construct_layer_upsample
                "Resize" => construct_layer_resize
                _ => error("Unknown operation $(node.op_type)")
            end)(net_type, name, inputs, node.output, layer_inputs...;params...)
        catch e
            display(stacktrace(catch_backtrace()))
            error("Error while processing node $(name) of type $(node.op_type): $(e)\nFull Node: $(node)")
        end
    end

    #export Network, Layer, Dense, ReLU
    export load_network_dict, VNNLibNetworkConstructor
end