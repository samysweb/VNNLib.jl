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
            

    function load_network_dict(net_type::Type{<:NetworkType},filename::String)
        onnx_proto_model = open(filename,"r") do f
            input = ProtoDecoder(f)
            return decode(input, onnx.ModelProto)
        end
        graph = onnx_proto_model.graph

        initializer_map = Dict(i.name => i for i in graph.initializer)

        node_map = Dict()
        for node in graph.node
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
        params = []
        for input in node.input
            if haskey(initializer_map,input)
                push!(params,
                    tensor_to_array(initializer_map[input])
                )
            end
        end
        inputs = [i for i in node.input if !haskey(initializer_map,i)]
        @match node.op_type begin
            "Sub" => return construct_layer_sub(net_type, node.name, inputs, node.output, params...)
            "Add" => return construct_layer_add(net_type, node.name, inputs, node.output, params...)
            "MatMul" => return construct_layer_matmul(net_type, node.name, inputs, node.output, params...)
            "Relu" => return construct_layer_relu(net_type, node.name, inputs, node.output, params...)
            "Gemm" => return construct_layer_gemm(net_type, node.name, inputs, node.output, params...)
            "Flatten" => return construct_layer_flatten(net_type, node.name, inputs, node.output, params...)
            _ => error("Unknown operation $(node.op_type)")
        end
    end

    export Network, Layer, Dense, ReLU
    export load_network_dict
end