module NNLoader
    include("onnx/onnx.jl")

    using LinearAlgebra
    using ProtoBuf

    using .onnx

    include("NPHelper.jl")
    include("Definition.jl")
    include("Layers_Vector.jl")

    function ensure_one_dim(bias)
        if length(filter(x->x>1,size(bias))) > 1
            error("Bias has more than one dimension")
        else
            if length(size(bias))>1
                bias = reshape(bias,:)
            end
        end
        return bias
    end

    function create_network(layers)
        next_bias = nothing
        next_weight = nothing

        output_layers = []
        for (op,weight,bias) in layers
            println("op: $op, weight: $(isnothing(weight) ? nothing : size(weight)), bias: $(isnothing(bias) ? nothing : size(bias))")
            if op == :bias
                @assert isnothing(weight) "Weight is nothing for bias layer"
                @assert !isnothing(bias) "Bias is not nothing for bias layer"
                if !isnothing(next_bias)
                    next_bias = next_bias .+ bias
                else
                    next_bias = bias
                end
            elseif op == :matmul
                @assert !isnothing(weight) "Weight is not nothing for matmul layer"
                @assert isnothing(bias) "Bias is nothing for matmul layer"
                if !isnothing(next_bias)
                    next_bias = weight * next_bias
                end
                if !isnothing(next_weight)
                    next_weight = weight * next_weight
                else
                    next_weight = weight
                end
            elseif op == :relu
                @assert isnothing(weight) "Weight is nothing for relu layer"
                @assert isnothing(bias) "Bias is nothing for relu layer"
                if !isnothing(next_weight)
                    if !isnothing(next_bias)
                        @assert size(next_weight,1) == length(next_bias) "Weight and bias have different sizes"
                        push!(output_layers,Dense(next_weight,next_bias))
                    else
                        push!(output_layers,Dense(next_weight,zeros(size(next_weight,1))))
                    end
                elseif !isnothing(next_bias)
                    next_weight = Matrix(1.0I,size(next_bias,1),size(next_bias,1))
                    push!(output_layers,Dense(next_weight,next_bias))
                end
                push!(output_layers,ReLU())
                next_bias = nothing
                next_weight = nothing
            elseif op == :gemm
                if !isnothing(next_weight)
                    if !isnothing(next_bias)
                        @assert size(next_weight,1) == length(next_bias) "Weight and bias have different sizes"
                        push!(output_layers,Dense(next_weight,next_bias))
                    else
                        push!(output_layers,Dense(next_weight,zeros(size(next_weight,1))))
                    end
                elseif !isnothing(next_bias)
                    next_weight = Matrix(1.0I,size(next_bias,1),size(next_bias,1))
                    push!(output_layers,Dense(next_weight,next_bias))
                end
                push!(output_layers,Dense(weight,bias))
                next_bias = nothing
                next_weight = nothing
            elseif op == :flatten
                @warn "Flatten layer is not supported"
                if !isnothing(next_weight)
                    if !isnothing(next_bias)
                        @assert size(next_weight,1) == length(next_bias) "Weight and bias have different sizes"
                        push!(output_layers,Dense(next_weight,next_bias))
                    else
                        push!(output_layers,Dense(next_weight,zeros(size(next_weight,1))))
                    end
                elseif !isnothing(next_bias)
                    next_weight = Matrix(1.0I,size(next_bias,1),size(next_bias,1))
                    push!(output_layers,Dense(next_weight,next_bias))
                end
                next_bias = nothing
                next_weight = nothing
            else
                error("Unknown operation $op")
            end
        end
        if !isnothing(next_weight)
            if !isnothing(next_bias)
                @assert size(next_weight,1) == length(next_bias) "Weight and bias have different sizes"
                push!(output_layers,Dense(next_weight,next_bias))
            else
                push!(output_layers,Dense(next_weight,zeros(size(next_weight,1))))
            end
        elseif !isnothing(next_bias)
            next_weight = Matrix(1.0I,size(next_bias,1),size(next_bias,1))
            push!(output_layers,Dense(next_weight,next_bias))
        end
        return Network(output_layers)
    end
            

    function load_network(filename::String)
        onnx_proto_model = open(filename,"r") do f
            input = ProtoDecoder(f)
            return decode(input, onnx.ModelProto)
        end
        graph = onnx_proto_model.graph

        all_input_names = reduce(vcat,[[i for i in n.input] for n in graph.node])
        all_output_names = reduce(vcat,[[i for i in n.output] for n in graph.node])
        all_initializer_names = [i.name for i in graph.initializer]

        network_input = nothing
        for i in all_input_names
            if i ∉ all_initializer_names && i ∉ all_output_names
                @assert isnothing(network_input) "Multiple network inputs found: $network_input and $i"
                println("Found input $i")
                network_input = i
                # break
            end
        end
        input_map = Dict(i.name => i for i in graph.input)
        init_map = Dict(i.name => i for i in graph.initializer)
        node_by_input = Dict(i => n for n in graph.node for i in n.input)

        cur_node = node_by_input[network_input]
        cur_input_name = network_input
        cur_input = input_map[cur_input_name]

        layers = []

        while !isnothing(cur_node)
            @assert cur_input_name in cur_node.input "Node $cur_node does not have input $cur_input_name"

            op = cur_node.op_type
            
            if op == "Add" || op == "Sub"
                @assert length(cur_node.input) == 2 "Add/Sub node $cur_node has more than 2 inputs"
                init = init_map[cur_node.input[2]]
                bias = tensor_to_array(init)
                bias = ensure_one_dim(bias)
                if op == "Sub"
                    bias = -bias
                end
                push!(layers,(:bias,nothing,bias))
            elseif op == "Flatten"
                push!(layers,(:flatten,nothing,nothing))
            elseif op == "MatMul"
                if haskey(init_map,cur_node.input[1])
                    init = init_map[cur_node.input[1]]
                    weight = tensor_to_array(init)
                    push!(layers,(:matmul,weight',nothing))
                else
                    init = init_map[cur_node.input[2]]
                    weight = tensor_to_array(init)
                    push!(layers,(:matmul,weight,nothing))
                end
            elseif op == "Relu"
                push!(layers,(:relu,nothing,nothing))
            elseif op == "Gemm"
                @assert length(cur_node.input) == 3 "Gemm node $cur_node has more than 3 inputs"
                if haskey(init_map, cur_node.input[1])
                    init = init_map[cur_node.input[1]]
                    weight = tensor_to_array(init)
                    init = init_map[cur_node.input[2]]
                    bias = tensor_to_array(init)
                    bias = ensure_one_dim(bias)
                    push!(layers,(:gemm,weight,bias))
                else
                    init = init_map[cur_node.input[2]]
                    weight = tensor_to_array(init)
                    init = init_map[cur_node.input[3]]
                    bias = tensor_to_array(init)
                    bias = ensure_one_dim(bias)
                    push!(layers,(:gemm,weight',bias))
                end
            else
                println("WARNING: Operation $op is not supported")
                println("WE ARE IGNORING THIS OPERATION -- IF YOU DO NOT KNOW WHAT YOU ARE DOING THIS MAY BE UNSOUND!")
            end
            @assert length(cur_node.output) == 1 "Node $cur_node has more than one output"
            cur_input_name = cur_node.output[1]
            cur_input = cur_node
            cur_node = get(node_by_input,cur_input_name,nothing)
        end

        network = create_network(layers)
        
        return network
    end

    export Network, Layer, Dense, ReLU
    export load_network
end