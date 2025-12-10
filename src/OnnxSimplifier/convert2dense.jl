
function node2dense(net::OnnxNet{S,N1,N2}, node_name::S, out_dict::Dict{S,AN}; verbosity=0) where {S,N1,N2,AN}
    L = net.nodes[node_name]
    xs = OXP.collect_inputs(net, node_name, out_dict)

    verbosity > 0 && println("[node2dense] input sizes = ", [size(x) for x in xs])

    if length(L.outputs) <= 1
        A, b = affop2mat(OXP.onnx_node_to_flux_layer(L), xs..., batched=batched_linearization(L))

        if length(xs) > 1
            # For multiple inputs, we first need to concatenate them to a large vector.
            # Then we make sure that the [Concat, Linear] block has the same inputs as the original layer 
            # and returns the same outputs.

            # TODO: make sure that we get a unique name!!!
            concat_outs = [L.name * "_input_concat"]
            concat_layer = OXP.ONNXConcat(L.inputs, concat_outs, L.name * "_concat", 0)
            linear_layer = OXP.ONNXLinear(concat_outs, L.outputs, L.name, A, b, transpose=false)
            nodes = [concat_layer, linear_layer]
        else 
            linear_layer = OXP.ONNXLinear(L.inputs, L.outputs, L.name, A, b, transpose=false)
            nodes = [linear_layer]
        end
    else 
        # need one Aᵢ, bᵢ for every output
        @assert length(xs) == 1 "Nodes with multiple inputs and multiple outputs are not supported yet!"

        params = [affop2mat(x -> OXP.onnx_node_to_flux_layer(L)(x)[i], xs..., batched=batched_linearization(L)) for i in 1:length(L.outputs)]
        As = first.(params)
        bs = last.(params)

        nodes = [OXP.ONNXLinear(L.inputs, [o], L.name * "_$i", A, b, transpose=false) for (i, (o, A, b)) in enumerate(zip(L.outputs, As, bs))]
    end

    nodes
end


"""
Handle broadcasting of inputs to nonlinear nodes.
"""
function process_nonlinear(net::OnnxNet{S,N1,N2}, node_name::S, out_dict::Dict{S,AN}; verbosity=0) where {S,N1,N2,AN}
    L = net.nodes[node_name]
    xs = OXP.collect_inputs(net, node_name, out_dict)
    shapes = [size(x) for x in xs]
    
    verbosity > 0 && println("[process_nonlinear] input sizes = ", shapes)

    # need to wrap shapes[1] in a separate tuple. Otherwise Julia wants to compare its elements to each 
    # element of the shapes vector.
    same_shapes = all(shapes .== (shapes[1],))

    if same_shapes
        # don't have to worry about broadcasting, just return the nonlinear node
        nodes = [deepcopy(L)]
    elseif length(xs) > 2
        @assert false "For nonlinear nodes with > 2 inputs, we currently don't resolve broadcasting! Got shapes $shapes"
    elseif length(xs) == 2
        A, B = broadcast_to_mat(xs[1], xs[2])

        node = deepcopy(L)
        nodes = Vector{OXP.Node{S}}()
        if A != I 
            # only add a linear layer, if A is not the identity matrix 
            intermediate1 = L.inputs[1] * "_broadcast_resolved"
            # TODO: do we really need Float64 here?
            push!(nodes, OXP.ONNXLinear([L.inputs[1]], [intermediate1], L.name * "_broadcast_resolve_1", Flux.Dense(Float64.(A)), false))
            node.inputs[1] = intermediate1
        end

        if B != I 
            # only add a linear layer, if B is not the identity matrix 
            intermediate2 = L.inputs[2] * "_broadcast_resolved"
            # TODO: do we really need Float64 here?
            push!(nodes, OXP.ONNXLinear([L.inputs[2]], [intermediate2], L.name * "_broadcast_resolve_2", Flux.Dense(Float64.(B)), false))
            node.inputs[2] = intermediate2
        end

        push!(nodes, node)
    else
        @assert false "There is a node with a single input and unqual input shapes. That should not happen!"
    end

    return nodes 
end


"""
Converts OnnxNet to a computational graph that only contains non-linear operations, ONNXLinear and ONNXConcat nodes.

Note that while the network architecture is much simpler, we may loose efficiency both in terms of runtime and memory.

We guarantee (up to floating point accuracy) that the outputs are equal:

    vec(net(x)) == net_dense(vec(x))

args:
    net - original network 
    x_dict - dictionary mapping input names to concrete inputs for the original network 

kwargs:
    verbosity - silent if verbosity == 0 (default 0)

returns:
    net_dense - converted original network
"""
function net2dense(net::OnnxNet{S,N1,N2}, x_dict::Dict{S,<:AbstractArray}; verbosity=0) where {S,N1,N2}
    # TODO: we really only want to compute the shapes of the output in the future.
    #       with the current approach we might need lots of memory
    y_dict = OXP.compute_all_outputs(net, x_dict)

    nodes_dense = Dict{S, OXP.Node{S}}()
    for (name, node) in net.nodes 
        verbosity > 0 && println("[net2dense] converting node $name")
        if OXP.islinear(node)
            nodes = node2dense(net, name, y_dict, verbosity=verbosity-1)

        else
            nodes = process_nonlinear(net, name, y_dict, verbosity=verbosity-1)
        end

        for n in nodes
            nodes_dense[n.name] = n
        end
    end

    # at least shapes should be different! i.e. they should be flattened
    flat_input_shapes  = Dict(keys(net.input_shapes) .=> [(prod(v),) for v in values(net.input_shapes)])
    flat_output_shapes = Dict(keys(net.output_shapes) .=> [(prod(v),) for v in values(net.output_shapes)])
    return OnnxNet(values(nodes_dense), net.start_nodes, net.final_nodes, flat_input_shapes, flat_output_shapes)
end