
using VNNLib.OnnxSimplifier


SimpleABSTOLERANCE = 1e-6


"""
Compare difference between dense model and original onnx file.

Important because we incur floating point error when 
1) converting from .onnx to OnnxNet (with same operators) and
2) converting from OnnxNet (with same operators) to dense OnnxNet 
"""
function test_onnx_vs_dense(model_path)
    model = load_model(model_path)
    input_data = create_random_inputs(model)
    input_data_flat = Dict(keys(model.input_shapes) .=> [vec(v) for v in values(input_data)])

    model_dense = net2dense(model, input_data)

    outputs_oxp = onnx_parser_forward(model_dense, input_data_flat)
    outputs_ox  = onnx_runtime_forward(model_path, input_data)

    max_diff_all = 0
    correct_all = true
    for on in keys(outputs_oxp)
        max_diff = maximum(abs.(vec(outputs_ox[on]) .- outputs_oxp[on]))
        max_diff_all = max(max_diff_all, max_diff)

        res = all(isapprox.(outputs_oxp[on],vec(outputs_ox[on]);atol=SimpleABSTOLERANCE,rtol=sqrt(eps(Float32))))
        # res = (vec(outputs_ox[on]) ≈ outputs_oxp[on])
        correct_all = res & correct_all

        @test size(vec(outputs_ox[on])) == size(outputs_oxp[on])

        if !res
            println(on, ": ", res)
            println("\tmax diff: ", max_diff)
        end
    end

    println("\tmaximum error onnx vs dense: ", max_diff_all)
    
    @test correct_all
end


"""
Compare OnnxNet with same operators as original .onnx file to results for dense OnnxNet.

Here we can compare the results for all intermediate layers!
"""
function test_original_vs_dense(model_path; verbosity=0)
    model = load_model(model_path)
    input_data = Dict(keys(model.input_shapes) .=> [randn(v...) for v in values(model.input_shapes)])
    input_data_flat = Dict(keys(model.input_shapes) .=> [vec(v) for v in values(input_data)])

    model_dense = net2dense(model, input_data, verbosity=verbosity)

    y       = compute_all_outputs(model, input_data)
    y_dense = compute_all_outputs(model_dense, input_data_flat)


    max_diff_all = 0
    correct_all = true
    for on in keys(y)
        max_diff = maximum(abs.(vec(y[on]) .- y_dense[on]))
        max_diff_all = max(max_diff_all, max_diff)

        res = all(isapprox.(y_dense[on], vec(y[on]);atol=SimpleABSTOLERANCE,rtol=sqrt(eps(Float32))))
        # res = (vec(y[on]) ≈ y_dense[on])
        correct_all = res & correct_all

        @test size(vec(y[on])) == size(y_dense[on])

        if !res
            println(on, ": ", res)
            println("\tmax diff: ", max_diff)
        end
    end

    println("\tmaximum error model vs dense: ", max_diff_all)

    @test correct_all
end


@testset "OnnxSimplifier.jl" verbose=true begin
    for file in readdir(joinpath(@__DIR__, "../../../resources/full_network_tests"))
        @info "Testing network $file"
        model_path = joinpath(@__DIR__, "../../../resources/full_network_tests", file)

        try
            test_original_vs_dense(model_path)
            test_onnx_vs_dense(model_path)
        catch e 
            @error "Test failed for file $file: $e"
            @test false
        end
    end
end