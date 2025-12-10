
function test_multi_input_multi_output()
    @info "Testing model with multiple inputs and outputs"
    model_path = joinpath(@__DIR__, "../../../resources/small_onnx_tests/add12_mul12.onnx")
    compare_model_file(model_path)
end


function test_individual_nodes()
    @info "Testing individual ONNX nodes"
    # Just walk through all the ONNX files in the directory and see if they can be parsed and execute correctly
    for file in readdir(joinpath(@__DIR__, "../../../resources/small_onnx_tests/individual_nodes"))
        @info "Testing individual node: $file"
        model_path = joinpath(@__DIR__, "../../../resources/small_onnx_tests/individual_nodes", file)
        
        try
            compare_model_file(model_path)
        catch e 
            @error "Test failed for file $file: $e"
            @test false
        end
    end
end



@testset "OnnxParser.jl" verbose=true begin
    test_multi_input_multi_output()
    test_individual_nodes()
end
