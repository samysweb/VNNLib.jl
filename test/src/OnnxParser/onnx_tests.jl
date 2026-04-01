
function test_multi_input_multi_output()
    @info "Testing model with multiple inputs and outputs"
    model_path = joinpath(@__DIR__, "../../../resources/small_onnx_tests/add12_mul12.onnx")
    compare_model_file(model_path)
end

function test_acas()
    @info "Testing ACAS model (unnamed nodes)"
    model_path = joinpath(@__DIR__,"../../../resources/small_onnx_tests/ACASXU_run2a_1_1_batch_2000.onnx")
    compare_model_file(model_path)
end

function test_lhc()
    @info "Testing LHC model (integration test)"
    model_path = joinpath(@__DIR__, "../../../resources/small_onnx_tests/2_80-1-0.1.onnx")
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
    test_acas()
    test_lhc()
end
