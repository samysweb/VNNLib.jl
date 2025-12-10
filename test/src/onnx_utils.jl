
using VNNLib
using VNNLib.OnnxParser
using Test
using ONNXRunTime
const OX = ONNXRunTime
const OXP = VNNLib.OnnxParser
const NNL = VNNLib.NNLoader


ABSTOLERANCE = 1e-8


reversedims(A::AbstractArray) = permutedims(A, reverse(tuple(1:ndims(A)...)))


function load_model(model_path::String)
    model = NNL.load_network_dict(OXP.OnnxType, model_path)
    model
end


function create_random_inputs(model::OXP.OnnxNet)
    input_data = Dict{String, AbstractArray}()

    for (k, v) in model.input_shapes
        # Create random input data with the specified shape
        input_data[k] = rand(v...)
    end

    return input_data   
end


function onnx_parser_forward(model::OXP.OnnxNet, input_data::Dict{String, <:AbstractArray})
    OXP.compute_outputs(model, input_data)
end


function onnx_runtime_forward(model_path, input_data::Dict{String, <:AbstractArray})
    model = OX.load_inference(model_path)

    @assert issubset(model.input_names, keys(input_data)) "Input data keys do not match model input names."
    # ONNXRunTime expects inputs to be Float32 and NCHW order
    onnx_inputs = Dict(name => Float32.(reversedims(input_data[name])) for name in keys(input_data))

    outputs = model(onnx_inputs)
    # convert back to WHCN order
    outputs = Dict(name => reversedims(outputs[name]) for name in keys(outputs))
    return outputs
end


function compare_outputs(outputs_oxp::Dict{String,<:AbstractArray}, outputs_ox::AbstractDict{String,<:AbstractArray})
    @test issubset(keys(outputs_oxp), keys(outputs_ox)) #"Outputs keys do not match! Got $(keys(outputs_oxp)) but expected $(keys(outputs_ox))"
    @test issubset(keys(outputs_ox), keys(outputs_oxp)) #"Outputs keys do not match! Got $(keys(outputs_oxp)) but expected $(keys(outputs_ox))"

    for (name, output_oxp) in outputs_oxp
        output_ox = outputs_ox[name]
        @test size(output_oxp) == size(output_ox) #"Output sizes do not match for $name! Got $(size(output_oxp)) but expected $(size(output_ox))"
        @test all(output_oxp .≈ output_ox) #"Outputs do not match for $name! Got $(output_oxp) but expected $(output_ox)"
    end
end


function compare_model_file(model_path)
    model = load_model(model_path)
    input_data = create_random_inputs(model)
    outputs_oxp = onnx_parser_forward(model, input_data)
    outputs_ox = onnx_runtime_forward(model_path, input_data)
    compare_outputs(outputs_oxp, outputs_ox)
end