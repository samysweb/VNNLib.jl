using ProtoBuf


# ,"onnx/onnx-data.proto","onnx/onnx-operators.proto"
protojl(["onnx/onnx.proto"], (@__DIR__)*"/onnx/",(@__DIR__)*"/../src/NNLoader")