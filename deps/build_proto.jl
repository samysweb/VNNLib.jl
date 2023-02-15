using ProtoBuf

# Clone repository
run(`git clone https://github.com/onnx/onnx.git $(@__DIR__)/onnx`)

# ,"onnx/onnx-data.proto","onnx/onnx-operators.proto"
protojl(["onnx/onnx.proto"], (@__DIR__)*"/onnx/",(@__DIR__)*"/../src/NNLoader")

run(`rm -rf $(@__DIR__)/onnx`)