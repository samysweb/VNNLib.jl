# VNNLib.jl

[![Run tests](https://github.com/samysweb/VNNLib.jl/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/samysweb/VNNLib.jl/actions/workflows/ci.yml)
[![codecov](https://codecov.io/github/samysweb/VNNLib.jl/branch/main/graph/badge.svg?token=G23F6Z1LH3)](https://codecov.io/github/samysweb/VNNLib.jl)

This library helps with the parsing and processing of properties in VNN Lib and neural networks in ONNX format.

# VNNLib Properties
```julia
using VNNLib

# Load VNNLib property as an AST
f, n_input, n_output = get_ast("test/examples/acas2.vnn")

# n_input: Number of input variables
# n_output: Number of output variables

# Iterate over linear constraints encoded by acas2.vnn
for (bounds, matrix, bias, num) in f
    # Process linear constraints
    # bounds: variable bounds
    # matrix: constraint matrix M
    # bias: constraint biases b
    # num: Number of rows with input constraints (1..num are input constraints, remaining constraints are mixed/output constraints)
    # Overall this encodes Mx <= b
end
```
**Warning:** The for loop is currently very slow for large VNN files containing large numbers of disjunctions, e.g. `test/examples/nn4sys.vnn`

## Advanced usage
There is multiple options for advanced usage

### Variable naming
By default `get_ast` will use the variable names `X_N` for input variables and `Y_N` for output variables.
In order to encode a different scheme, you may pass a variable labeling function to `get_ast` as follows:
```julia
function my_labeler(name)
    if name == "in1"
        return AST.Input, (1,)
    elseif name == "in2"
        return AST.Input, (2,)
    else
        return AST.Output, (1)
    end
end

function get_ast("test/examples/acas2.vnn",variable_labeler=my_labeler)
```

### Simplification using SymbolicUtils
The AST encoding of VNNLib supports processing through `SymbolicUtils.jl`.
To this end, we currently support the following simplification functions:
- `prepare_linearization`: bring a conjunction of atoms into the form processable by `ast_to_lp`
- `to_dnf`: convert formula into disjunctive normal form (**Warning:** Currently this is very slow for large formulas)

### Linearization of custom formulas
In order to obtain an LP representation a formula must be a conjunction over atoms.
Once this is the case, you can use `ast_to_lp` to obtain the LP representation from above.

### Structure of formulas
The structure of formulas is described through the AST definitions in `src/AST/Definition.jl`


# ONNX Loader


```Julia
using VNNLib

model = load_onnx_model("./resources/small_onnx_tests/add_2_inputs.onnx")

model.input_shapes()
# Dict{String, Tuple{Int64, Int64}} with 2 entries:
#  "input2" => (5, 1)
#  "input1" => (5, 1)

model.output_shapes()
#Dict{String, Tuple{Int64, Int64}} with 1 entry:
#  "output" => (5, 1)

input_data = Dict("input1" => randn(5,1), "input2" => randn(5,1))
# also works with an array, if the model has a single input
compute_outputs(model, input_data)  
# Dict{String, Matrix{Float64}} with 1 entry:
#  "output" => [1.73978; -0.782722; … ; -2.16249; -0.83229;;]

compute_output(model, input_data)  # if the model has a single output
```

## Adding a new Operator

To add support for a new ONNX operator, you need to define the following structs and functions:
```Julia

struct MyNode{S} <: Node{S}
    inputs::AbstractVector{S}   # names of inputs intermediate results
    outputs::AbstractVector{S}  # names of output intermediate results
    name::S
    # whatever other attributes you need
end

# return a Julia function that computes the result of the node
# given its input
onnx_node_to_flux_layer(node::MyNode) = x -> ...

function NNL.construct_layer_my_node(::Type{OnnxType}, name, inputs, outputs, <list of inputs>; <list of attributes>)
    # <list of inputs>: one parameter for each input listed 
    #                   in the ONNX standard
    #                   Julia also allows to set default values for
    #                   positional arguments. These can be used for
    #                   optional ONNX inputs.
    # <list of attributes>: one keyword argument for each attribute 
    #                   listed in the ONNX standard

    # Construct an instance of MyNode
end
```

Have a look at the definitions of nodes in `src/OnnxParser/{indexing.jl|linear.jl|nonlinear.jl}` for guidance.

You also need to register your `construct_layer_my_node` implementation in the large case distinction in `src/NNLoader/Main.jl`.
Just add another case
```Julia
"MyNode" => construct_layer_my_node
```



## Writing A Custom ONNX Parser

`VNNLib.jl` provides the `OnnxParser` introduced above, but also allows you to create your own ONNX parser.

To load an ONNX network using a loader type `MyOnnxType`:
```julia
using VNNLib
const NNL = VNNLib.NNLoader

nodes, input_nodes, output_nodes, input_shape, output_shape = NNL.load_network_dict(MyOnnxType, "path/to/model.onnx")
```

There is an implementation for the `NNL.VnnLibNetworkConstructor` type that you can use to load simple networks.