# VNNLib.jl

[![Run tests](https://github.com/samysweb/VNNLib.jl/actions/workflows/ci.yml/badge.svg?branch=test)](https://github.com/samysweb/VNNLib.jl/actions/workflows/ci.yml)
[![codecov](https://codecov.io/github/samysweb/VNNLib.jl/branch/test/graph/badge.svg?token=G23F6Z1LH3)](https://codecov.io/github/samysweb/VNNLib.jl)

This library helps with the parsing and processing of properties in the VNN Lib format.

## Basic usage
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


# Contribute

## Adding a new Operator

To add a new operator, you need to
1. add a new type `struct YourType{T} <: VNNLibLayer{T} ... end` (see the already defined types in `Definition.jl`)
2. add constructors `construct_layer_yourlayer(N, name, inputs, outputs, <list of inputs>; <list of attributes>)`
    1. `<list of inputs>` are the inputs as defined in the onnx specification. They are provided as positional arguments to the constructor. (Remember, Julia also allows to set default values for positional arguments allowing to model the optional inputs in the onnx specification)
    2. `<list of attributes>` are the attributes as defined in the onnx specification. They are provided as **keyword** arguments to the constructor
3. register your constructor in the switch-statement in `Main.jl`.

