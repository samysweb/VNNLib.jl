import numpy as np
import onnx
from onnx import numpy_helper

# Simple reader for onnx files
# slightly modified from https://github.com/stanleybak/nnenum/blob/master/src/nnenum/onnx_network.py


def find_node_with_input(graph, input_name):
    result_node = None
    for n in graph.node:
        for i in n.input:
            if i == input_name:
                assert result_node is None, f"multiple nodes ({result_node}, {n}) accept input {input_name}"
                result_node = n
                
    return result_node


def ensure_one_dim(bias):
    if len([d for d in bias.shape if d > 1]) > 1:
        # haven't implemented convolutions yet
        raise ValueError('Bias has more than one dimension!: {bias.shape}')
    else:
        if len(bias.shape) > 1:
            print('Flattening bias')
            
        bias = bias.squeeze()
        
    return bias


"""
    Produces list of layers of structure (weight, bias, activation).
    
    This is achieved by merging the weights and biases of consecutive linear layers.
    Whenever a non-linear activation layer is encountered, the combined weights and biases along with 
    that activation are stored in the list of layers.
"""
def merge_layers(layer_list):
    weight = None
    bias = None
    
    layers = []
    for w, b, op in layer_list:
        if w is not None:
            if weight is None:
                weight = w
            else:
                weight = w @ weight
                
            if bias is not None:
                bias = w @ bias
                
        if b is not None:
            if bias is None:
                bias = b
            else:
                bias = bias + b
                
        if op in ['Relu']:
            layers.append((weight, bias, op))
            weight = None
            bias = None
            
    if weight is not None and bias is not None:
        layers.append((weight, bias, 'linear'))
    elif weight is not None and bias is None:
        layers.append((weight, np.zeros(weight.shape[0]), 'linear'))
    elif weight is None and bias is not None:
        layers.append((np.eye(len(bias)), bias, 'linear'))
            
    return layers


class OnnxReader:

    def __init__(self, filename):
        self.layers = []
        self.filename = filename

    

    def load_onnx_network(self):
        model = onnx.load(self.filename)
        onnx.checker.check_model(model)

        all_input_names = sum([[str(i) for i in n.input] for n in model.graph.node], [])
        all_output_names = sum([[str(o) for o in n.output] for n in model.graph.node], [])
        all_initializer_names = [i.name for i in model.graph.initializer]

        network_input = None
        for i in all_input_names:
            # why can we find the input like this?
            if i not in all_initializer_names and i not in all_output_names:
                assert network_input is None, f"multiple inputs {network_input} and {i}"
                network_input = i
                
        assert network_input, "did not find network input"
        assert len(model.graph.output) == 1, "Network defines multiple outputs!"

        input_map = {i.name : i for i in model.graph.input}
        init_map  = {i.name : i for i in model.graph.initializer}

        cur_node = find_node_with_input(model.graph, network_input)
        cur_input_name = network_input
        cur_input = input_map[cur_input_name]

        # list containing tuples (W, b, type), where type is e.g. 'ReLU', 'Add', ...
        layers = []

        while cur_node is not None:
            # for nn4sys benchmark, input is at input[1]!
            assert cur_input_name in cur_node.input, f"input of current node {cur_node} should be previous output {cur_input}, but is {cur_node.input[0]}!"
            #assert cur_node.input[0] == cur_input_name, f"input of current node {cur_node} should be previous output {cur_input}, but is {cur_node.input[0]}!"
            
            op = cur_node.op_type
            
            layer = None
            
            if op in ['Add', 'Sub']:
                assert len(cur_node.input) == 2
                init = init_map[cur_node.input[1]]  # can we factor out init = ... (does it occur in every case?)
                bias = numpy_helper.to_array(init)
                
                bias = ensure_one_dim(bias)
                    
                if op == 'Sub':
                    bias = -bias
                    
                layers.append((None, bias, 'Add'))
            elif op == 'Flatten':
                # if only one dimension is greater than one, we can ignore all other dimensions
                # !!! don't use network input, but the input of the layer !!!
                # why is input of the layer not in input_map?
                if len([d.dim_value for d in input_map[find_node_with_input(model.graph, network_input).input[0]].type.tensor_type.shape.dim if d.dim_value > 1]) > 1:
                #if len([d.dim_value for d in input_map[cur_node.input[0]].type.tensor_type.shape.dim if d.dim_value > 1]) > 1:
                    raise ValueError('Flatten layer is not implemented yet!')
                else:
                    print("Safely ignoring Flattening layer.")
                    layers.append((None, None, 'Flatten'))
            elif op == 'MatMul':
                assert len(cur_node.input) == 2
                
                if cur_node.input[0] in init_map:
                    # we have MatMul(W, x) -> y = Wx
                    # therefore we don't need to transpose the weights
                    init = init_map[cur_node.input[0]]
                    weight = numpy_helper.to_array(init)
                elif cur_node.input[1] in init_map:
                    init = init_map[cur_node.input[1]]
                    # MatMul(x, W) -> y = xW
                    # transpose weights, s.t. we have y = Wx + b instead of xW + b
                    weight = numpy_helper.to_array(init).T
                    
                layers.append((weight, None, 'MatMul'))          
            elif op == 'Relu':
                assert layers, "Expected layers before activation function"
                layers.append((None, None, 'Relu'))
            elif op == 'Gemm':
                assert len(cur_node.input) == 3
                weight_init = init_map[cur_node.input[1]]
                bias_init   = init_map[cur_node.input[2]]
                
                weight = numpy_helper.to_array(weight_init)
                bias   = numpy_helper.to_array(bias_init)
                
                bias = ensure_one_dim(bias)
                
                layers.append((weight, bias, 'Gemm'))
            else:
                raise ValueError(f'Unsupported operation {op}!')
                
            
            assert len(cur_node.output) == 1, f"multiple outputs for node {cur_node}: {cur_node.output}!"
            
            cur_input_name = cur_node.output[0]
            cur_node = find_node_with_input(model.graph, cur_input_name)

        self.layers = merge_layers(layers)

    def print_layers(self):
        assert self.layers, "load_onnx_network() has to be executed first!"
        for i, l in enumerate(self.layers):
            w, b, activation = l
            print(f'{i}: {activation} -> w={w.shape}, b={b.shape}')

    def get_weights_and_biases(self):
        assert self.layers, "load_onnx_network() has to be executed first!"
        weights = []
        biases = []
        for w, b, _ in self.layers:
            weights.append(w)
            biases.append(b)

        return weights, biases