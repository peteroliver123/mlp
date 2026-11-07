from Example import Example
from NodeType import NodeType
from Weight import Weight
from Node import Node
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def delta_rule(delta_j, x_i):
    learning_rate = 1
    return learning_rate * delta_j * x_i

def fill_examples():
    examples = [
                    Example((0, 0), 0),
                    Example((0, 1), 1),
                    Example((1,0), 1),
                    Example((1,1), 0)
                ]
    return examples

def main():
    print("MLP Project Started!")

    examples = fill_examples()

    num_examples = len(examples)

    num_inputs = 2

    num_outputs = 1

    num_hidden_units = 4

    output_nodes = []
    input_nodes = []
    hidden_nodes = []

    weight_array_layer_1 = []
    weight_array_layer_2 = []
    for i in range(0, num_inputs):
        input_forward_weights = []
        for j in range(0, num_hidden_units):
            weight_value = np.random.uniform(-0.1, 0.1)
            input_forward_weights.append(Weight(i, j, weight_value))
        weight_array_layer_1.append(input_forward_weights)
        input_nodes.append(Node(i, NodeType.INPUT, [], input_forward_weights))

    for k in range(0, num_outputs):
        output_backward_weights = []
        for j in range(0, num_hidden_units):
            weight_value = np.random.uniform(-0.1, 0.1)
            output_backward_weights.append(Weight(j, k, weight_value))
        weight_array_layer_2.append(output_backward_weights)
        output_nodes.append(Node(k, NodeType.OUTPUT, output_backward_weights, []))

    for i in range(0, num_inputs):
        for j in range(0, num_hidden_units):
            weight = weight_array_layer_1[i][j]
            print(weight.start_node, weight.end_node, weight.value)

    for k in range(0, num_outputs):
        for j in range(0, num_hidden_units):
            weight = weight_array_layer_2[k][j]
            print(weight.start_node, weight.end_node, weight.value)

    for j in range(0, num_hidden_units):
        forward_hidden_weights = []
        backward_hidden_weights = []
        for weight_array in weight_array_layer_1:
            for weight in weight_array:
                if weight.end_node == j:
                    backward_hidden_weights.append(weight)
        for weight_array in weight_array_layer_2:
            for weight in weight_array:
                if weight.start_node == j:
                    forward_hidden_weights.append(weight)
        hidden_nodes.append(Node(j, NodeType.HIDDEN, backward_hidden_weights, forward_hidden_weights))

    for inp in input_nodes:
        inp.reveal_node()

    for hid in hidden_nodes:
        hid.reveal_node()

    for out in output_nodes:
        out.reveal_node()

    """for j in range(0, num_hidden_units):
        for i in range(0, num_inputs):
            weight_value = np.random.uniform(-0.1, 0.1)
            hidden_weights_backward.append(Weight(i, j, weight_value))
            weights_forward_input.append(Weight(i, j, weight_value))

        for k in range(0, num_outputs):

            weight_value = np.random.uniform(-0.1, 0.1)
            hidden_weights_forward.append(Weight(j, k, weight_value))
            weights_backward_output.append(Weight(j, k, weight_value))
            output_nodes.append(Node(k, NodeType.OUTPUT, weights_backward_output, []))
        hidden_nodes.append(Node(j, NodeType.HIDDEN, hidden_weights_backward, hidden_weights_forward))


    input_nodes.append(Node(i, NodeType.INPUT, [], weights_forward_input))

    for input_n in input_nodes:
        input_n.reveal_node()
    for hidden_n in hidden_nodes:
        hidden_n.reveal_node()
    for output_n in output_nodes:
        output_n.reveal_node()

    # set an example
    # then for each input node, calculate y for each connected hidden unit

    input_nodes[0].output = 0 #replace with actual example
    input_nodes[1].output = 0 #replace with actual example
    target = 0

    for hidden in hidden_nodes:
        weighted_sum = 0
        for weight in hidden.weights_backward:
            weighted_sum += (input_nodes[weight.start_node].output * weight.value)
        hidden.output = sigmoid(weighted_sum)
        hidden.sum_of_weighted_inputs = weighted_sum
        print(hidden.output)

    for output in output_nodes:
        weighted_sum = 0
        for weight in output.weights_backward:
            weighted_sum += (hidden_nodes[weight.start_node].output * weight.value)
        output.output = sigmoid(weighted_sum)
        print(output.output)
        delta = target - output.output
        output.delta = delta"""


    #what if we made a recursive function, which starts with a node, passes weight and delta (after calculating)
    # of this node to all weights_behind starts.


    # how does target differ from output.output

    # start back prop

if __name__ == "__main__":
    main()
