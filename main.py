from Example import Example
from NodeType import NodeType
from Weight import Weight
from Node import Node
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    return z * (1 - z)

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

    total_nodes = num_inputs + num_hidden_units + num_outputs

    weights = np.random.uniform(-0.1, 0.1, size=(total_nodes, total_nodes))

    print("[")
    for i in range(0, total_nodes):
        print("[", end="")
        for j in range(0, total_nodes):
            print(weights[i][j], "", end="")
        print("]")
    print("]")
    print("")

    output_nodes = []
    input_nodes = []
    hidden_nodes = []
    learning_rate = 1

    weight_array_layer_1 = []
    weight_array_layer_2 = []
    for i in range(0, num_inputs):
        input_forward_weights = []
        for j in range(0, num_hidden_units):
            input_forward_weights.append(Weight(i, j))
        weight_array_layer_1.append(input_forward_weights)
        input_nodes.append(Node(i, NodeType.INPUT, [], input_forward_weights))

    for k in range(0, num_outputs):
        output_backward_weights = []
        for j in range(0, num_hidden_units):
            output_backward_weights.append(Weight(j, k))
        weight_array_layer_2.append(output_backward_weights)
        output_nodes.append(Node(k, NodeType.OUTPUT, output_backward_weights, []))

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

    # set an example
    # then for each input node, calculate y for each connected hidden unit

    input_nodes[0].output = 0 #replace with actual example
    input_nodes[1].output = 0 #replace with actual example
    target = 0

    for hidden in hidden_nodes:
        weighted_sum = 0
        for weight in hidden.weights_backward:
            weighted_sum += (input_nodes[weight.start_node].output * weights[weight.start_node][weight.end_node])
        hidden.output = sigmoid(weighted_sum)
        hidden.sum_of_weighted_inputs = weighted_sum
        print(hidden.output)

    for output in output_nodes:
        weighted_sum = 0
        for weight in output.weights_backward:
            weighted_sum += (hidden_nodes[weight.start_node].output * weights[weight.start_node][weight.end_node])
        output.output = sigmoid(weighted_sum)
        print(output.output)
        delta = target - output.output
        output.delta = delta
        for weight in output.weights_backward:
            input_node = hidden_nodes[weight.start_node]
            weights[weight.start_node][weight.end_node] = -1 * learning_rate * output.delta * input_node.output

    # backprop
    for hidden in hidden_nodes:
        summer = 0
        for weight in hidden.weights_forward:
            previous_node = output_nodes[weight.end_node]
            print("Value of Z: ", hidden.sum_of_weighted_inputs)
            summer += previous_node.delta * weights[weight.start_node][weight.end_node] * sigmoid_derivative(hidden.sum_of_weighted_inputs)
        hidden.delta = summer
        for weight in hidden.weights_backward:
            input_node = input_nodes[weight.start_node]
            weights[weight.start_node][weight.end_node] = -1 * learning_rate * hidden.delta * input_node.output

    print("[")
    for i in range(0, total_nodes):
        print("[", end="")
        for j in range(0, total_nodes):
            print(weights[i][j], "", end="")
        print("]")
    print("]")
    print("")

    # calculate and save deltas dont have to do for input layer
    # then apply weight change rule
    # update weights by associated weight change


    #what if we made a recursive function, which starts with a node, passes weight and delta (after calculating)
    # of this node to all weights_behind starts.


    # how does target differ from output.output

    # start back prop

if __name__ == "__main__":
    main()
