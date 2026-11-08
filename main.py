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

def conversion_scheme(weight, num_inputs, num_hidden_units):
    if weight < num_inputs:
        return weight
    elif num_inputs <= weight < num_inputs + num_hidden_units:
        return weight - num_inputs
    else:
        return weight - num_inputs - num_hidden_units

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

    input_nodes = [Node(i, NodeType.INPUT) for i in range(0, num_inputs)]
    hidden_nodes = [Node(j, NodeType.HIDDEN) for j in range(num_inputs, num_inputs + num_hidden_units)]
    output_nodes = [Node(k, NodeType.OUTPUT) for k in range(num_inputs + num_hidden_units, total_nodes)]

    learning_rate = 1

    weight_array_layer_1 = []
    weight_array_layer_2 = []
    for input_n in input_nodes:
        for hidden_n in hidden_nodes:
            input_n.weights_forward.append(Weight(input_n.id, hidden_n.id))
            hidden_n.weights_backward.append(Weight(input_n.id, hidden_n.id))
        weight_array_layer_1.append(input_n.weights_forward)

    for output_n in output_nodes:
        for hidden_n in hidden_nodes:
            output_n.weights_backward.append(Weight(hidden_n.id, output_n.id))
            hidden_n.weights_forward.append(Weight(hidden_n.id, output_n.id))
        weight_array_layer_2.append(output_n.weights_backward)

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

    #if hidden 2 -> 0 3 -> 1 -> 4 -> 2 5-> 3 (subtract num_inputs)
    #if output 6 -> 0 (subtract num_inputs + num_outputs)

    for hidden in hidden_nodes:
        # for each hidden node get its weights, use these to get nodes somehow from array have a conversion scheme
        weighted_sum = 0
        for weight in hidden.weights_backward:
            weighted_sum += (input_nodes[conversion_scheme(weight.start_node, num_inputs, num_hidden_units)].output * weights[weight.start_node][weight.end_node])
        hidden.output = sigmoid(weighted_sum)
        hidden.sum_of_weighted_inputs = weighted_sum
        print(hidden.output)

    for output in output_nodes:
        weighted_sum = 0
        for weight in output.weights_backward:
            weighted_sum += (hidden_nodes[conversion_scheme(weight.start_node, num_inputs, num_hidden_units)].output * weights[weight.start_node][weight.end_node])
        output.output = sigmoid(weighted_sum)
        print(output.output)
        delta = target - output.output
        output.delta = delta
        for weight in output.weights_backward:
            input_node = hidden_nodes[conversion_scheme(weight.start_node, num_inputs, num_hidden_units)]
            weights[weight.start_node][weight.end_node] = -1 * learning_rate * output.delta * input_node.output

    # backprop
    for hidden in hidden_nodes:
        summer = 0
        for weight in hidden.weights_forward:
            previous_node = output_nodes[conversion_scheme(weight.end_node, num_inputs, num_hidden_units)]
            summer += previous_node.delta * weights[weight.start_node][weight.end_node] * sigmoid_derivative(hidden.sum_of_weighted_inputs)
        hidden.delta = summer
        for weight in hidden.weights_backward:
            input_node = input_nodes[conversion_scheme(weight.start_node, num_inputs, num_hidden_units)]
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
