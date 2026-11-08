from Example import Example
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

    nodes = [Node(i) for i in range(0, total_nodes)]
    #can this be one large nodes array

    learning_rate = 1

    for i in range(0, num_inputs):
        input_n = nodes[i]
        for j in range(num_inputs, num_inputs + num_hidden_units):
            hidden_n = nodes[j]
            input_n.weights_forward.append(Weight(input_n.id, hidden_n.id))
            hidden_n.weights_backward.append(Weight(input_n.id, hidden_n.id))

    for k in range(num_inputs + num_hidden_units, total_nodes):
        output_n = nodes[k]
        for j in range(num_inputs, num_inputs + num_hidden_units):
            hidden_n = nodes[j]
            output_n.weights_backward.append(Weight(hidden_n.id, output_n.id))
            hidden_n.weights_forward.append(Weight(hidden_n.id, output_n.id))

    for i in range(0, num_inputs):
        input_n = nodes[i]
        input_n.reveal_node()

    for j in range(num_inputs, num_inputs + num_hidden_units):
        hidden_n = nodes[j]
        hidden_n.reveal_node()

    for k in range(num_inputs + num_hidden_units, total_nodes):
        output_n = nodes[k]
        output_n.reveal_node()

    # set an example
    # then for each input node, calculate y for each connected hidden unit

    nodes[0].output = 0 #replace with actual example
    nodes[1].output = 0 #replace with actual example
    target = 0

    #if hidden 2 -> 0 3 -> 1 -> 4 -> 2 5-> 3 (subtract num_inputs)
    #if output 6 -> 0 (subtract num_inputs + num_outputs)

    for j in range(num_inputs, num_inputs + num_hidden_units):
        hidden = nodes[j]
        # for each hidden node get its weights, use these to get nodes somehow from array have a conversion scheme
        weighted_sum = 0
        for weight in hidden.weights_backward:
            weighted_sum += (nodes[weight.start_node].output * weights[weight.start_node][weight.end_node])
        hidden.output = sigmoid(weighted_sum)
        hidden.sum_of_weighted_inputs = weighted_sum
        print(hidden.output)

    for k in range(num_inputs + num_hidden_units, total_nodes):
        output = nodes[k]
        weighted_sum = 0
        for weight in output.weights_backward:
            weighted_sum += (nodes[weight.start_node].output * weights[weight.start_node][weight.end_node])
        output.output = sigmoid(weighted_sum)
        print(output.output)
        delta = target - output.output
        output.delta = delta
        for weight in output.weights_backward:
            input_node = nodes[weight.start_node]
            weights[weight.start_node][weight.end_node] += -1 * learning_rate * output.delta * input_node.output

    # backprop
    for j in range(num_inputs, num_inputs + num_hidden_units):
        hidden = nodes[j]
        summer = 0
        for weight in hidden.weights_forward:
            previous_node = nodes[weight.end_node]
            summer += previous_node.delta * weights[weight.start_node][weight.end_node] * sigmoid_derivative(hidden.sum_of_weighted_inputs)
        hidden.delta = summer
        for weight in hidden.weights_backward:
            input_node = nodes[weight.start_node]
            weights[weight.start_node][weight.end_node] += -1 * learning_rate * hidden.delta * input_node.output

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
