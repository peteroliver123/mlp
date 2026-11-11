from Example import Example
from Weight import Weight
from Node import Node
import numpy as np

def sigmoid(z):
    if z >= 0:
        return 1 / (1 + np.exp(-z))
    else:
        exp_z = np.exp(z)
        return exp_z / (1 + exp_z)

def sigmoid_derivative(z):
    return z * (1 - z)

def delta_rule(delta_j, x_i, learning_rate):
    return learning_rate * delta_j * x_i

def fill_examples():
    examples = [
                    Example((0, 0), 0),
                    Example((0, 1), 1),
                    Example((1,0), 1),
                    Example((1,1), 0)
                ]
    return examples

def output_weight_matrix(total_nodes, weights):
    print("[")
    for i in range(0, total_nodes):
        print("[", end="")
        for j in range(0, total_nodes):
            print(weights[i][j], "", end="")
        print("]")
    print("]")
    print("")

def weight_update(current_layer, nodes, weights, learning_rate):
    current_layer.bias += learning_rate * current_layer.delta
    for weight in current_layer.weights_backward:
        input_node = nodes[weight.start_node]
        weights[weight.start_node][weight.end_node] += delta_rule(current_layer.delta, input_node.output, learning_rate)

def get_weighted_sum(nodes, weights, index):
    current_layer = nodes[index]
    weighted_sum = current_layer.bias
    for weight in current_layer.weights_backward:
        weighted_sum += (nodes[weight.start_node].output * weights[weight.start_node][weight.end_node])
    current_layer.output = sigmoid(weighted_sum)
    current_layer.sum_of_weighted_inputs = weighted_sum
    return current_layer

def main():
    print("MLP Project Started!")
    examples = fill_examples()
    num_examples = len(examples)

    num_inputs = 2
    num_outputs = 1
    num_hidden_units = 4
    learning_rate = 0.5

    total_nodes = num_inputs + num_hidden_units + num_outputs
    weights = np.random.uniform(-1, 1, size=(total_nodes, total_nodes))

    ### Verify Weight Matrix Initialisation ###
    output_weight_matrix(total_nodes, weights)

    nodes = [Node(i) for i in range(0, total_nodes)]
    #can this be one large nodes array

    ### Setup Weights ###
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

    ### Verify Weights Attached to Nodes ###
    for i in range(0, total_nodes):
        node = nodes[i]
        node.reveal_node()

    ### True Start of Back Prop ###
    for epoch in range(0, 11600):
        total_error = 0
        for example in range(0, num_examples):
            for h in range(0, num_inputs):
                nodes[h].output = examples[example].input_of[h]
            target = examples[example].target

            for j in range(num_inputs, num_inputs + num_hidden_units):
                # Forward pass
                get_weighted_sum(nodes, weights, j)

            for k in range(num_inputs + num_hidden_units, total_nodes):
                # Forward pass
                output = get_weighted_sum(nodes, weights, k)
                print("The Result after example", example, "epoch", epoch, "was", output.output)
                total_error += 0.5 * (target - output.output)**2
                #Backward pass for output unit
                output.delta = (target - output.output) * sigmoid_derivative(output.output)

            for j in range(num_inputs, num_inputs + num_hidden_units):
                #Backward pass for hidden unit
                hidden = nodes[j]
                summer = 0
                for weight in hidden.weights_forward:
                    next_node = nodes[weight.end_node]
                    summer += (next_node.delta * weights[weight.start_node][weight.end_node])
                hidden.delta = summer * sigmoid_derivative(hidden.output)

            for k in range(num_inputs + num_hidden_units, total_nodes):
                output = nodes[k]
                weight_update(output, nodes, weights, learning_rate)

            for j in range(num_inputs, num_inputs + num_hidden_units):
                hidden = nodes[j]
                weight_update(hidden, nodes, weights, learning_rate)

            ### Verify Weights Changed After Back Prop ###
            #output_weight_matrix(total_nodes, weights)
        print("Average Error: ", total_error / num_examples)

if __name__ == "__main__":
    main()
