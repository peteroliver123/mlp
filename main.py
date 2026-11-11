from Example import Example
from Weight import Weight
from Node import Node
import numpy as np
import math
from MLP import MLP

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

    # MLP Setting
    num_inputs = 2
    num_hidden_units = 4
    num_outputs = 1
    examples = fill_examples()

    ### MLP Setup ###
    my_mlp = MLP(num_inputs, num_hidden_units, num_outputs, examples)
    my_mlp.verify_weight_init()
    my_mlp.weight_attaching()
    my_mlp.verify_weight_attaching()
    backprop(my_mlp)

def backprop(my_mlp):
    ### True Start of Back Prop ###
    for epoch in range(0, 11600):
        total_error = 0
        for ex in range(0, my_mlp.num_examples):
            for h in range(0, my_mlp.num_input_units):
                my_mlp.nodes_array[h].output = my_mlp.examples[ex].input_of[h]
            target = my_mlp.examples[ex].target

            for j in range(my_mlp.num_input_units, my_mlp.num_input_units + my_mlp.num_hidden_units):
                # Forward pass
                get_weighted_sum(my_mlp.nodes_array, my_mlp.weights, j)

            for k in range(my_mlp.num_input_units + my_mlp.num_hidden_units, my_mlp.total_nodes):
                # Forward pass
                output = get_weighted_sum(my_mlp.nodes_array, my_mlp.weights, k)
                print("The Result after example", ex, "epoch", epoch, "was", output.output)
                total_error += 0.5 * (target - output.output)**2
                #Backward pass for output unit
                output.delta = (target - output.output) * sigmoid_derivative(output.output)

            for j in range(my_mlp.num_input_units, my_mlp.num_input_units + my_mlp.num_hidden_units):
                #Backward pass for hidden unit
                hidden = my_mlp.nodes_array[j]
                summer = 0
                for weight in hidden.weights_forward:
                    next_node = my_mlp.nodes_array[weight.end_node]
                    summer += (next_node.delta * my_mlp.weights[weight.start_node][weight.end_node])
                hidden.delta = summer * sigmoid_derivative(hidden.output)

            for k in range(my_mlp.num_input_units + my_mlp.num_hidden_units, my_mlp.total_nodes):
                output = my_mlp.nodes_array[k]
                weight_update(output, my_mlp.nodes_array, my_mlp.weights, my_mlp.LEARNING_RATE)

            for j in range(my_mlp.num_input_units, my_mlp.num_input_units + my_mlp.num_hidden_units):
                hidden = my_mlp.nodes_array[j]
                weight_update(hidden, my_mlp.nodes_array, my_mlp.weights, my_mlp.LEARNING_RATE)

            ### Verify Weights Changed After Back Prop ###
            #output_weight_matrix(total_nodes, weights)
        print("Average Error: ", total_error / my_mlp.num_examples)

    ### TESTING ###
    print()
    print("### TESTING ###")
    for example in range(0, my_mlp.num_examples):
        result_string = "( "
        for h in range(0, my_mlp.num_input_units):
            my_mlp.nodes_array[h].output = my_mlp.examples[example].input_of[h]
            result_string = result_string + str(my_mlp.examples[example].input_of[h]) + " "
        result_string += ")"
        target = my_mlp.examples[example].target

        for j in range(my_mlp.num_input_units, my_mlp.num_input_units + my_mlp.num_hidden_units):
            # Forward pass
            get_weighted_sum(my_mlp.nodes_array, my_mlp.weights, j)

        for k in range(my_mlp.num_input_units + my_mlp.num_hidden_units, my_mlp.total_nodes):
            # Forward pass
            output = get_weighted_sum(my_mlp.nodes_array, my_mlp.weights, k)
            if math.isclose(output.output, target, abs_tol=0.1):
                print("Input: ", result_string, "gave approx output: ", target)


if __name__ == "__main__":
    main()
