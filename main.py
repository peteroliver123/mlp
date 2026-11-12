from Example import Example
import math
from MLP import MLP
import numpy as np
import random

def fill_examples_q1():
    examples = [
                    Example((0, 0), 0),
                    Example((0, 1), 1),
                    Example((1,0), 1),
                    Example((1,1), 0)
                ]
    return examples

def fill_examples_q2():
    examples = []
    for i in range(0, 500):
        inputs = np.random.uniform(-1, 1, size=4)
        output = math.sin(inputs[0] - inputs[1] + inputs[2] - inputs[3])
        inputs = Example(inputs, output)
        examples.append(inputs)
    return examples

def sigmoid(z):
    if z >= 0:
        return 1 / (1 + np.exp(-z))
    else:
        exp_z = np.exp(z)
        return exp_z / (1 + exp_z)

def sigmoid_derivative(z):
    return z * (1 - z)

def tanh(z):
    return math.tanh(z)

def tanh_derivative(z):
    return 1 - z**2

def linear(z):
    return z

def linear_derivative(z):
    return 1

def main():
    print("MLP Project Started!")

    # MLP Setting
    # Q1
    # num_inputs = 2
    #num_hidden_units = 4
    # num_outputs
    #file_title = "results/results_q1.dat"
    #examples = fill_examples_q1()
    #test_data = examples
    #f = sigmoid
    #f_prime = sigmoid_derivative

    #Q3
    num_inputs = 4
    num_hidden_units = 6
    num_outputs = 1
    file_title = "results/results_q3.dat"
    all_data = fill_examples_q2()
    examples = all_data[:400]
    test_data = all_data[400:500]
    f_hidden = tanh
    f_prime_hidden = tanh_derivative
    f_output = linear
    f_prime_output = linear_derivative

    ### MLP Setup ###
    my_mlp = MLP(num_inputs, num_hidden_units, num_outputs, examples)
    my_mlp.verify_weight_init()
    my_mlp.weight_attaching()
    my_mlp.verify_weight_attaching()
    train_model(my_mlp, 10000, file_title, f_hidden, f_prime_hidden, f_output, f_prime_output)
    test_model(my_mlp, test_data, f_hidden, f_output)

def train_model(my_mlp, num_epochs, file_title, f_hidden, f_prime_hidden, f_output, f_prime_output):
    ### True Start of Back Prop ###
    with open(file_title, "w") as file:
        file.write("Epoch Average_Error\n")
        for epoch in range(0, num_epochs):
            random.shuffle(my_mlp.examples)
            total_error = 0
            for ex in range(0, my_mlp.num_examples):
                for h in range(0, my_mlp.num_input_units):
                    my_mlp.nodes_array[h].output = my_mlp.examples[ex].input_of[h]
                target = my_mlp.examples[ex].target

                for j in range(my_mlp.num_input_units, my_mlp.num_input_units + my_mlp.num_hidden_units):
                    # Forward pass
                    my_mlp.get_weighted_sum(j, f_hidden)

                for k in range(my_mlp.num_input_units + my_mlp.num_hidden_units, my_mlp.total_nodes):
                    # Forward pass
                    output = my_mlp.get_weighted_sum(k, f_output)
                    #print("The Result after example", ex, "epoch", epoch, "was", output.output)
                    total_error += 0.5 * (target - output.output)**2
                    #Backward pass for output unit
                    output.delta = (target - output.output) * f_prime_output(output.output)

                for j in range(my_mlp.num_input_units, my_mlp.num_input_units + my_mlp.num_hidden_units):
                    #Backward pass for hidden unit
                    hidden = my_mlp.nodes_array[j]
                    summer = 0
                    for weight in hidden.weights_forward:
                        next_node = my_mlp.nodes_array[weight.end_node]
                        summer += (next_node.delta * my_mlp.weights[weight.start_node][weight.end_node])
                    hidden.delta = summer * f_prime_hidden(hidden.output)

                for k in range(my_mlp.num_input_units + my_mlp.num_hidden_units, my_mlp.total_nodes):
                    output = my_mlp.nodes_array[k]
                    my_mlp.weight_update(output)

                for j in range(my_mlp.num_input_units, my_mlp.num_input_units + my_mlp.num_hidden_units):
                    hidden = my_mlp.nodes_array[j]
                    my_mlp.weight_update(hidden)

            average_error = total_error / my_mlp.num_examples
            file.write(f"{epoch} {average_error}\n")
            print("On Epoch", epoch)

def test_model(my_mlp, examples, f_hidden, f_output):
    ### TESTING with Exterior Examples ###
    print()
    print("### TESTING ###")
    for ex in range(0, len(examples)):
        result_string = "( "
        for h in range(0, my_mlp.num_input_units):
            my_mlp.nodes_array[h].output = examples[ex].input_of[h]
            result_string = result_string + str(examples[ex].input_of[h]) + " "
        result_string += ")"
        target = examples[ex].target

        for j in range(my_mlp.num_input_units, my_mlp.num_input_units + my_mlp.num_hidden_units):
            # Forward pass
            my_mlp.get_weighted_sum(j, f_hidden)

        for k in range(my_mlp.num_input_units + my_mlp.num_hidden_units, my_mlp.total_nodes):
            # Forward pass
            output = my_mlp.get_weighted_sum(k, f_output)
            if math.isclose(output.output, target, abs_tol=0.1):
                print("Input: ", result_string, "gave approx output: ", target)
            else:
                print("TEST FAILED")
                print("Input: ", result_string, "gave", output.output," and didn't approx equal: ", target)


if __name__ == "__main__":
    main()
