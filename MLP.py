import numpy as np
from Weight import Weight
from Node import Node

class MLP:
    LEARNING_RATE = 0.05
    def __init__(self, num_input_units, num_hidden_units, num_output_units, examples):
        self.num_input_units = num_input_units
        self.num_hidden_units = num_hidden_units
        self.num_output_units = num_output_units
        self.examples = examples

        # Calculated
        self.total_nodes = self.num_output_units + self.num_hidden_units + self.num_input_units
        self.num_examples = len(self.examples)
        self.weights = np.random.uniform(-1, 1, size=(self.total_nodes, self.total_nodes))
        self.nodes_array = [Node(i) for i in range(0, self.total_nodes)]

    ### BACKWARD PASS & LEARNING FUNCTIONS ###
    def weight_update(self, current_layer):
        current_layer.bias += self.LEARNING_RATE * current_layer.delta
        for weight in current_layer.weights_backward:
            input_node = self.nodes_array[weight.start_node]
            self.weights[weight.start_node][weight.end_node] += self.delta_rule(current_layer.delta, input_node.output)

    ### FORWARD PASS FUNCTIONS ###
    def get_weighted_sum(self, index, activation_function):
        current_layer = self.nodes_array[index]
        weighted_sum = current_layer.bias
        for weight in current_layer.weights_backward:
            weighted_sum += (self.nodes_array[weight.start_node].output * self.weights[weight.start_node][weight.end_node])
        current_layer.output = activation_function(weighted_sum)
        current_layer.sum_of_weighted_inputs = weighted_sum
        return current_layer

    ### UTILITY FUNCTIONS ###
    def delta_rule(self, delta_j, x_i):
        return self.LEARNING_RATE * delta_j * x_i

    ### SETUP FUNCTIONS ###

    def weight_attaching(self):
        for i in range(0, self.num_input_units):
            input_n = self.nodes_array[i]
            for j in range(self.num_input_units, self.num_input_units + self.num_hidden_units):
                hidden_n = self.nodes_array[j]
                input_n.weights_forward.append(Weight(input_n.id, hidden_n.id))
                hidden_n.weights_backward.append(Weight(input_n.id, hidden_n.id))

        for k in range(self.num_input_units + self.num_hidden_units, self.total_nodes):
            output_n = self.nodes_array[k]
            for j in range(self.num_input_units, self.num_input_units + self.num_hidden_units):
                hidden_n = self.nodes_array[j]
                output_n.weights_backward.append(Weight(hidden_n.id, output_n.id))
                hidden_n.weights_forward.append(Weight(hidden_n.id, output_n.id))

    def verify_weight_attaching(self):
        for i in range(0, self.total_nodes):
            node = self.nodes_array[i]
            node.reveal_node()

    def verify_weight_init(self):
        print("[")
        for i in range(0, self.total_nodes):
            print("[", end="")
            for j in range(0, self.total_nodes):
                print(self.weights[i][j], "", end="")
            print("]")
        print("]")
        print("")