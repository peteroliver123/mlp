from NodeType import NodeType

class Node:
    def __init__(self, id_in_array, node_type, weights_backward, weights_forward):
        self.id_in_array = id_in_array
        self.node_type = node_type
        self.weights_backward = weights_backward
        self.weights_forward = weights_forward

        self.output = None
        self.delta = None
        self.sum_of_weighted_inputs = None

    def reveal_node(self):
        print("Node of", self.node_type, "with id", self.id_in_array)
        print("Backward Weights:")
        print("[")
        for w in self.weights_backward:
            print(w.start_node, w.end_node, w.value)
        print("]")
        print("Forward Weights:")
        print("[")
        for w in self.weights_forward:
            print(w.start_node, w.end_node, w.value)
        print("]")



# if I'm an input no need to calculate y, so I traverse my weights, bringing with me an input value and a weight value and pass them to
# calculate output function of node with id at end of weight.