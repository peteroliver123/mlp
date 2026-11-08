class Node:
    def __init__(self, id):
        self.id = id
        self.weights_backward = []
        self.weights_forward = []

        self.output = None
        self.delta = None
        self.sum_of_weighted_inputs = None

    def reveal_node(self):
        print("Node with id", self.id)
        print("Backward Weights:")
        print("[")
        for w in self.weights_backward:
            print(w.start_node, w.end_node)
        print("]")
        print("Forward Weights:")
        print("[")
        for w in self.weights_forward:
            print(w.start_node, w.end_node)
        print("]")