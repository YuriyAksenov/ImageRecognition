import json
import sys

import numpy as np

from Network import Network

def save(network: Network, filename: str):
        """Save the neural network to the file ``filename``."""
        data = {"sizes": network.sizes,
                "weights": [w.tolist() for w in network.weights],
                "biases": [b.tolist() for b in network.biases],
                "cost": str(network.cost.__name__)}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()

#### Loading a Network
def load(filename):
    """Load a neural network from the file ``filename``.  Returns an
    instance of Network.

    """
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    cost = getattr(sys.modules[__name__], data["cost"])
    net = Network(data["sizes"], cost=cost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net