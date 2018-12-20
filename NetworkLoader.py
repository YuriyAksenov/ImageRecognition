import json
import sys

import numpy as np

from Network import Network



def save(network: Network, filename: str):
        data = {"sizes": network.sizes,
                "weights": [w.tolist() for w in network.weights],
                "biases": [b.tolist() for b in network.biases]}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()

#### Loading a Network
def load(filename):
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    net = Network(data["sizes"])
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net