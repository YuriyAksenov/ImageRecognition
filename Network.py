
import json
import random
import sys
import numpy as np
from Metrics import * 



class Network:

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]] # гуасиановская инициализация байасов и весов случайными значениями. Байасы только для выхода из нейрона, поэтому для первого слоя нет байеса
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]


    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def run(self, training_data, epochs, mini_batch_size, eta,
            test_data=None,
            monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=False,
            monitor_training_cost=False,
            monitor_training_accuracy=False):

        training_data = list(training_data)
        n = len(training_data)

        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)

        evaluation_cost = []
        evaluation_accuracy = []
        training_cost = []
        training_accuracy = []

        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

            print("Epoch %s training complete" % j)

            # if monitor_training_cost:
            #     cost = self.total_cost(training_data)
            #     training_cost.append(cost)
            #     print("Cost on training data: {}".format(cost))
            if monitor_training_accuracy:
                accuracy, predicted_digits = self.accuracy(training_data, convert=True)
                training_accuracy.append(accuracy)
                print(self.predicted_digits_accuracy(predicted_digits))
                print("Accuracy on training data: {} / {}".format(accuracy, n))
            # if monitor_evaluation_cost:
            #     cost = self.total_cost(test_data, convert=True)
            #     evaluation_cost.append(cost)
            #     print("Cost on evaluation data: {}".format(cost))
            if monitor_evaluation_accuracy:
                accuracy, predicted_digits = self.accuracy(test_data)
                evaluation_accuracy.append(accuracy)
                print(self.predicted_digits_accuracy(predicted_digits))
                print("Accuracy on evaluation data: {} / {}".format(accuracy, n_test))

            conf = self.conf_matrix(test_data)
            print(conf)
            print(self.fscore(conf));

            aplot_confusion_matrix(cm=np.array(conf), normalize=True, target_names=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"], title="Confusion Matrix, Normalized")
            
            print("----------------------------------------")

           
        return evaluation_cost, evaluation_accuracy, training_cost, training_accuracy

    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def cost_derivative(self, output_activations, y):
        return (output_activations-y)
    
    def conf_matrix(self, data, convert=False):
        conf = np.zeros((10,10))
        if convert:
            results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                       for (x, y) in data]
            
        else:
            results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in data]
        conf = np.zeros((10,10))
        for  (predicted, real) in results:
            conf[predicted][real]+=1
        return conf

    def fscore(self, conf_matrix):
        precisions = np.zeros(10)
        recalls = np.zeros(10)
        for i in range(0,10):
            precisions[i] = conf_matrix[i][i] / sum(conf_matrix[i,:])
            recalls[i] = conf_matrix[i][i] / sum(conf_matrix[:,i])

        precision_av = sum(precisions) / len(precisions)
        recall_av = sum(recalls) / len(recalls)
        return 2*(precision_av*recall_av / (precision_av+recall_av))

    def accuracy(self, data, convert=False):
        if convert:
            results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                       for (x, y) in data]
        else:
            results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in data]

        result_accuracy = sum(int(x == y) for (x, y) in results)
        predicted_digits = np.zeros((10,2));
        for (x, y) in results:
            if(x==y):
                predicted_digits[y][0] +=1
            else:
                predicted_digits[y][1] +=1
        return result_accuracy, predicted_digits

    def predicted_digits_accuracy(self, predicted_digits):
        result = []
        index = 0
        for item in predicted_digits:
            result.append((index, item[0]/(item[0]+item[1])))
            index+=1
        return result;


#### Miscellaneous functions
def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

def predict(filename: str, net: Network):
    
    import cv2
    import numpy as np

    import os.path
    isExist = os.path.isfile(filename) 
    if(isExist):
        print("exist")
    else:
        print("not exist")
    m = filename.split('/')[-1]
    #m =  cv2.imread('./HandTestImages/'+str(m))
    m =  cv2.imread('./HandTest/'+str(m))

    # get image properties.
    (h, w, _) = np.shape(m)
    
    # iterate over the entire image.
    x = []
    for py in range(0,h):
        for px in range(0,w):
            x.append(m[py][px][0])

    x = np.array([np.array([item], dtype=float) for item in x])
    x = 1 - x / 255 

    result = net.feedforward(x)
    summ = sum(result)
    for i in range(len(result)):
        print(str(i) + " = " + str(result[i]/summ*100))
    return np.argmax(result)