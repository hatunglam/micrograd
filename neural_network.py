import math
import numpy as np 
from gradient import Value
import random

class Neuron:

    def __init__(self, nin):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1,1))

    def __call__(self, x):
        # w . x + b
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b) 
        out = act.tanh()
        return out
    
    def parameters(self):
        return self.w + [self.b]
    
class Layer:

    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        # --> list of size n_out: [---#weights_neuron---]
        return out[0] if len(out) == 1 else out
    
    def parameters(self):
        # params = []
        # for neuron in self.neurons:
        #     params.extend(neuron.parameters())
        return [p for neuron in self.neurons for p in neuron.parameters()]
        

class MLP:

    def __init__(self, n_in, n_out):
        # n_out: list of output size for each layer
        size = [n_in] + n_out
        # --> [nin, nout_1, nout_2, ..., nout_n]

        self.layers = [Layer(size[i], size[i+1]) for i in range(len(n_out))]
        # ---> [(nin,nout_1),  <-- layer 0
        #       (nout_1, nout_2),
        #        ...........
        #       (nout_n-1, nout_n)]    <-- layer nout_n-1       

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x) # recursively compute x output and feed to the next iteration
        return x       

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    

def training_loop(model, data, label, lr, n_epoch):
    for k in range(n_epoch):

        # forward pass 
        y_pred = [model(x) for x in data]
        loss = sum((ypred - y)**2 for y, ypred in zip(label, y_pred))

        # zero gradients
        for p in model.parameters():
            p.grad = 0.0

        # calculate gradients
        loss.backward()

        # update
        for p in model.parameters():
            p.data += -lr * p.grad
        
        print(f"epoch {k}, loss: {loss.data}")