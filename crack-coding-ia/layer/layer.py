import numpy as np
from neuron.neuron import Neuron


class Layer:

    def __init__(self,num_neuros, input_size):
        self.neurons=[Neuron(input_size) for _ in range(num_neuros)]


    def forward(self, inputs):
        return np.array([neuron.forward(inputs) for neuron in self.neurons])
    
    def backward(self, d_outputs, learning_rate):
        d_inputs = np.zeros(self.neurons[0].input.shape)
        for i, neuron in enumerate(self.neurons):
            d_inputs+=neuron.backward(d_outputs[i], learning_rate)
        return d_inputs
    
if __name__ == "__main__":
    layer = Layer(3, 4)
    inputs = np.array([1,2,3,4])
    outputs = layer.forward(inputs)
    print("Layer outputs:", outputs)