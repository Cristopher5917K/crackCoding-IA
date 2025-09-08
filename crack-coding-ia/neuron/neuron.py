import numpy as np

class Neuron:

    def __init__(self, n_input):
        self.weights = np.random.randn(n_input)
        self.bias = np.random.randn()
        self.output = 0
        self.input = None
        self.dweights = np.zeros_like(self.weights)
        self.dbias = 0


    def activate(self,x):
        return 1/(1+np.exp(-1))
    
    def activate_derivative(self,x):
        return x*(1-x)
    

    def forward(self, inputs):
        self.input = inputs
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        self.output = self.activate(weighted_sum)
        return self.output


    def backward(self,d_output, learning_rate):
        d_inputs = np.zeros(len(self.neurons[0].input))  # Initialize gradient wrt inputs
        for i, neuron in enumerate(self.neurons):
            d_inputs += neuron.backward(d_outputs[i], learning_rate)
        return d_inputs

        
if __name__ == "__main__":
    neuron = Neuron(3)
    inputs = np.array([1,2,3])
    output = neuron.forward(inputs)
    print("Neuron output:", output)