import numpy as np
from activation_functions import *
from cost_functions import *

default_settings = {
        "cost_function": sum_of_square,
        "activation_function": sigmoid,
        "learning_rate": 0.5,
        "size": [3,3,1], #[size of input layer, ... , size of output layer] 
        "init_bias": 0.01,
        "min_weight":-0.01,
        "max_weight":0.01
}

class NeuralNetwork:
    def __init__( self, settings=default_settings ):
        self.__dict__.update( settings )
        self.num_neurons = np.sum( self.size ) 
        self.num_layers = len(self.size)-1

        self.weights = [ np.random.uniform( self.min_weight, self.max_weight, (self.size[k+1], self.size[k]) )  for k in range(self.num_layers) ]
        self.biases  = [ np.array( [self.init_bias]*self.size[k+1] ) for k in range(self.num_layers) ]
        

    def feedforward( self, inputs, trace=False):

        outputs = inputs
        if trace:
            derivatives = [] #Dérivé de la fonction d'activation
            detailled_outputs = [outputs]

        for k in range( self.num_layers ): 
            aggregation = np.dot( self.weights[k], outputs ) + self.biases[k]
            outputs = self.activation_function( aggregation )

            if trace:
                detailled_outputs.append( outputs )
                derivatives.append( self.activation_function( aggregation, derivative=True ) )
       
        if trace : return derivatives, detailled_outputs
        return outputs 
   

    def update_weights( self, gradient ):

        for k in range( self.num_layers ):
            self.weights[k] -= self.learning_rate*gradient[0][k]
            self.biases[k] -= self.learning_rate*gradient[1][k]


    def gradient( self, inputs, expected_outputs ):
            
        derivatives, detailled_outputs = self.feedforward( inputs, trace=True )
        cost_derivative = self.cost_function( detailled_outputs[-1], expected_outputs, derivative=True )

        delta = derivatives[-1]*cost_derivative #Delta de la dernière couche
        deltas = []

        for k in range(self.num_layers-2,-1,-1):
            deltas.append( delta )       
            delta = derivatives[k]*np.dot( delta, self.weights[k+1] ) 
        deltas.append( delta )
        deltas = deltas[::-1]
        
        weights_gradient = [ np.array( [ detailled_outputs[k]*delta for delta in deltas[k] ] ) for k in range(self.num_layers) ]
        biases_gradient = deltas
        return weights_gradient, biases_gradient





NN = NeuralNetwork()
#print("\n",NN.feedforward([1,2,4],True),"\n")
#print("\n",NN.gradient(np.array([1,2,4]),1))

a=np.array([1,2,1])
b=np.array([1,2,9])
c=np.array([1,2,5])
for i in range(7):
    for i in range(50):
        NN.update_weights(NN.gradient(a,0.1))
        NN.update_weights(NN.gradient(b,0.9))
        NN.update_weights(NN.gradient(c,0.5))
    print(NN.feedforward(a),NN.feedforward(b),NN.feedforward(c))
    
print(NN.feedforward(np.array([1,2,3])))
print(NN.feedforward(np.array([1,2,7])))
