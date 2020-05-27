import numpy as np 
import sqlite3
import pickle
from activation_functions import *
from cost_functions import *

default_settings = {
        "cost_function": sum_of_square,
        "activation_function": sigmoid,
        "size": [3,1], #[size of input layer, ... , size of output layer] 
        "init_bias": 0.1,
        "min_weight":-1,
        "max_weight":1
}

class Perceptron:
    def __init__( self, settings=default_settings ):
        self.__dict__.update( settings )
        self.num_neurons = np.sum( self.size ) 
        self.num_layers = len(self.size)-1

        self.weights = np.array([ np.random.uniform( self.min_weight, self.max_weight, (self.size[k+1], self.size[k]) )  for k in range(self.num_layers) ])
        self.biases  = np.array([ np.array( [self.init_bias]*self.size[k+1] ) for k in range(self.num_layers) ])
        

    def update_weights( self, gradient, learning_rate ):
        self.weights -= learning_rate*gradient

    def update_biases( self, gradient, learning_rate ):
        self.biases -= learning_rate*gradient


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
        
        weights_gradient = np.array([ np.array( [ detailled_outputs[k]*delta for delta in deltas[k] ] ) for k in range(self.num_layers) ])
        biases_gradient = deltas
        return weights_gradient, biases_gradient


    def train( self, X, Y, learning_rate, batch_size=10, epoch=200 ): 
        
        n = len(X)
        
        for _ in range(epoch) :
            indices = np.arange(n)
            np.random.shuffle(indices)
            X = X[indices]
            Y = Y[indices]

            for k in range( 0, n, batch_size ):
                temp_weights_gradient = np.array([np.zeros_like(k) for k in self.weights])
                temp_biases_gradient = np.array([np.zeros_like(k) for k in self.biases])

                for i in range( k, min( k+batch_size, n ) ):
                    temp_gradient = self.gradient( X[i], Y[i] ) 
                    temp_weights_gradient += temp_gradient[0]
                    temp_biases_gradient += temp_gradient[1]
                
                temp_weights_gradient *= 1/batch_size
                temp_biases_gradient *= 1/batch_size
                
                self.update_weights( temp_weights_gradient, learning_rate ) 
                self.update_biases( temp_biases_gradient, learning_rate )
            

    def evaluate( self, X, Y ):
        
        n=len(X)
        X_forwarded = np.array([ self.feedforward(k) for k in X ])
        mean_error = self.cost_function( X_forwarded, Y) / n
        range_of_forwarded_values = np.max(X_forwarded) - np.min(X_forwarded)
        return 1 - ( mean_error / range_of_forwarded_values )


    def export_to_bs( self, filename ): 
        
       with open( filename, 'wb' ) as f:
           print("\nPickling... ")
           pickle.dump(self.__dict__, f)
           print("completed!\n")
    
    def import_from_bs( self, filename ):
        
        with open( filename, 'rb' ) as f:
            print("\nUnpickling... ")
            self.__dict__.update( pickle.load( f ) )
            print("completed!\n")


  


