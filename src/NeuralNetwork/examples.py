import numpy as np
import matplotlib.pyplot as plt
from multilayer_perceptron import *


## Exemple 1 : Simple apprentissage "manuel"##
def ex1():
    NN = Perceptron()
    a=np.array([1,2,1])
    b=np.array([1,2,9])
    c=np.array([1,2,5])

    for i in range(20):
        for i in range(60):
            NN.update_weights(NN.gradient(a,0.1)[0],1)
            NN.update_weights(NN.gradient(b,0.9)[0],1)
            NN.update_weights(NN.gradient(c,0.5)[0],1)
        print(NN.feedforward(a),NN.feedforward(b),NN.feedforward(c))
        
    print(NN.feedforward(np.array([1,2,3])))
    print(NN.feedforward(np.array([1,2,7])))

## Exemple 2 : Mêmes données, plus avancé ##
def ex2():
    NN = Perceptron()
    X=np.array([np.array([1,2,3]), np.array([1,2,5]),np.array([1,2,8]),np.array([1,2,0]),np.array([1,2,9])])
    X2=np.array([np.array([1,2,1]), np.array([1,2,0]),np.array([1,2,2]),np.array([1,2,4]),np.array([1,2,8])])
    Y=np.array([0.3,0.5,0.8,0.0,0.9])
    for k in X :
        print(" ",NN.feedforward(k))
    print(NN.evaluate(X,Y))
    NN.train(X,Y,1,4,200)

    print("\n")
    for k in X:
        print(" ",NN.feedforward(k))
    print(NN.evaluate(X,Y))
    print("\n")
    for k in X2:
        print(" ",NN.feedforward(k))
    print(NN.evaluate(X2,Y))

## Exemple 3 : Mêmes donnés, avec graphe ##
def ex3():
    NN = Perceptron()
    X=np.array([np.array([1,2,3]), np.array([1,2,5]),np.array([1,2,8]),np.array([1,2,0]),np.array([1,2,9])])
    Y=np.array([0.3,0.5,0.8,0.0,0.9])
    Xe = np.arange(0,30)
    Ye = [NN.evaluate(X,Y)]
    for k in range(1,30):
        NN.train(X,Y,1,4,1)
        Ye.append(NN.evaluate(X,Y))
    plt.plot(Xe,Ye)
    plt.show()

## Gros test ##
def ex4():

    settings = {
        "cost_function": sum_of_square,
        "activation_function": sigmoid,
        "size": [400,200,1], #[size of input layer, ... , size of output layer] 
        "init_bias": 0.1,
        "min_weight":-1,
        "max_weight":1
    }
    
    X = np.array([ np.random.uniform(-1,1, 400) for k in range(500) ])
    Y = np.array([ np.random.uniform(0,0) for k in range(500) ])

    NN = Perceptron(settings)
    NN.train(X,Y,300,1200,1)
    
    Xe = np.arange(0,10)
    Ye = [NN.evaluate(X,Y)]
    for k in range(1,10):
        NN.train(X,Y,1,300,10)
        Ye.append(NN.evaluate(X,Y))
    plt.plot(Xe,Ye)
    plt.show()

ex4()
