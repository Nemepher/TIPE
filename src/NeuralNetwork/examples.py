import numpy as np
import matplotlib.pyplot as plt
from multilayer_perceptron import *
from progress.bar import ChargingBar
from progress.spinner import Spinner


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
    
    Xe = np.arange(0,10)
    Ye = [NN.evaluate(X,Y)]
    for k in range(1,10):
        NN.train(X,Y,1,300,10)
        Ye.append(NN.evaluate(X,Y))
    plt.plot(Xe,Ye)
    plt.show()

## Test graphique, on trace un rond  ##
def ex5(x,y,R,L):
    # Figure and selected points
    S = np.linspace(0,np.pi*2)
    plt.fill([x+R*np.cos(t) for t in S],[y+R*np.sin(t) for t in S])
    X = np.array([ np.random.uniform(0,L,2) for k in range(700) ])
    Y = np.array([ ((X[k][0]-x)**2 + (X[k][1]-y)**2)<=R for k in range(700) ]) 
    plt.plot(X[:,0],X[:,1],'+',color="red")    
    plt.axis([0,L,0,L])
    plt.show()

    #Creating NN
    settings = {
        "cost_function": sum_of_square,
        "activation_function": sigmoid,
        "size": [2,4,4,1], #[size of input layer, ... , size of output layer] 
        "init_bias": 0.1,
        "min_weight":-1,
        "max_weight":1
    }
    NN = Perceptron(settings)
    
    # Initial figure
    x = np.linspace(0,L,L+1) 
    Xm = np.array([ [i,j] for j in x for i in x ])
    Ym = np.array([ [NN.feedforward(Xm[k+len(x)*l])[0] for k in range(len(x))] for l in range(len(x)) ])
    plt.imshow(Ym, cmap='hot')
    plt.axis([0,L,0,L])
    plt.show()
    
    # Training
    bar = ChargingBar('training', max = 20)
    Xe = np.arange(0,20)
    Ye = [NN.evaluate(X,Y)]
    for k in range(1,20):
        NN.train(X,Y,1,200,70)
        Ye.append(NN.evaluate(X,Y))
        bar.next()
    plt.plot(Xe,Ye)
    bar.finish()
    plt.show()
    
    # After-training results
    Ym = np.array([ [NN.feedforward(Xm[k+len(x)*l])[0] for k in range(len(x))] for l in range(len(x)) ])
    plt.imshow(Ym, cmap='hot')
    plt.axis([0,L,0,L])
    plt.show()
   

ex5(10,0,8,10)
