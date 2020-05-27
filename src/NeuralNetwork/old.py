import numpy as np 

#Activation function
sigmoid = lambda x : 1/(1+np.exp(-x))
d_sigmoid = lambda x : 1

p Loss function
SoS = lambda X1,X2 : sum((X1-X2)**2)  #Sum of square error

class Layer:
    def __init__(self, inputSize, outputSize, activationFunction):
        self.weights = np.random.rand(outputSize, inputSize)
        self.bias = np.random.rand(outputSize)
        self.actfun = activationFunction

    def process(self,inputs):
        return self.actfun( np.dot(self.weights,inputs)+self.bias ) # Voir doc, tres pratique pour faire toutes les opérations en meme temps !!
    
    def adjust_bias(self,correction):
        self.bias+=correction
    def adjust_weight(self,correction):
        self.weight[n,m]=correction

class NeuralNetwork:
    def __init__(self, sizes, activationFunction, d_activationFunction, lossFunction): #size [inputSize1,.,outputSize]
        layers = []
        length = len(sizes)-1
        for i in range(length):
            layers.append(Layer(sizes[i], sizes[i+1], activationFunction) )
        self.layers = layers
        self.lossfun = lossFunction 
        self.dactfunc = d_activationFunction
        
    def feedforward(self,inputs):
        i = inputs
        for layer in self.layers:
            i = layer.process(i)
        return i

    def backpropagate(self,train_inputs,train_outputs):
        outputs = self.feedforward(inputs)
        error = self.lossfun(outputs,train_outputs)

    def get_global_error():
        pass

def train1( NN,iterations,inputs,outputs ):
    for _ in range(iterations):
        NN.backpropagate(inputs,outputs)
    return NN.get_global_error()

test = NeuralNetwork([3,4,1], sigmoid, d_sigmoid, SoS)
for _ in range(100):
    print(test.feedforward(np.random.rand(3)))

''' Outdated
class Neuron:
    def __init__(self, inputWeights, inputBias=0):
        self.weights = inputWeights
        self.bias = inputBias
    def setWeight(n,v):
        self.weights[n]=v
    def process(self,inputs):
        return np.dot(inputs, self.weights) + self.bias #Shorter and faster

'''

  def export_to_db( self, filename ):  #TODO mais pas nécessaire
        
        if name[-7:] != '.sqlite' : filename += '.sqlite'
        conn = sqlite3.connect(filename)
        cur = conn.cursor()
        '''
        cur.execute(
               "DROP TABLE IF EXISTS 'layers';DROP TABLE IF EXISTS 'weights';DROP TABLE IF EXISTS 'biases';
                CREATE TABLE layers  ( id INT PRIMARY KEY AUTO INCREMENT=0, size INT NOT NULL);
                CREATE TABLE weights ( id INT PRILARY KEY AUTO INCREMENT=0, weight FLOAT NOT NULL);
                CREATE TABLE biases  ( id INT PRIMARY KEY AUTO INCREMENT=0, bias FLOAT NOT NULL);")
        '''
        insert_query = ""
        conn.commit()
        conn.close()
