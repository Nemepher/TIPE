import numpy as np

def sigmoid (inputs, derivative=False):
    outputs = 1/(1+np.exp(-inputs))
    if derivative:
        outputs = np.exp(-inputs)*(outputs**2)
    return outputs

def ReLU (inputs, derivative=False):
    if derivative:
        return inputs>0
    return np.max(inputs,np.zeros_like(inputs))
