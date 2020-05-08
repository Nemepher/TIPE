import numpy as np

def sigmoid (inputs, derivative=False):
    outputs = 1/(1+np.exp(-inputs))
    if derivative:
        outputs = np.exp(-inputs)*(outputs**2)
    return outputs

