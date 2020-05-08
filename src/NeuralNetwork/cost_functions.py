import numpy as np

def sum_of_square (outputs, expected_outputs, derivative=False ):
    if derivative : 
        return np.sum( outputs - expected_outputs )
    return 0.5*np.sum( (outputs - expected_outputs)**2 )
