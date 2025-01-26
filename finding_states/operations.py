import numpy as np

def adjoint(state):
    ''' Returns the adjoint of a state vector. For a np.matrix, can use .H'''
    return np.conjugate(state).T