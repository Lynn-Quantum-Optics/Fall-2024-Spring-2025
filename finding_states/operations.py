import numpy as np
import states_and_gates as gates

def adjoint(state):
    ''' Returns the adjoint of a state vector. For a np.matrix, can use .H'''
    return np.conjugate(state).T

def partial_transpose(rho, subsys='B'):
    """ 
    Helper function to compute the partial transpose of a density matrix. 
    Useful for the Peres-Horodecki criterion, which states that if the partial transpose 
    of a density matrix has at least one negative eigenvalue, then the state is entangled.
    
    Params:
        rho: density matrix
        subsys: which subsystem to compute partial transpose wrt, i.e. 'A' or 'B'
    """
    # decompose rho into blocks
    b1 = rho[:2, :2]
    b2 = rho[:2, 2:]
    b3 = rho[2:, :2]
    b4 = rho[2:, 2:]

    PT = np.matrix(np.block([[b1.T, b2.T], [b3.T, b4.T]]))

    if subsys=='B':
        return PT
    elif subsys=='A':
        return PT.T


# Rotation operations
def rotate_z(m, theta):
    """Rotate matrix m by theta about the z axis"""
    return gates.R_z(theta) @ m @ adjoint(gates.R_z(theta))

def rotate_x(m, theta):
    """Rotate matrix m by theta about the x axis"""
    return gates.R_x(theta) @ m @ adjoint(gates.R_x(theta))

def rotate_y(m, theta):
    """Rotate matrix m by theta about the y axis"""
    return gates.R_y(theta) @ m @ adjoint(gates.R_y(theta))


def get_witness(W, rho):
    """
    Returns the value to be minimized to find the expectation value of W

    Params:
        W   - the witness matrix
        rho - the density matrix
    """
    return np.real(np.trace(W @ rho))