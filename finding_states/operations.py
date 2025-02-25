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

def rotate_m(m, n):
    """
    Rotate matrix m with matrix n
    
    Params:
    m - the matrix to be rotated
    n - the matrix that is doing the rotation (i.e. R_z)
    NOTE: m and n must be of the same size
    """
    return n @ m @ adjoint(n)


if __name__ == "__main__":

    print("===== PAULI_X about z =====")
    print("Actual: \n", op.rotate_z(PAULI_X, theta), "\n")
    print("Predicted: \n", np.cos(theta)*PAULI_X + np.sin(theta)*PAULI_Y, "\n")

    print("===== PAULI_Y about z =====")
    print("Actual: \n", op.rotate_z(PAULI_Y, theta), "\n")
    print("Predicted: \n", np.cos(theta)*PAULI_Y - np.sin(theta)*PAULI_X, "\n")

    print("===== PAULI_X about y =====")
    print("Actual: \n", op.rotate_y(PAULI_X, theta), "\n")
    print("Predicted: \n", np.cos(theta)*PAULI_X - np.sin(theta)*PAULI_Z, "\n")

    print("===== PAULI_Z about y =====")
    print("Actual: \n", op.rotate_y(PAULI_Z, theta), "\n")
    print("Predicted: \n", np.cos(theta)*PAULI_Z + np.sin(theta)*PAULI_X, "\n")

    print("===== PAULI_Y about x =====")
    print("Actual: \n", op.rotate_x(PAULI_Y, theta), "\n")
    print("Predicted: \n", np.cos(theta)*PAULI_Y - np.sin(theta)*PAULI_Z, "\n")

    print("===== PAULI_Z about x =====")
    print("Actual: \n", op.rotate_x(PAULI_Z, theta), "\n")
    print("Predicted: \n", np.cos(theta)*PAULI_Z + np.sin(theta)*PAULI_Y, "\n")

    ### Predicted - Actual (expected: all 0s for all ###
    # print((np.cos(theta)*PAULI_X + np.sin(theta)*PAULI_Y) - rotate_z(PAULI_X, theta))
    # print((np.cos(theta)*PAULI_Y - np.sin(theta)*PAULI_X) - rotate_z(PAULI_Y, theta))
    # print((np.cos(theta)*PAULI_X - np.sin(theta)*PAULI_Z) - rotate_y(PAULI_X, theta))  
    # print((np.cos(theta)*PAULI_Z + np.sin(theta)*PAULI_X) - rotate_y(PAULI_Z, theta))
    # print((np.cos(theta)*PAULI_Y - np.sin(theta)*PAULI_Z) - rotate_x(PAULI_Y, theta))
    # print((np.cos(theta)*PAULI_Z + np.sin(theta)*PAULI_Y) - rotate_x(PAULI_Z, theta))