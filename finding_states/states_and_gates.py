import numpy as np
import operations as op

# Pauli Gates
PAULI_Z = np.array([[1,0], [0,-1]])
PAULI_X = np.array([[0,1], [1, 0]])
PAULI_Y = np.array([[0, -1j], [1j, 0]])

# Bell States
PHI_P = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)])
PHI_M = np.array([1/np.sqrt(2), 0, 0, -1/np.sqrt(2)])
PSI_P = np.array([0, 1/np.sqrt(2),  1/np.sqrt(2), 0])
PSI_M = np.array([0, 1/np.sqrt(2),  -1/np.sqrt(2), 0])

# Rotation Matrices & Functions
def R_z(theta):
    return np.array([[np.cos(theta/2) - np.sin(theta/2)*1j, 0], 
                    [0, np.cos(theta/2) + np.sin(theta/2)*1j]])

def R_x(theta):
    return np.array([[np.cos(theta/2), np.sin(theta/2)*1j],
                    [np.sin(theta/2)*1j, np.cos(theta/2)]])

def R_y(theta):
    return np.array([[np.cos(theta/2), -(np.sin(theta/2))],
                    [np.sin(theta/2), (np.cos(theta/2))]])

def rotate_z(m, theta):
    """Rotate matrix m by theta about the z axis"""
    return R_z(theta) @ m @ op.adjoint(R_z(theta))

def rotate_x(m, theta):
    """Rotate matrix m by theta about the x axis"""
    return R_x(theta) @ m @ op.adjoint(R_x(theta))

def rotate_y(m, theta):
    """Rotate matrix m by theta about the y axis"""
    return R_y(theta) @ m @ op.adjoint(R_y(theta))

##########
## TESTS
##########
if __name__ == '__main__':
    theta = 3*np.pi/2

    print("===== PAULI_X about z =====")
    print("Actual: \n", rotate_z(PAULI_X, theta), "\n")
    print("Predicted: \n", np.cos(theta)*PAULI_X + np.sin(theta)*PAULI_Y, "\n")

    print("===== PAULI_Y about z =====")
    print("Actual: \n", rotate_z(PAULI_Y, theta), "\n")
    print("Predicted: \n", np.cos(theta)*PAULI_Y - np.sin(theta)*PAULI_X, "\n")

    print("===== PAULI_X about y =====")
    print("Actual: \n", rotate_y(PAULI_X, theta), "\n")
    print("Predicted: \n", np.cos(theta)*PAULI_X - np.sin(theta)*PAULI_Z, "\n")

    print("===== PAULI_Z about y =====")
    print("Actual: \n", rotate_y(PAULI_Z, theta), "\n")
    print("Predicted: \n", np.cos(theta)*PAULI_Z + np.sin(theta)*PAULI_X, "\n")

    print("===== PAULI_Y about x =====")
    print("Actual: \n", rotate_x(PAULI_Y, theta), "\n")
    print("Predicted: \n", np.cos(theta)*PAULI_Y - np.sin(theta)*PAULI_Z, "\n")

    print("===== PAULI_Z about x =====")
    print("Actual: \n", rotate_x(PAULI_Z, theta), "\n")
    print("Predicted: \n", np.cos(theta)*PAULI_Z + np.sin(theta)*PAULI_Y, "\n")

    ### Predicted - Actual (expected: all 0s for all ###
    # print((np.cos(theta)*PAULI_X + np.sin(theta)*PAULI_Y) - rotate_z(PAULI_X, theta))
    # print((np.cos(theta)*PAULI_Y - np.sin(theta)*PAULI_X) - rotate_z(PAULI_Y, theta))
    # print((np.cos(theta)*PAULI_X - np.sin(theta)*PAULI_Z) - rotate_y(PAULI_X, theta))  
    # print((np.cos(theta)*PAULI_Z + np.sin(theta)*PAULI_X) - rotate_y(PAULI_Z, theta))
    # print((np.cos(theta)*PAULI_Y - np.sin(theta)*PAULI_Z) - rotate_x(PAULI_Y, theta))
    # print((np.cos(theta)*PAULI_Z + np.sin(theta)*PAULI_Y) - rotate_x(PAULI_Z, theta))