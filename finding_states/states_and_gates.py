import numpy as np

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
    return R_z(theta) @ m @ R_z(theta).getH()

def rotate_x(m, theta):
    """Rotate matrix m by theta about the x axis"""
    return R_x(theta) @ m @ R_x(theta).getH()

def rotate_y(m, theta):
    """Rotate matrix m by theta about the y axis"""
    return R_y(theta) @ m @ R_y(theta).getH()

##########
## TESTS
##########
if __name__ == '__main__':
    theta = np.pi/2

    print("===== PAULI_X about z =====")
    print("Actual: \n", rotate_z(PAULI_X, theta), "\n")
    print("Predicted: \n", np.cos(theta)*PAULI_X - np.sin(theta)*PAULI_Y, "\n")

    print("===== PAULI_Y about z =====")
    print("Actual: \n", rotate_z(PAULI_X, theta), "\n")
    print("Predicted: \n", np.cos(theta)*PAULI_Y + np.sin(theta)*PAULI_X, "\n")

    print("===== PAULI_X about y =====")
    print("Actual: \n", rotate_z(PAULI_X, theta), "\n")
    print("Predicted: \n", np.cos(theta)*PAULI_X - np.sin(theta)*PAULI_Z, "\n")

    print("===== PAULI_Z about y =====")
    print("Actual: \n", rotate_z(PAULI_X, theta), "\n")
    print("Predicted: \n", np.cos(theta)*PAULI_Z + np.sin(theta)*PAULI_X, "\n")

    print("===== PAULI_Y about x =====")
    print("Actual: \n", rotate_z(PAULI_X, theta), "\n")
    print("Predicted: \n", np.cos(theta)*PAULI_Y - np.sin(theta)*PAULI_Z, "\n")

    print("===== PAULI_Z about x =====")
    print("Actual: \n", rotate_z(PAULI_X, theta), "\n")
    print("Predicted: \n", np.cos(theta)*PAULI_Z + np.sin(theta)*PAULI_Y, "\n")
