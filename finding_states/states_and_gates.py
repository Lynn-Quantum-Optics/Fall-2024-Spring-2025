import numpy as np
import operations as op

### Column vectors ###
HH = np.array([1, 0, 0, 0]).reshape((4,1))
HV = np.array([0, 1, 0, 0]).reshape((4,1))
VH = np.array([0, 0, 1, 0]).reshape((4,1))
VV = np.array([0, 0, 0, 1]).reshape((4,1))

### Pauli Gates ###
PAULI_Z = np.array([[1,0], [0,-1]])
PAULI_X = np.array([[0,1], [1, 0]])
PAULI_Y = np.array([[0, -1j], [1j, 0]])

### Bell States ###
PHI_P = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)])
PHI_M = np.array([1/np.sqrt(2), 0, 0, -1/np.sqrt(2)])
PSI_P = np.array([0, 1/np.sqrt(2),  1/np.sqrt(2), 0])
PSI_M = np.array([0, 1/np.sqrt(2),  -1/np.sqrt(2), 0])


### Rotation Matrices & Functions ###
#
# Params: 
#  theta - the angle to rotate by
def R_z(theta):
    return np.array([[np.cos(theta/2) - np.sin(theta/2)*1j, 0], 
                    [0, np.cos(theta/2) + np.sin(theta/2)*1j]])

def R_x(theta):
    return np.array([[np.cos(theta/2), np.sin(theta/2)*1j],
                    [np.sin(theta/2)*1j, np.cos(theta/2)]])

def R_y(theta):
    return np.array([[np.cos(theta/2), -(np.sin(theta/2))],
                    [np.sin(theta/2), (np.cos(theta/2))]])


### Riccardi witnesses (W) ###
#
# Params: 
#  param - the parameter for the rank-1 projectors
#  expt  - bool, whether to use experimental data or not  
def W1(param, expt=False):
    a,b = np.cos(param), np.sin(param)

    if expt:
        return 0
    else:
        phi1 = a*PHI_P + b*PHI_M
        return op.partial_transpose(phi1 * op.adjoint(phi1))

def W2(param, expt=False):
    a,b = np.cos(param), np.sin(param)

    if expt:
        return 0
    else:
        phi2 = a*PSI_P + b*PSI_M
        return op.partial_transpose(phi2 * op.adjoint(phi2))

def W3(param, expt=False):
    a,b = np.cos(param), np.sin(param)

    if expt:
        return 0
    else:
        phi3 = a*PHI_P + b*PSI_P
        return op.partial_transpose(phi3 * op.adjoint(phi3))

def W4(param, expt=False):
    a,b = np.cos(param), np.sin(param)

    if expt:
        return 0
    else:
        phi4 = a*PHI_M + b*PSI_M
        return op.partial_transpose(phi4 * op.adjoint(phi4))

def W5(param, expt=False):
    a,b = np.cos(param), np.sin(param)

    if expt:
        return 0
    else:
        phi5 = a*PHI_P + 1j*b*PSI_M
        return op.partial_transpose(phi5 * op.adjoint(phi5))

def W6(param, expt=False):
    a,b = np.cos(param), np.sin(param)

    if expt:
        return 0
    else:
        phi6 = a*PHI_M + 1j*b*PSI_P
        return op.partial_transpose(phi6 * op.adjoint(phi6))


### W' witnesses ###
#
# Params: 
#  params - a list of 3 parameters (theta, alpha, beta) for the projectors
#  expt   - bool, whether to use experimental data or not  
def Wp1(params, expt=False):
    theta, alpha = params[0], params[1]

    if expt:
        return 0
    else:
        phi1_p = np.cos(theta)*PHI_P + np.exp(1j*alpha)*np.sin(theta)*PHI_M
        return op.partial_transpose(phi1_p * op.adjoint(phi1_p))
    
def Wp2(params, expt=False):
    theta, alpha = params[0], params[1]

    if expt:
        return 0
    else:
        phi2_p = np.cos(theta)*PSI_P + np.exp(1j*alpha)*np.sin(theta)*PSI_M
        return op.partial_transpose(phi2_p * op.adjoint(phi2_p))
    
def Wp3(params, expt=False):
    theta, alpha, beta = params[0], params[1]

    if expt:
        return 0
    else:
        phi3_p = (1/np.sqrt(2) * 
                  (np.cos(theta)*HH + np.exp(1j*(beta - alpha))*np.sin(theta)*HV + 
                   np.exp(1j*alpha)*np.sin(theta)*VH + np.exp(1j*beta)*np.cos(theta)*VV))
        return op.partial_transpose(phi3_p * op.adjoint(phi3_p))

def Wp4(params, expt=False):
    theta, alpha = params[0], params[1]

    if expt:
        return 0
    else:
        phi4_p = np.cos(theta)*PHI_P + np.exp(1j*alpha)*np.sin(theta)*PSI_P
        return op.partial_transpose(phi4_p * op.adjoint(phi4_p))
    

def Wp5(params, expt=False):
    theta, alpha = params[0], params[1]

    if expt:
        return 0
    else:
        phi5_p = np.cos(theta)*PHI_M + np.exp(1j*alpha)*np.sin(theta)*PSI_M
        return op.partial_transpose(phi5_p * op.adjoint(phi5_p))
    
def Wp6(params, expt=False):
    theta, alpha, beta = params[0], params[1], params[2]

    if expt:
        return 0
    else:
        phi6_p = (np.cos(theta)*np.cos(alpha)*HH + 1j * np.cos(theta)*np.sin(alpha)*HV 
                  + 1j * np.sin(theta)*np.sin(beta)*VH + np.sin(theta)*np.cos(beta)*VV)
        return op.partial_transpose(phi6_p * op.adjoint(phi6_p))
    
def Wp7(params, expt=False):
    theta, alpha = params[0], params[1]

    if expt:
        return 0
    else:
        phi7_p = np.cos(theta)*PHI_P + np.exp(1j*alpha)*np.sin(theta)*PSI_M
        return op.partial_transpose(phi7_p * op.adjoint(phi7_p))
    
def Wp8(params, expt=False):
    theta, alpha = params[0], params[1]

    if expt:
        return 0
    else:
        phi8_p = np.cos(theta)*PHI_M + np.exp(1j*alpha)*np.sin(theta)*PSI_P
        return op.partial_transpose(phi8_p * op.adjoint(phi8_p))

def Wp9(params, expt=False):
    theta, alpha, beta = params[0], params[1]

    if expt:
        return 0
    else:
        phi9_p = (np.cos(theta)*np.cos(alpha)*HH + np.cos(theta)*np.sin(alpha)*HV
                   + np.sin(theta)*np.sin(beta)*VH + np.sin(theta)*np.cos(beta)*VV)
        return op.partial_transpose(phi9_p * op.adjoint(phi9_p))
    

##########
## TESTS
##########
if __name__ == '__main__':
    theta = 3*np.pi/2

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