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