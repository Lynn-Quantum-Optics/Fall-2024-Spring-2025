import finding_states.states_and_witnesses as states
import finding_states.operations as op
import numpy as np

### ROTATION MATRICES TESTS ###
theta = np.pi/2
print("===== PAULI_X about z =====")
print("Actual: \n", op.rotate_z(states.PAULI_X, theta), "\n")
print("Predicted: \n", np.cos(theta)*states.PAULI_X + np.sin(theta)*states.PAULI_Y, "\n")

print("===== PAULI_Y about z =====")
print("Actual: \n", op.rotate_z(states.PAULI_Y, theta), "\n")
print("Predicted: \n", np.cos(theta)*states.PAULI_Y - np.sin(theta)*states.PAULI_X, "\n")

print("===== PAULI_X about y =====")
print("Actual: \n", op.rotate_y(states.PAULI_X, theta), "\n")
print("Predicted: \n", np.cos(theta)*states.PAULI_X - np.sin(theta)*states.PAULI_Z, "\n")

print("===== PAULI_Z about y =====")
print("Actual: \n", op.rotate_y(states.PAULI_Z, theta), "\n")
print("Predicted: \n", np.cos(theta)*states.PAULI_Z + np.sin(theta)*states.PAULI_X, "\n")

print("===== PAULI_Y about x =====")
print("Actual: \n", op.rotate_x(states.PAULI_Y, theta), "\n")
print("Predicted: \n", np.cos(theta)*states.PAULI_Y - np.sin(theta)*states.PAULI_Z, "\n")

print("===== PAULI_Z about x =====")
print("Actual: \n", op.rotate_x(states.PAULI_Z, theta), "\n")
print("Predicted: \n", np.cos(theta)*states.PAULI_Z + np.sin(theta)*states.PAULI_Y, "\n")

### Predicted - Actual (expected: all 0s for all ###
# print((np.cos(theta)*PAULI_X + np.sin(theta)*PAULI_Y) - rotate_z(PAULI_X, theta))
# print((np.cos(theta)*PAULI_Y - np.sin(theta)*PAULI_X) - rotate_z(PAULI_Y, theta))
# print((np.cos(theta)*PAULI_X - np.sin(theta)*PAULI_Z) - rotate_y(PAULI_X, theta))  
# print((np.cos(theta)*PAULI_Z + np.sin(theta)*PAULI_X) - rotate_y(PAULI_Z, theta))
# print((np.cos(theta)*PAULI_Y - np.sin(theta)*PAULI_Z) - rotate_x(PAULI_Y, theta))
# print((np.cos(theta)*PAULI_Z + np.sin(theta)*PAULI_Y) - rotate_x(PAULI_Z, theta))