import numpy as np
import states_and_witnesses as states
import tensorflow as tf

#########################################
## State & Density Matrix Operations
#########################################
def adjoint(state):
    """
    Returns the adjoint of a state vector
    """
    return np.conjugate(state).T

def get_rho(state):
    """
    Computes the density matrix given a 2-qubit state vector

    Param: state - the 2-qubit state vector
    """
    return state @ adjoint(state)

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

    PT = np.array(np.block([[b1.T, b2.T], [b3.T, b4.T]]))

    if subsys=='B':
        return PT
    elif subsys=='A':
        return PT.T

# Rotation operations
def rotate_z(m, theta):
    """Rotate matrix m by theta about the z axis"""
    return states.R_z(theta) @ m @ adjoint(states.R_z(theta))

def rotate_x(m, theta):
    """Rotate matrix m by theta about the x axis"""
    return states.R_x(theta) @ m @ adjoint(states.R_x(theta))

def rotate_y(m, theta):
    """Rotate matrix m by theta about the y axis"""
    return states.R_y(theta) @ m @ adjoint(states.R_y(theta))

def rotate_m(m, n):
    """
    Rotate matrix m with matrix n
    
    Params:
    m - the matrix to be rotated
    n - the matrix that is doing the rotation (i.e. R_z)

    NOTE: m and n must be of the same size
    """
    return n @ m @ adjoint(n)


##################
## MINIMIZATION
##################
def minimize_witnesses(witness_class, rho):
    min_thetas = []
    min_vals = []
    ws = witness_class([np.pi/2]*3, rho=rho).get_witnesses()

    # Convert witness matrix function to TensorFlow
    def witness_matrix_tf(w):
        return tf.convert_to_tensor(w, dtype=tf.float64)
    
    # Convert density matrix to TensorFlow
    rho_tf = tf.convert_to_tensor(rho, dtype=tf.float64)

    def loss(W, theta):
        """
        Loss function for minimization: tr(W @ rho)
        """
        witness = witness_matrix_tf(W(theta))
        return tf.linalg.trace(tf.matmul(witness, rho_tf))
    
    # Optimize using gradient descent
    optimizer = tf.optimizers.SGD(learning_rate=0.1) # stochastic gradient descent

    # initial guess
    theta = tf.Variable(np.random.uniform(0, np.pi), dtype=tf.float64)

    for i, W in enumerate(ws):
            theta.assign(np.random.uniform(0, np.pi)) # initial guess

            for _ in range(100):  # Run optimization with 100 iterations
                with tf.GradientTape() as tape:
                    loss_value = loss(W, theta)
                grad = tape.gradient(loss_value, theta)

                if grad is not None:  # Check if gradient exists (avoid NoneType errors)
                    optimizer.apply_gradients([(grad, theta)])
                    theta.assign(tf.clip_by_value(theta, 0.0, np.pi))  # Enforce bounds

            min_thetas.append(theta.numpy())
            min_vals.append(np.trace(W(theta) @ rho))

    return (min_thetas, min_vals)

if __name__ == "__main__":
    theta = np.pi/2

    print("===== PAULI_X about z =====")
    print("Actual: \n", rotate_z(states.PAULI_X, theta), "\n")
    print("Predicted: \n", np.cos(theta)*states.PAULI_X + np.sin(theta)*states.PAULI_Y, "\n")

    print("===== PAULI_Y about z =====")
    print("Actual: \n", rotate_z(states.PAULI_Y, theta), "\n")
    print("Predicted: \n", np.cos(theta)*states.PAULI_Y - np.sin(theta)*states.PAULI_X, "\n")

    print("===== PAULI_X about y =====")
    print("Actual: \n", rotate_y(states.PAULI_X, theta), "\n")
    print("Predicted: \n", np.cos(theta)*states.PAULI_X - np.sin(theta)*states.PAULI_Z, "\n")

    print("===== PAULI_Z about y =====")
    print("Actual: \n", rotate_y(states.PAULI_Z, theta), "\n")
    print("Predicted: \n", np.cos(theta)*states.PAULI_Z + np.sin(theta)*states.PAULI_X, "\n")

    print("===== PAULI_Y about x =====")
    print("Actual: \n", rotate_x(states.PAULI_Y, theta), "\n")
    print("Predicted: \n", np.cos(theta)*states.PAULI_Y - np.sin(theta)*states.PAULI_Z, "\n")

    print("===== PAULI_Z about x =====")
    print("Actual: \n", rotate_x(states.PAULI_Z, theta), "\n")
    print("Predicted: \n", np.cos(theta)*states.PAULI_Z + np.sin(theta)*states.PAULI_Y, "\n")

    ### Predicted - Actual (expected: all 0s for all ###
    # print((np.cos(theta)*PAULI_X + np.sin(theta)*PAULI_Y) - rotate_z(PAULI_X, theta))
    # print((np.cos(theta)*PAULI_Y - np.sin(theta)*PAULI_X) - rotate_z(PAULI_Y, theta))
    # print((np.cos(theta)*PAULI_X - np.sin(theta)*PAULI_Z) - rotate_y(PAULI_X, theta))  
    # print((np.cos(theta)*PAULI_Z + np.sin(theta)*PAULI_X) - rotate_y(PAULI_Z, theta))
    # print((np.cos(theta)*PAULI_Y - np.sin(theta)*PAULI_Z) - rotate_x(PAULI_Y, theta))
    # print((np.cos(theta)*PAULI_Z + np.sin(theta)*PAULI_Y) - rotate_x(PAULI_Z, theta))