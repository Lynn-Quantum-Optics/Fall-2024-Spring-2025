import numpy as np
import finding_states.states_and_witnesses as states
import tensorflow as tf
from inspect import signature

## NOTE: Be sure to import states_and_witnesses before this file
##       when working in a REPL (e.g. ipython) or notebook

#########################################
## State & Density Matrix Operations
#########################################

def ket(state):
    """
    Return the given state (represented as an array) as a ket

    Example: ket([1, 0]) -> [[1]
                             [0]]
    """
    return np.array(state, dtype=complex).reshape(-1,1)

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

# TODO: REVIEW THIS BY LOOKING AT SUMMER 2024 PAPER DRAFT 
#       FIGURES 4,5,6 (SOLID LINES) AND EQUATIONS 3,4,5
def minimize_witnesses(witness_class, rho=None, counts=None, num_guesses=10):
    """
    Calculates the minimum expectation values for each the witnesses specified
    in a given witness class for a given theoretical density matrix or for given
    experimental data

    Params:
        witness_class - a class of witnesses (possible values: W3, W5, W7, W8)
        rho           - the density matrix
        counts        - experimental data 
        num_guesses   - the number of initial guesses to use in minimization
        TODO: The W7 and W8 witnesses have not been implemented yet

    Returns: (min_thetas, min_vals)
        min_thetas - a list of the thetas corresponding to the minimum expectation values
        min_vals   - a list of the minimum expectation values
        NOTE: These are listed in the order of the witnesses (e.g. W3_1 first and W5_9 last)
    """
    # Lists to keep track of minimum expectation values and their corresponding parameters
    min_params = []
    min_vals = []

    # Get necessary witnesses
    ws = witness_class(rho=rho, counts=counts).get_witnesses()

    # Convert witness matrix function to TensorFlow
    def witness_matrix_tf(w):
        return tf.convert_to_tensor(w, dtype=tf.complex64)
    
    # Convert density matrix to TensorFlow
    rho_tf = tf.convert_to_tensor(rho, dtype=tf.complex64)

    def loss(W, params):
        """
        Loss function for minimization: tr(W @ rho)

        NOTE: this is the expectation value of W
        """
        witness = witness_matrix_tf(W(*params))
        return tf.linalg.trace(tf.matmul(witness, rho_tf))
    
    # minimize using the Adam optimizer
    optimizer = tf.optimizers.Adam(learning_rate=0.05)

    def optimize(W, params, threshold=1e-10, max_iters=1000):
        """
        Generic minimization loop that works for any number of minimization parameters

        Parameters:
        W                    - the witness whose expectation value will be minimized
        params               - the witness parameters to be minimized (i.e. theta, alpha, beta)
        threshold (optional) - smallest allowed change in the loss function (i.e. expectation value)
        max_iters (optional) - maximum number of iterations allowed in the optimization loop

        NOTE: threshold is 1e-6 by default
        NOTE: max_iters is 1000 by default
        """
        prev_loss = float("inf")

        # Optimization loop, stops when minimized value starts converging or when
        # the maximum iterations 
        for _ in range(max_iters):
            with tf.GradientTape() as tape:
                loss_value = loss(W, params)

            loss_real = tf.math.real(loss_value).numpy()
        
            # Check if minimized value has converged within the threshold
            if abs(prev_loss - loss_real) < threshold:
                break
            prev_loss = loss_real

            grads = tape.gradient(loss_value, params)
            for g, p in zip(grads, params):
                if g is not None: # Check if gradient exists (avoid NoneType errors)
                    optimizer.apply_gradients([(g, p)])
                    p.assign(tf.clip_by_value(p, 0.0, np.pi))  # enforce bounds

        return [p.numpy() for p in param_vars], loss_real


    # Minimize each witness
    for W in ws:
        # determine number of parameters to be minimzed
        num_params = len(signature(W).parameters)
        
        # Try 10 different initial guesses at random and use the best result
        min_val = float("inf")
        for _ in range(num_guesses):
            # initial guesses
            theta = tf.Variable(np.random.uniform(0, np.pi), dtype=tf.float64)
            alpha = tf.Variable(np.random.uniform(0, np.pi), dtype=tf.float64)
            beta = tf.Variable(np.random.uniform(0, np.pi), dtype=tf.float64)

            # use the right number of parameters
            param_vars = [theta, alpha, beta][:num_params]
            this_min_params, this_min_val = optimize(W, param_vars)

            if this_min_val < min_val:
                min_params = this_min_params
                min_val = this_min_val

        min_params.append(min_params)
        min_vals.append(min_val)


    return (min_params, min_vals)

if __name__ == "__main__":
    print("Operations Loaded.")