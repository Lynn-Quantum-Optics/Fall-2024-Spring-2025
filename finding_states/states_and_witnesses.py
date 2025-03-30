import numpy as np
import operations as op
import tensorflow as tf


#######################
## QUANTUM STATES
#######################

### Jones/Column Vectors & Density Matrices ###
HH = np.array([1, 0, 0, 0]).reshape((4,1))
HV = np.array([0, 1, 0, 0]).reshape((4,1))
VH = np.array([0, 0, 1, 0]).reshape((4,1))
VV = np.array([0, 0, 0, 1]).reshape((4,1))
HH_RHO = op.get_rho(HH)
HV_RHO = op.get_rho(HV)
VH_RHO = op.get_rho(VH)
VV_RHO = op.get_rho(VV)

### Bell States & Density Matrices ###
PHI_P = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)]).reshape((4,1))
PHI_M = np.array([1/np.sqrt(2), 0, 0, -1/np.sqrt(2)]).reshape((4,1))
PSI_P = np.array([0, 1/np.sqrt(2),  1/np.sqrt(2), 0]).reshape((4,1))
PSI_M = np.array([0, 1/np.sqrt(2),  -1/np.sqrt(2), 0]).reshape((4,1))
PHI_P_RHO = op.get_rho(PHI_P)
PHI_M_RHO = op.get_rho(PHI_M)
PSI_P_RHO = op.get_rho(PSI_P)
PSI_M_RHO = op.get_rho(PSI_M)

### Werner States ###
def werner(p):
    """
    Returns Werner state with parameter p
    """
    return p*PHI_P + (1-p)*np.eye(4)/4

### Eritas's States (see spring 2023 writeup) ###
def E_PSI(eta, chi, rho = False):
    """
    Returns the state cos(eta)PSI_P + e^(i*chi)*sin(eta)PSI_M
    as either a density matrix or vector state

    Params:
    eta, chi       - parameters of the state
    rho (optional) - if true, return state as a density matrix, 
                     return as vector otherwise
    """
    state = np.cos(eta)*PSI_P + np.exp(chi*1j)*np.sin(eta)*PSI_M

    if rho:
        return op.get_rho(state)
    return state

def E_PHI(eta, chi, rho = False):
    """
    Returns the state cos(eta)PHI_P + e^(i*chi)*sin(eta)PHI_M
    as either a density matrix or vector state

    Params:
    eta, chi       - parameters of the state
    rho (optional) - if true, return state as a density matrix, 
                     return as vector otherwise
    """
    state = np.cos(eta)*PHI_P + np.exp(chi*1j)*np.sin(eta)*PHI_M
    if rho:
        return op.get_rho(state)
    return state

## TODO: LOOK AT SUMMER 2024 PAPER DRAFT FIGURES 4,5,6 (SOLID LINES) AND EQUATIONS 3,4,5


##################
## MATRICES
##################

### Pauli Matrices ###
PAULI_X = np.array([[0,1], [1, 0]])
PAULI_Y = np.array([[0, -1j], [1j, 0]])
PAULI_Z = np.array([[1,0], [0,-1]])
IDENTITY = np.array([[1,0], [0,1]])

### Rotation Matrices ###
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


###########################
## ENTANGLEMENT WITNESSES
###########################

## Keep track of count indices
# int to str, i.e. COUNT[0] = 'HH'
COUNTS = ['HH', 'HV', 'HD', 'HA', 'HR', 'HL', 'VH', 'VV', 'VD', 'VA', 'VR', 'VL', 
          'DH', 'DV', 'DD', 'DA', 'DR', 'DL', 'AH', 'AV', 'AD', 'AA', 'AR', 'AL', 
          'RH', 'RV', 'RD', 'RA', 'RR', 'RL', 'LH', 'LV', 'LD', 'LA', 'LR', 'LL']
# str to int, i.e. COUNTS_INDEX['HH'] = 0
COUNTS_INDEX = {count: index for index, count in enumerate(COUNTS)} 

class RiccardiWitness:
    """
    Riccardi witnesses (W)

    Attributes: 
    theta             - the parameter for the rank-1 projectors
    rho (optional)    - the density matrix for the entangled photon state
    counts (optional) - np array of photon counts and uncertainties from experimental data
    expt              - whether or not to calculate the experimental density matrix using counts

    NOTE: If counts is given, experimental calculations will be used
    NOTE: If rho is given, theoretical calculations will be used
    NOTE: expt is True by default
    """
    def __init__(self, theta, rho = None, counts=None, expt=True):
        self.theta = theta
        self.counts = counts
        self.expt = expt

        # counts not given, so we want to use the given theoretical rho
        if not counts:
            self.rho = rho

        else:
            # counts given, but we just want witness expectation values, so
            # rho is not desired in order to minimize the number of measurements
            assert rho is None, "ERROR: counts was given, so rho should not be given"

            # store individual counts in variables
            self.hh, self.hv = counts[COUNTS_INDEX['HH']], counts[COUNTS_INDEX['HV']]
            self.hd, self.ha = counts[COUNTS_INDEX['HD']], counts[COUNTS_INDEX['HA']]
            self.hr, self.hl = counts[COUNTS_INDEX['HR']], counts[COUNTS_INDEX['HL']]
            self.vh, self.vv = counts[COUNTS_INDEX['VH']], counts[COUNTS_INDEX['VV']]
            self.vd, self.va = counts[COUNTS_INDEX['VD']], counts[COUNTS_INDEX['VA']]
            self.vr, self.vl = counts[COUNTS_INDEX['VR']], counts[COUNTS_INDEX['VL']]
            self.dh, self.dv = counts[COUNTS_INDEX['DH']], counts[COUNTS_INDEX['DV']]
            self.dd, self.da = counts[COUNTS_INDEX['DD']], counts[COUNTS_INDEX['DA']]
            self.dr, self.dl = counts[COUNTS_INDEX['DR']], counts[COUNTS_INDEX['DL']]
            self.ah, self.av = counts[COUNTS_INDEX['AH']], counts[COUNTS_INDEX['AV']]
            self.ad, self.aa = counts[COUNTS_INDEX['AD']], counts[COUNTS_INDEX['AA']]
            self.ar, self.al = counts[COUNTS_INDEX['AR']], counts[COUNTS_INDEX['AL']]
            self.rh, self.rv = counts[COUNTS_INDEX['RH']], counts[COUNTS_INDEX['RV']]
            self.rd, self.ra = counts[COUNTS_INDEX['RD']], counts[COUNTS_INDEX['RA']]
            self.rr, self.rl = counts[COUNTS_INDEX['RR']], counts[COUNTS_INDEX['RL']]
            self.lh, self.lv = counts[COUNTS_INDEX['LH']], counts[COUNTS_INDEX['LV']]
            self.ld, self.la = counts[COUNTS_INDEX['LD']], counts[COUNTS_INDEX['LA']]
            self.lr, self.ll = counts[COUNTS_INDEX['LR']], counts[COUNTS_INDEX['LL']]

            self.stokes = self.calculate_stokes()

            # counts given, and we want to calculate the experimental rho 
            # NOTE: this requires using a full tomography
            if expt:
                self.rho = self.expt_rho()                

        # parameters
        self.a = np.cos(self.theta)
        self.b = np.sin(self.theta)

    def W1(self, theta=None):
        """
        The first W^3 witness

        Params:
        theta - a theta to override the class parameter theta primarily
                used for minimization
        """
        # expectation value (for when we don't want to use full tomography)
        if self.counts and not self.expt:
            return 0.25*(self.stokes[0] + self.stokes[15] + 
                        (self.a**2 - self.b**2)*self.stokes[5] + 
                        (self.a**2 - self.b**2)*self.stokes[10] + 
                        2*self.a*self.b*(self.stokes[12] + self.stokes[3]))
        

        # W1 matrix (for when we use full tomography & calculate rho)
        else:
            if theta is not None:
                phi1 = np.cos(theta)*PHI_P + np.sin(theta)*PHI_M
            else:
                phi1 = self.a*PHI_P + self.b*PHI_M
            return op.partial_transpose(phi1 * op.adjoint(phi1))
        
    def W2(self, theta = None):
        if self.counts and not self.expt:
            return 0.25*(self.stokes[0] - self.stokes[15] + 
                        (self.a**2 - self.b**2)*self.stokes[5] - 
                        (self.a**2 - self.b**2)*self.stokes[10] + 
                        2*self.a*self.b*(self.stokes[12] - self.stokes[3]))
        else:
            if theta is not None:
                phi2 = np.cos(theta)*PSI_P + np.sin(theta)*PSI_M
            else:
                phi2 = self.a*PSI_P + self.b*PSI_M
            return op.partial_transpose(phi2 * op.adjoint(phi2))
        
    def W3(self, theta):
        if self.counts and not self.expt:
            return 0.25*(self.stokes[0] + self.stokes[5] + 
                         (self.a**2 - self.b**2)*self.stokes[15] + 
                         (self.a**2 - self.b**2)*self.stokes[10] + 
                         2*self.a*self.b*(self.stokes[4] + self.stokes[1]))
        else:
            if theta is not None:
                phi3 = np.cos(theta)*PHI_P + np.sin(theta)*PSI_P
            else:
                phi3 = self.a*PHI_P + self.b*PSI_P
            return op.partial_transpose(phi3 * op.adjoint(phi3))

    def W4(self, theta):
        if self.counts and not self.expt:
            return 0.25*(self.stokes[0] - self.stokes[5] + 
                         (self.a**2 - self.b**2)*self.stokes[15] - 
                         (self.a**2 - self.b**2)*self.stokes[10] - 
                         2*self.a*self.b*(self.stokes[4] - self.stokes[1]))
        else:
            if theta is not None:
                phi4 = np.cos(theta)*PHI_M + np.sin(theta)*PSI_M
            else:
                phi4 = self.a*PHI_M + self.b*PSI_M
            return op.partial_transpose(phi4 * op.adjoint(phi4))
        
    def W5(self, theta):
        if self.counts and not self.expt:
            return 0.25*(self.stokes[0] + self.stokes[10] + 
                         (self.a**2 - self.b**2)*self.stokes[15] + 
                         (self.a**2 - self.b**2)*self.stokes[5] - 
                         2*self.a*self.b*(self.stokes[8] + self.stokes[2]))
        else:
            if theta is not None:
                phi5 = np.cos(theta)*PHI_P + np.sin(theta)*PSI_M
            else:
                phi5 = self.a*PHI_P + 1j*self.b*PSI_M
            return op.partial_transpose(phi5 * op.adjoint(phi5))
        
    def W6(self, theta):
        if self.counts and not self.expt:
            return 0.25*(self.stokes[0] - self.stokes[10] + 
                         (self.a**2 - self.b**2)*self.stokes[15] - 
                         (self.a**2 - self.b**2)*self.stokes[5] + 
                         2*self.a*self.b*(self.stokes[8] - self.stokes[2]))
        else:
            if theta is not None:
                phi6 = np.cos(theta)*PHI_M + np.sin(theta)*PSI_P
            else:
                phi6 = self.a*PHI_M + 1j*self.b*PSI_P
            return op.partial_transpose(phi6 * op.adjoint(phi6))
        
    def expt_rho(self):
        """Calculates the theoretical density matrix"""
        
        # Calculate tensor'd Pauli matrices
        pauli = [np.kron(IDENTITY, IDENTITY), np.kron(IDENTITY, PAULI_X), np.kron(IDENTITY, PAULI_Y),
                 np.kron(IDENTITY, PAULI_Z), np.kron(PAULI_X, IDENTITY), np.kron(PAULI_X, PAULI_X),
                 np.kron(PAULI_X, PAULI_Y), np.kron(PAULI_X, PAULI_Z), np.kron(PAULI_Y, IDENTITY),
                 np.kron(PAULI_Y, PAULI_X), np.kron(PAULI_Y, PAULI_Y), np.kron(PAULI_Y, PAULI_Z), 
                 np.kron(PAULI_Z, IDENTITY), np.kron(PAULI_Z, PAULI_X), np.kron(PAULI_Z, PAULI_Y), 
                 np.kron(PAULI_Z, PAULI_Z)
                ]
        
        rho = np.zeros((4, 4), dtype='complex128')
        for i in range(len(pauli)):
            rho += self.stokes[i] * pauli[i]
        
        return 0.25 * rho

    def calculate_stokes(self):
        """
        Calculates Stokes parameters
        
        NOTE: The order of this list is the same as S_{i, j} as listed in the 
              1-photon and 2-photon states from Beili Nora Hu paper (page 9)
        """
        assert self.counts is not None, "ERROR: counts not given"

        stokes = [1,
            (self.dd - self.da + self.ad - self.aa)/(self.dd + self.da + self.ad + self.aa),
            (self.rr + self.lr - self.rl - self.ll)/(self.rr + self.lr + self.rl + self.ll),
            (self.hh - self.hv - self.vh - self.vv)/(self.hh + self.hv + self.vh + self.vv),
            (self.dd + self.da - self.ad - self.aa)/(self.dd + self.da + self.ad + self.da),
            (self.dd - self.da - self.ad + self.aa)/(self.dd + self.da + self.ad + self.aa),
            (self.dr - self.dl - self.ar + self.al)/(self.dr + self.dl + self.ar + self.al),
            (self.dh - self.dv - self.ah + self.av)/(self.dh + self.dv + self.ah + self.av),
            (self.rr - self.lr + self.rl - self.ll)/(self.rr + self.lr + self.rl + self.ll),
            (self.rd - self.ra - self.ld + self.la)/(self.rd + self.ra + self.ld + self.la),
            (self.rr - self.rl - self.lr + self.ll)/(self.rr + self.rl + self.lr + self.ll),
            (self.rh - self.rv - self.lh + self.lv)/(self.rh + self.rv + self.lh + self.lv),
            (self.hh + self.hv - self.vh - self.vv)/(self.hh + self.hv + self.vh + self.vv),
            (self.hd - self.ha - self.vd + self.va)/(self.hd + self.ha + self.vd + self.va),
            (self.hr - self.hl - self.vr + self.vl)/(self.hr + self.hl + self.vr + self.vl),
            (self.hh - self.hv - self.vh + self.vv)/(self.hh + self.hv + self.vh + self.vv)
        ]
        
        return stokes

    def get_witnesses(self):
        """
        Returns the value to be minimized to find the expectation value of W

        Params:
            w   - the witness matrix
            rho - the density matrix
        """
        ws = [self.W1(), self.W2(), self.W3(), self.W4(), self.W5(), self.W6()]
        vals = [None for i in range(len(ws))]

        # When we don't want to use rho
        if self.counts and not self.expt:
            return ws
        
        # We use rho
        else: 
            for i, w in enumerate(ws):
                vals[i] = np.trace(w @ self.rho).real
        
        return vals
    
    # TODO: Review this
    def minimize_witnesses(self):
        min_thetas = []
        min_vals = []
        all_W = [self.W1, self.W2, self.W3, self.W4, self.W5, self.W6]

        # Convert witness matrix function to TensorFlow
        def witness_matrix_tf(w):
            return tf.convert_to_tensor(w, dtype=tf.float64)

        # Convert density matrix to TensorFlow
        rho_tf = tf.convert_to_tensor(self.rho, dtype=tf.float64)

        # Loss function is tr(W @ rho)
        def loss(W, theta):
            witness = witness_matrix_tf(W(theta))
            return tf.linalg.trace(tf.matmul(witness, rho_tf))

        # Optimize using gradient descent
        optimizer = tf.optimizers.SGD(learning_rate=0.1) # stochastic gradient descent

        # initial guess
        theta = tf.Variable(np.random.uniform(0, np.pi), dtype=tf.float64)

        for i, W in enumerate(all_W):
            theta.assign(np.random.uniform(0, np.pi)) # initial guess

            for _ in range(100):  # Run optimization with 100 iterations
                with tf.GradientTape() as tape:
                    loss_value = loss(W, theta)
                grad = tape.gradient(loss_value, theta)

                if grad is not None:  # Check if gradient exists (avoid NoneType errors)
                    optimizer.apply_gradients([(grad, theta)])
                    theta.assign(tf.clip_by_value(theta, 0.0, np.pi))  # Enforce bounds

            min_thetas.append(theta.numpy())
            min_vals.append(np.trace(W(theta) @ self.rho))
        
        return (min_thetas, min_vals)


class Wp(RiccardiWitness):
    """
    W' witnesses, calculated with Paco's rotations

    Attributes:
    counts (optional) - np array of photon counts and uncertainties from experimental data
    rho (optional)    - the density matrix for the entangled photon state
    expt              - whether or not to calculate the experimental density matrix using counts
    angles            - a list of angles to be used in rotations
        + angles[0] = theta (rank-1 projector parameter)
        + angles[1] = alpha
        + angles[2] = beta

        
    NOTE: this class inherits from RiccardiWitnesses, so all methods in that class can be used here
    """
    def __init__(self, angles, rho=None, counts=None, expt=True):
        assert len(angles) == 3, "ERROR: 3 angles must be provided (theta, alpha, beta)"

        super().__init__(angles[0], rho=rho, counts=counts, expt=expt)
        self.angles = angles
        self.alpha = angles[1]
        self.beta = angles[2]

    def Wp1(self):
        w1 = super().W1()
        rotation = np.kron(R_z(self.alpha), IDENTITY)
        return op.rotate_m(w1, rotation)
    
    def Wp2(self):
        w2 = super().W2()
        rotation = np.kron(R_z(self.alpha), IDENTITY)
        return op.rotate_m(w2, rotation)
    
    def Wp3(self):
        w3 = super().W3()
        rotation = np.kron(R_z(self.alpha), R_z(self.beta))
        return op.rotate_m(w3, rotation)
    
    def Wp4(self):
        w3 = super().W3()
        rotation = np.kron(R_x(self.alpha), IDENTITY)
        return op.rotate_m(w3, rotation)
    
    def Wp5(self):
        w4 = super().W4()
        rotation = np.kron(R_x(self.alpha), IDENTITY)
        return op.rotate_m(w4, rotation)
    
    def Wp6(self):
        w1 = super().W1()
        rotation = np.kron(R_x(self.alpha), R_y(self.beta))
        return op.rotate_m(w1, rotation)
    
    def Wp7(self):
        w5 = super().W5()
        rotation = np.kron(R_y(self.alpha), IDENTITY)
        return op.rotate_m(w5, rotation)
    
    def Wp8(self):
        w6 = super().W6()
        rotation = np.kron(R_y(self.alpha), IDENTITY)
        return op.rotate_m(w6, rotation)

    def Wp9(self):
        w1 = super().W1()
        rotation = np.kron(R_y(self.alpha), R_y(self.beta))
        return op.rotate_m(w1, rotation)
    
    def minimize_witnesses(self):
        min_thetas, min_vals = super().minimize_witnesses()

        all_Wp = [self.Wp1, self.Wp2, self.Wp3, 
                  self.Wp4, self.Wp5, self.Wp6, 
                  self.Wp7, self.Wp8, self.Wp9]
        
        # Wp3,6,9 have 3, the rest of 2
        return (min_thetas, min_vals)
    
    def __str__(self):
        return (
            f'Parameter: {self.param}\n'
            f'Counts: {self.counts}\n'
            f'Angles: {self.angles}'
        )



if __name__ == '__main__':
    print("States, Matrices, and Witnesses Loaded.")