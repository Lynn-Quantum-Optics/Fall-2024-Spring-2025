import numpy as np
import finding_states.operations as op
import tensorflow as tf

#######################
## QUANTUM STATES
#######################

# Single-qubit States
H = op.ket([1,0])
V = op.ket([0,1])
R = op.ket([1/np.sqrt(2) * 1, 1/np.sqrt(2) * (-1j)])
L = op.ket([1/np.sqrt(2) * 1, 1/np.sqrt(2) * (1j)])
D = op.ket([1/np.sqrt(2) * 1, 1/np.sqrt(2) * (1)])
A = op.ket([1/np.sqrt(2) * 1, 1/np.sqrt(2) * (-1)])

### Jones/Column Vectors & Density Matrices ###
HH = op.ket([1, 0, 0, 0])
HV = op.ket([0, 1, 0, 0])
VH = op.ket([0, 0, 1, 0])
VV = op.ket([0, 0, 0, 1])
HH_RHO = op.get_rho(HH)
HV_RHO = op.get_rho(HV)
VH_RHO = op.get_rho(VH)
VV_RHO = op.get_rho(VV)

### Bell States & Density Matrices ###
PHI_P = (HH + VV)/np.sqrt(2)
PHI_M = (HH - VV)/np.sqrt(2)
PSI_P = (HV + VH)/np.sqrt(2)
PSI_M = (HV - VH)/np.sqrt(2)
PHI_P_RHO = op.get_rho(PHI_P)
PHI_M_RHO = op.get_rho(PHI_M)
PSI_P_RHO = op.get_rho(PSI_P)
PSI_M_RHO = op.get_rho(PSI_M)

### Eritas's States (see spring 2023 writeup) ###
def E0_PSI(eta, chi, rho = False):
    """
    Returns the state of form cos(eta)PSI_P + e^(i*chi)*sin(eta)PSI_M
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

def E0_PHI(eta, chi, rho = False):
    """
    Returns the state of form cos(eta)PHI_P + e^(i*chi)*sin(eta)PHI_M
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

def E1(eta, chi, rho = False):
    """
    Returns the state of the form 
    1/sqrt(2) * (cos(eta)*(PSI_P + iPSI_M) + e^(i*chi)*sin(eta)*(PHI_P + iPHI_M))
    as either a density matrix or vector state

    Params:
    eta, chi       - parameters of the state
    rho (optional) - if true, return state as a density matrix, 
                     return as vector otherwise
    """
    state = 1/np.sqrt(2) * (np.cos(eta)*(PSI_P + PSI_M*1j) + np.sin(eta)*np.exp(chi*1j)*(PHI_P + PHI_M*1j))
    if rho:
        return op.get_rho(state)
    return state


### Amplitude Damped States ###
def ADS(gamma):
    """
    Returns the amplitude damped state with parameter gamma
    """
    return np.array([[.5, 0, 0, .5*np.sqrt(1-gamma)], 
                     [0, 0, 0, 0], [0, 0, .5*gamma, 0], 
                     [.5*np.sqrt(1-gamma), 0, 0, .5-.5*gamma]])

### Sample state to illustrate power of W5 over W3 ###
def sample_state(phi):
    """
    Returns a state with parameter phi
    """
    ex1 = np.cos(phi)*np.kron(H,D) - np.sin(phi)*np.kron(V,A)
    return op.get_rho(ex1)

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

class W3:
    """
    W3 (Riccardi) witnesses. These use local measurements on 3 Pauli bases.

    Attributes: 
    rho (optional)    - the density matrix for the entangled photon state
    counts (optional) - np array of photon counts and uncertainties from experimental data
    expt              - whether or not to calculate the experimental density matrix using counts

    NOTE: One of rho or counts must be given
    NOTE: If counts is given, experimental calculations will be used
    NOTE: If rho is given, theoretical calculations will be used
    NOTE: expt is True by default
    """
    def __init__(self, rho = None, counts=None, expt=True):
        self.counts = counts
        self.expt = expt
            
        # counts not given, so we want to use the given theoretical rho
        if not counts:
            assert rho is not None, "ERROR: counts not given, so rho should be given"
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

    def W1(self, theta):
        """
        The first W3 witness

        Params:
        theta - the parameter for the rank-1 projector
        """
        a = np.cos(theta)
        b = np.sin(theta)

        # expectation value (for when we don't want to use full tomography)
        if self.counts and not self.expt:
            return 0.25*(self.stokes[0] + self.stokes[15] + 
                        (a**2 - b**2)*self.stokes[5] + 
                        (a**2 - b**2)*self.stokes[10] + 
                        2*a*b*(self.stokes[12] + self.stokes[3]))
        
        # W1 as a matrix (for when we use full tomography & thus have a rho)
        phi1 = a*PHI_P + b*PHI_M
        return op.partial_transpose(phi1 * op.adjoint(phi1))
        
    def W2(self, theta):
        a = np.cos(theta)
        b = np.sin(theta)

        if self.counts and not self.expt:
            return 0.25*(self.stokes[0] - self.stokes[15] + 
                        (a**2 - b**2)*self.stokes[5] - 
                        (a**2 - b**2)*self.stokes[10] + 
                        2*a*b*(self.stokes[12] - self.stokes[3]))
        
        phi2 = a*PSI_P + b*PSI_M
        return op.partial_transpose(phi2 * op.adjoint(phi2))
        
    def W3(self, theta = None):
        a = np.cos(theta)
        b = np.sin(theta)

        if self.counts and not self.expt:
            return 0.25*(self.stokes[0] + self.stokes[5] + 
                         (a**2 - b**2)*self.stokes[15] + 
                         (a**2 - b**2)*self.stokes[10] + 
                         2*a*b*(self.stokes[4] + self.stokes[1]))
        
        phi3 = a*PHI_P + b*PSI_P
        return op.partial_transpose(phi3 * op.adjoint(phi3))

    def W4(self, theta = None):
        a = np.cos(theta)
        b = np.sin(theta)

        if self.counts and not self.expt:
            return 0.25*(self.stokes[0] - self.stokes[5] + 
                         (a**2 - b**2)*self.stokes[15] - 
                         (a**2 - b**2)*self.stokes[10] - 
                         2*a*b*(self.stokes[4] - self.stokes[1]))

        phi4 = a*PHI_M + b*PSI_M
        return op.partial_transpose(phi4 * op.adjoint(phi4))
        
    def W5(self, theta = None):
        a = np.cos(theta)
        b = np.sin(theta)

        if self.counts and not self.expt:
            return 0.25*(self.stokes[0] + self.stokes[10] + 
                         (a**2 - b**2)*self.stokes[15] + 
                         (a**2 - b**2)*self.stokes[5] - 
                         2*a*b*(self.stokes[8] + self.stokes[2]))

        phi5 = a*PHI_P + 1j*b*PSI_M
        return op.partial_transpose(phi5 * op.adjoint(phi5))
        
    def W6(self, theta = None):
        a = np.cos(theta)
        b = np.sin(theta)

        if self.counts and not self.expt:
            return 0.25*(self.stokes[0] - self.stokes[10] + 
                         (a**2 - b**2)*self.stokes[15] - 
                         (a**2 - b**2)*self.stokes[5] + 
                         2*a*b*(self.stokes[8] - self.stokes[2]))

        phi6 = a*PHI_M + 1j*b*PSI_P
        return op.partial_transpose(phi6 * op.adjoint(phi6))
        
    def expt_rho(self):
        """Calculates the theoretical density matrix"""
        
        # Get tensor'd Pauli matrices
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

    def get_witnesses(self, ops=True, theta=None):
        """
        Returns the expectation values of all 6 witnesses with a given theta
        or the operators themselves

        Params:
        ops (optional)   - if true, return the witness operators as functions, return expectation values otherwise
        theta (optional) - the theta value to calculate expectation values for when ops is false
        
        NOTE: ops is true by default
        NOTE: theta is not given by default
        NOTE: ops takes precedence over theta, so if ops is true, only the witness operators will be
              returned (and no expectation values for the given theta, even if theta is given)
        """
        ws = [self.W1, self.W2, self.W3, self.W4, self.W5, self.W6]
    
        ## Return the operators if specified
        if ops:
            return ws

        ## Otherwise, return the expectation values with the given theta
        ws = [w(theta) for w in ws]
        vals = []

        # When we don't have a density matrix
        if self.counts and not self.expt:
            return [w(theta) for w in ws]
        
        # When we do have a density matrix
        for i, w in enumerate(ws):
            vals += [np.trace(w @ self.rho).real]
        return vals

    def __str__(self):
        return (
            f'Rho: {self.rho}\n'
            f'Counts: {self.counts}\n'
            f'Expt: {self.expt}'
        )

class W5(W3):
    """
    W' witnesses, calculated with Paco's rotations

    Attributes:
    counts (optional) - np array of photon counts and uncertainties from experimental data
    rho (optional)    - the density matrix for the entangled photon state
    expt (optional)   - whether or not to calculate the experimental density matrix using counts
    angles            - a list of angles to be used in rotations
        + angles[0] = alpha
        + angles[1] = beta

        
    NOTE: this class inherits from W3, so all methods in that class can be used here, and all notes apply
    """
    def __init__(self, angles, rho=None, counts=None, expt=True):
        assert len(angles) == 2, "ERROR: 2 angles must be provided (alpha, beta)"

        super().__init__(rho=rho, counts=counts, expt=expt)
        self.angles = angles
        self.alpha = angles[0]
        self.beta = angles[1]

    def Wp1(self, theta=None):
        w1 = super().W1(theta)
        rotation = np.kron(R_z(self.alpha), IDENTITY)
        return op.rotate_m(w1, rotation)
    
    def Wp2(self, theta=None):
        w2 = super().W2(theta)
        rotation = np.kron(R_z(self.alpha), IDENTITY)
        return op.rotate_m(w2, rotation)
    
    def Wp3(self, theta=None):
        w3 = super().W3(theta)
        rotation = np.kron(R_z(self.alpha), R_z(self.beta))
        return op.rotate_m(w3, rotation)
    
    def Wp4(self, theta=None):
        w3 = super().W3(theta)
        rotation = np.kron(R_x(self.alpha), IDENTITY)
        return op.rotate_m(w3, rotation)
    
    def Wp5(self, theta=None):
        w4 = super().W4(theta)
        rotation = np.kron(R_x(self.alpha), IDENTITY)
        return op.rotate_m(w4, rotation)
    
    def Wp6(self, theta=None):
        w1 = super().W1(theta)
        rotation = np.kron(R_x(self.alpha), R_y(self.beta))
        return op.rotate_m(w1, rotation)
    
    def Wp7(self, theta=None):
        w5 = super().W5(theta)
        rotation = np.kron(R_y(self.alpha), IDENTITY)
        return op.rotate_m(w5, rotation)
    
    def Wp8(self, theta=None):
        w6 = super().W6(theta)
        rotation = np.kron(R_y(self.alpha), IDENTITY)
        return op.rotate_m(w6, rotation)

    def Wp9(self, theta=None):
        w1 = super().W1(theta)
        rotation = np.kron(R_y(self.alpha), R_y(self.beta))
        return op.rotate_m(w1, rotation)
    
    def get_witnesses(self, ops=True, theta=None):
        w5s = [self.Wp1, self.Wp2, self.Wp3, 
                self.Wp4, self.Wp5, self.Wp6,
                self.Wp7, self.Wp8, self.Wp9]
        # Return operators
        if ops:
            ws = super().get_witnesses()
            ws += w5s
            return ws
        
        ## Return expectation values
        vals = super().get_witnesses(ops, theta)
        w5s = [w(theta) for w in w5s]
        if self.counts and not self.expt:
            return vals + w5s

        for i, w in enumerate(w5s):
            vals += [np.trace(w @ self.rho).real]
        return vals

    
    def __str__(self):
        return (
            f'Rotation Angles: {self.angles}\n'
            f'Rho: {self.rho}\n'
            f'Counts: {self.counts}\n'
            f'Expt: {self.expt}'
        )



if __name__ == '__main__':
    print("States, Matrices, and Witnesses Loaded.")