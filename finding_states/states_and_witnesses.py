import numpy as np
import finding_states.operations as op

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


##########################################
##        ENTANGLEMENT WITNESSES        ## 
##########################################

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
    rho (optional)    - the density matrix for the 2-photon state
    counts (optional) - np array of photon counts and uncertainties from experimental data

    NOTE: One of rho or counts must be given
    NOTE: If counts is given, experimental calculations will be used
    NOTE: If rho is given, theoretical calculations will be used
    """
    def __init__(self, rho = None, counts=None):
        self.counts = counts
            
        # counts not given, so we want to use the given theoretical rho
        if not counts:
            assert rho is not None, "ERROR: counts not given, so rho should be given"
            self.rho = rho

        else:
            # counts given, so we will construct an experimental density matrix, so
            # rho should not be given as it's for theoretical states
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

            # calculate experimental density matrix
            # NOTE: doesn't necessarily represent a full state
            self.rho = self.expt_rho()


    ##################################
    ## WITNESS DEFINITIONS/FUNCTIONS
    ##################################
    def W3_1(self, theta):
        """
        The first W3 witness

        Params:
        theta - the parameter for the rank-1 projector
        """
        a = np.cos(theta)
        b = np.sin(theta)

        # For experimental data, ensure we have the necessary counts
        if self.counts:
            W3.check_counts(self)
        
        # W1 as a matrix (for when we use full tomography & thus have a rho)
        phi1 = a*PHI_P + b*PHI_M
        return op.partial_transpose(phi1 * op.adjoint(phi1))
        
    def W3_2(self, theta):
        a = np.cos(theta)
        b = np.sin(theta)

        if self.counts:
            W3.check_counts(self)
        
        phi2 = a*PSI_P + b*PSI_M
        return op.partial_transpose(phi2 * op.adjoint(phi2))
        
    def W3_3(self, theta):
        a = np.cos(theta)
        b = np.sin(theta)

        if self.counts:
            W3.check_counts(self)
        
        phi3 = a*PHI_P + b*PSI_P
        return op.partial_transpose(phi3 * op.adjoint(phi3))

    def W3_4(self, theta):
        a = np.cos(theta)
        b = np.sin(theta)

        if self.counts:
            W3.check_counts(self)

        phi4 = a*PHI_M + b*PSI_M
        return op.partial_transpose(phi4 * op.adjoint(phi4))
        
    def W3_5(self, theta):
        a = np.cos(theta)
        b = np.sin(theta)

        if self.counts:
            W3.check_counts(self)

        phi5 = a*PHI_P + 1j*b*PSI_M
        return op.partial_transpose(phi5 * op.adjoint(phi5))
        
    def W3_6(self, theta):
        a = np.cos(theta)
        b = np.sin(theta)

        if self.counts:
            W3.check_counts(self)

        phi6 = a*PHI_M + 1j*b*PSI_P
        return op.partial_transpose(phi6 * op.adjoint(phi6))

    def get_witnesses(self, vals=False, theta=None):
        """
        Returns the expectation values of all 6 witnesses with a given theta
        or the operators themselves

        Params:
        vals (optional)  - if true, return the witness expectation values, otherwise return 
                           the operators as functions
        theta (optional) - the theta value to calculate expectation values for when vals is True
        
        NOTE: vals is False by default
        NOTE: theta is not given by default, and must be given when vals is True
        NOTE: operators will always be returned if vals is False even if theta is given
        """
        ws = [self.W3_1, self.W3_2, self.W3_3, self.W3_4, self.W3_5, self.W3_6]
    
        ## By default, return the operators
        if not vals:
            return ws

        ## Otherwise, return the expectation values with the given theta
        assert theta is not None, "ERROR: theta not given"
        ws = [w(theta) for w in ws]
        values = []
        
        for w in ws:
            values += [np.trace(w @ self.rho).real]
        return values


    ##################################
    ## COUNT HANDLING FUNCTIONS
    ##################################
    def expt_rho(self):
        """
        Calculates the experimental density matrix
        
        NOTE: this only represents an actual state if all counts were given
              otherwise, the resulting matrix only contains partial information
              of the state
        """
        
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

        stokes = [1,                                                                         # 0
            (self.dd - self.da + self.ad - self.aa)/(self.dd + self.da + self.ad + self.aa),
            (self.rr + self.lr - self.rl - self.ll)/(self.rr + self.lr + self.rl + self.ll),
            (self.hh - self.hv - self.vh - self.vv)/(self.hh + self.hv + self.vh + self.vv),
            (self.dd + self.da - self.ad - self.aa)/(self.dd + self.da + self.ad + self.da),
            (self.dd - self.da - self.ad + self.aa)/(self.dd + self.da + self.ad + self.aa), # 5
            (self.dr - self.dl - self.ar + self.al)/(self.dr + self.dl + self.ar + self.al),
            (self.dh - self.dv - self.ah + self.av)/(self.dh + self.dv + self.ah + self.av),
            (self.rr - self.lr + self.rl - self.ll)/(self.rr + self.lr + self.rl + self.ll),
            (self.rd - self.ra - self.ld + self.la)/(self.rd + self.ra + self.ld + self.la),
            (self.rr - self.rl - self.lr + self.ll)/(self.rr + self.rl + self.lr + self.ll), # 10
            (self.rh - self.rv - self.lh + self.lv)/(self.rh + self.rv + self.lh + self.lv),
            (self.hh + self.hv - self.vh - self.vv)/(self.hh + self.hv + self.vh + self.vv),
            (self.hd - self.ha - self.vd + self.va)/(self.hd + self.ha + self.vd + self.va),
            (self.hr - self.hl - self.vr + self.vl)/(self.hr + self.hl + self.vr + self.vl),
            (self.hh - self.hv - self.vh + self.vv)/(self.hh + self.hv + self.vh + self.vv)  # 15
        ]
        
        return stokes
    
    def check_zz(self, quiet=False):
        """
        Checks the necessary counts to determine if the zz measurement was taken

        Params:
        quiet - if true, don't tell the user if the measurement is given
        """
        assert self.hh != 0, "Missing HH measurement"
        assert self.hv != 0, "Missing HV measurement"
        assert self.vh != 0, "Missing VH measurement"
        assert self.vv != 0, "Missing VV measurement"

        if not quiet:
            print("ZZ measurement was taken!")

    def check_xx(self, quiet=False):
        """Determines if the xx measurement was taken"""
        assert self.dd != 0, "Missing DD measurement"
        assert self.da != 0, "Missing DA measurement"
        assert self.ad != 0, "Missing AD measurement"
        assert self.aa != 0, "Missing AA measurement"

        if not quiet:
            print("XX measurement was taken!")

    def check_yy(self, quiet=False):
        """Determines if the yy measurement was taken"""
        assert self.rr != 0, "Missing RR measurement"
        assert self.rl != 0, "Missing RL measurement"
        assert self.lr != 0, "Missing LR measurement"
        assert self.ll != 0, "Missing LL measurement"

        if not quiet:
            print("YY measurement was taken!")

    def check_xy(self, quiet=False):
        """Determines if the xy measurement was taken"""
        assert self.dr != 0, "Missing DR measurement"
        assert self.dl != 0, "Missing DL measurement"
        assert self.ar != 0, "Missing AR measurement"
        assert self.al != 0, "Missing AL measurement"

        if not quiet:
            print("XY measurement was taken!")

    def check_yx(self, quiet=False):
        """Determines if the yx measurement was taken"""
        assert self.rd != 0, "Missing RD measurement"
        assert self.ra != 0, "Missing RA measurement"
        assert self.ld != 0, "Missing LD measurement"
        assert self.la != 0, "Missing LA measurement"

        if not quiet:
            print("YX measurement was taken!")

    def check_zy(self, quiet=False):
        """Determines if the zy measurement was taken"""
        assert self.hr != 0, "Missing HR measurement"
        assert self.hl != 0, "Missing HL measurement"
        assert self.vr != 0, "Missing VR measurement"
        assert self.vl != 0, "Missing VL measurement"

        if not quiet:
            print("ZY measurement was taken!")

    def check_yz(self, quiet=False):
        """Determines if the yz measurement was taken"""
        assert self.rh != 0, "Missing RH measurement"
        assert self.rv != 0, "Missing RV measurement"
        assert self.lh != 0, "Missing LH measurement"
        assert self.lv != 0, "Missing LV measurement"   

        if not quiet:
            print("YZ measurement was taken!")

    def check_xz(self, quiet=False):
        """Determines if the xz measurement was taken"""
        assert self.dh != 0, "Missing DH measurement"
        assert self.dv != 0, "Missing DV measurement"
        assert self.ah != 0, "Missing AH measurement"
        assert self.av != 0, "Missing AV measurement"

        if not quiet:
            print("XZ measurement was taken!")

    def check_zx(self, quiet=False):
        """Determines if the zx measurement was taken"""
        assert self.hd != 0, "Missing HD measurement"
        assert self.ha != 0, "Missing HA measurement"
        assert self.vd != 0, "Missing VD measurement"
        assert self.va != 0, "Missing VA measurement"

        if not quiet:
            print("ZX measurement was taken!")

    def check_all_counts(self):
        """
        Check that all counts were provided, this is the same
        as taking a full tomography
        """
        self.check_zz(quiet=True)
        self.check_yy(quiet=True)
        self.check_xx(quiet=True)
        self.check_xy(quiet=True)
        self.check_yx(quiet=True)
        self.check_zy(quiet=True)
        self.check_yz(quiet=True)
        self.check_xz(quiet=True)
        self.check_zx(quiet=True)

        print("Full tomography taken!")

    def check_counts(self):
        """
        Checks to see that the necessary counts have been 
        given when calculating a witness with experimental data
        """
        self.check_zz(quiet=True)
        self.check_xx(quiet=True)
        self.check_yy(quiet=True)

    def __str__(self):
        return (
            f'Rho: {self.rho}\n'
            f'Counts: {self.counts}\n'
        )

class W5(W3):
    """
    W5 witnesses, calculated with Paco's rotations (section 3.4.2 of Navarro thesis).
    These use local measurements on 5 Pauli bases.

    Attributes:
    rho (optional)    - the density matrix for the 2-photon state
    counts (optional) - np array of photon counts and uncertainties from experimental data
        
    NOTE: this class inherits from W3, so all methods in that class can be used here, and all notes apply
    """
    def __init__(self, rho=None, counts=None):
        super().__init__(rho=rho, counts=counts)

    ## Triplet 1: Rotate about z ##
    def W5_1(self, theta, alpha):
        """
        First W5 witness, rotates particle 1 about the z-axis
        from the W3_1 witness

        Params:
        theta - free parameter used in W3_1
        alpha - rotation angle
        """
        w1 = self.W3_1(theta)
        
        if self.counts:
            W5.check_counts(self, triplet=1)

        rotation = np.kron(R_z(alpha), IDENTITY)
        return op.rotate_m(w1, rotation)
    
    def W5_2(self, theta, alpha):
        w2 = self.W3_2(theta)

        if self.counts:
            W5.check_counts(self, triplet=1)

        rotation = np.kron(R_z(alpha), IDENTITY)
        return op.rotate_m(w2, rotation)
    
    def W5_3(self, theta, alpha, beta):
        w3 = self.W3_3(theta)

        if self.counts:
            W5.check_counts(self, triplet=1)

        rotation = np.kron(R_z(alpha), R_z(beta))
        return op.rotate_m(w3, rotation)
    

    ## Triplet 2: Rotate about x ##
    def W5_4(self, theta, alpha):
        w3 = self.W3_3(theta)

        if self.counts:
            W5.check_counts(self, triplet=2)

        rotation = np.kron(R_x(alpha), IDENTITY)
        return op.rotate_m(w3, rotation)
    
    def W5_5(self, theta, alpha):
        w4 = self.W3_4(theta)

        if self.counts:
            W5.check_counts(self, triplet=2)

        rotation = np.kron(R_x(alpha), IDENTITY)
        return op.rotate_m(w4, rotation)
    
    def W5_6(self, theta, alpha, beta):
        w1 = self.W3_1(theta)

        if self.counts:
            W5.check_counts(self, triplet=2)

        rotation = np.kron(R_x(alpha), R_y(beta))
        return op.rotate_m(w1, rotation)
    
    
    ## Triplet 3: Rotate about y ##
    def W5_7(self, theta, alpha):
        w5 = self.W3_5(theta)

        if self.counts:
            W5.check_counts(self, triplet=3)

        rotation = np.kron(R_y(alpha), IDENTITY)
        return op.rotate_m(w5, rotation)
    
    def W5_8(self, theta, alpha):
        w6 = self.W3_6(theta)

        if self.counts:
            W5.check_counts(self, triplet=3)

        rotation = np.kron(R_y(alpha), IDENTITY)
        return op.rotate_m(w6, rotation)

    def W5_9(self, theta, alpha, beta):
        w1 = self.W3_1(theta)

        if self.counts:
            W5.check_counts(self, triplet=3)

        rotation = np.kron(R_y(alpha), R_y(beta))
        return op.rotate_m(w1, rotation)
    
    def get_witnesses(self, vals=False, theta=None, alpha=None, beta=None):
        w5s = [self.W5_1, self.W5_2, self.W5_3, 
                self.W5_4, self.W5_5, self.W5_6,
                self.W5_7, self.W5_8, self.W5_9]
        
        # Return operators
        if not vals:
            ws = super().get_witnesses()
            ws += w5s
            return ws
        
        ## Return expectation values
        assert theta is not None, "ERROR: theta not given"
        assert alpha is not None, "ERROR: alpha not given"
        assert beta is not None, "ERROR: beta not given"

        values = super().get_witnesses(vals, theta)

        # Get the W5s with the given theta, alpha, and beta
        for i, W in enumerate(w5s):
            if i == 2 or i == 5 or i == 8:
                w5s[i] = W(theta, alpha, beta)
            else:
                w5s[i] = W(theta, alpha)
        
        for w in w5s:
            values += [np.trace(w @ self.rho).real]
        return values
    
    def check_counts(self, triplet):
        """
        Checks to see that the necessary counts have been 
        given when calculating a witness with experimental data

        Params:
        triplet - which triplet the witness belongs to

        NOTE: The xx, yy, and zz measurement have already been checked
              by a previous call to W3.check_counts, which happens when
              getting the W3 witness to perform the rotation on to get the W5
        """
        if triplet == 1:
            self.check_xy(quiet=True)
            self.check_yx(quiet=True)

        elif triplet == 2:
            self.check_zy(quiet=True)
            self.check_yz(quiet=True)

        elif triplet == 3:
            self.check_xz(quiet=True)
            self.check_zx(quiet=True)

        else:
            assert False, "Invalid triplet specified"
    

class W8(W5):
    """
    W8 witnesses, calculated with Paco's rotations (section 4.1 of Navarro thesis).
    These use local measurements on 8 Pauli bases.

    Attributes:
    rho (optional)    - the density matrix for the 2-photon state
    counts (optional) - np array of photon counts and uncertainties from experimental data
        
    NOTE: this class inherits from W5, so all methods in that class can be used here, and all notes apply
    """
    def __init__(self, rho=None, counts=None):
        super().__init__(rho=rho, counts=counts)


    ######################################################
    ## SET 1 (W8_{1-6}): EXCLUDES XY MEASUREMENT
    ######################################################
    
    ## Triplet 1: Rotate particle 1 on y then x ##
    def W8_1(self, theta, alpha, beta, for_w7=False):
        """
        First W8 witness
        """
        w5_7 = self.W5_7(theta, alpha)

        if self.counts and not for_w7:
            W8.check_counts(triplet=1)

        rotation = np.kron(R_x(beta), IDENTITY)
        return op.rotate_m(w5_7, rotation)
    
    def W8_2(self, theta, alpha, beta, for_w7=False):
        w5_8 = self.W5_8(theta, alpha)

        if self.counts and not for_w7:
            W8.check_counts(triplet=1)
        
        rotation = np.kron(R_x(beta), IDENTITY)
        return op.rotate_m(w5_8, rotation)
    
    def W8_3(self, theta, alpha, beta, gamma, for_w7=False):
        w5_9 = self.W5_9(theta, alpha, beta)

        if self.counts and not for_w7:
            W8.check_counts(triplet=1)
        
        rotation = np.kron(R_x(gamma), IDENTITY)
        return op.rotate_m(w5_9, rotation)
    

    ## Triplet 2: Rotate about x, then rotate particle 2 about y ##
    def W8_4(self, theta, alpha, beta, for_w7=False):
        w5_4 = self.W5_4(theta, alpha)

        if self.counts and not for_w7:
            W8.check_counts(triplet=2)
        
        rotation = np.kron(IDENTITY, R_y(beta))
        return op.rotate_m(w5_4, rotation)
    
    def W8_5(self, theta, alpha, beta, for_w7=False):
        w5_5 = self.W5_5(theta, alpha)

        if self.counts and not for_w7:
            W8.check_counts(triplet=2)
        
        rotation = np.kron(IDENTITY, R_y(beta))
        return op.rotate_m(w5_5, rotation)
    
    def W8_6(self, theta, alpha, beta, gamma, for_w7=False):
        w5_6 = self.W5_6(theta, alpha, beta)

        if self.counts and not for_w7:
            W8.check_counts(triplet=2)
        
        rotation = np.kron(IDENTITY, R_y(gamma))
        return op.rotate_m(w5_6, rotation)
    

    #############################################
    ## SET 2: EXCLUDES YX
    #############################################

    ## Triplet 3: Rotate particle 1 on x then y ##
    def W8_7(self, theta, alpha, beta, for_w7=False):
        w5_4 = self.W5_4(theta, alpha)

        if self.counts and not for_w7:
            W8.check_counts(triplet=3)

        rotation = np.kron(R_y(beta), IDENTITY)
        return op.rotate_m(w5_4, rotation)
    
    def W8_8(self, theta, alpha, beta, for_w7=False):
        w5_5 = self.W5_5(theta, alpha)

        if self.counts and not for_w7:
            W8.check_counts(triplet=3)

        rotation = np.kron(R_y(beta), IDENTITY)
        return op.rotate_m(w5_5, rotation)
    
    def W8_9(self, theta, alpha, beta, gamma, for_w7=False):
        w5_6 = self.W5_6(theta, alpha, beta)

        if self.counts and not for_w7:
            W8.check_counts(triplet=3)

        rotation = np.kron(R_y(gamma), IDENTITY)
        return op.rotate_m(w5_6, rotation)
    

    ## Triplet 4: Rotate about y, then rotate particle 2 about x ##
    def W8_10(self, theta, alpha, beta, for_w7=False):
        w5_7 = self.W5_7(theta, alpha)

        if self.counts and not for_w7:
            W8.check_counts(triplet=4)

        rotation = np.kron(IDENTITY, R_x(beta))
        return op.rotate_m(w5_7, rotation)
    
    def W8_11(self, theta, alpha, beta, for_w7=False):
        w5_8 = self.W5_8(theta, alpha)

        if self.counts and not for_w7:
            W8.check_counts(triplet=4)

        rotation = np.kron(IDENTITY, R_x(beta))
        return op.rotate_m(w5_8, rotation)
    
    def W8_12(self, theta, alpha, beta, gamma, for_w7=False):
        w5_9 = self.W5_9(theta, alpha, beta)

        if self.counts and not for_w7:
            W8.check_counts(triplet=4)

        rotation = np.kron(IDENTITY, R_x(gamma))
        return op.rotate_m(w5_9, rotation)
    
    #############################################
    ## SET 3: EXCLUDES XZ
    #############################################

    ## Triplet 5: Rotate particle 1 by z then x ##
    def W8_13(self, theta, alpha, beta, for_w7=False):
        w5_1 = self.W5_1(theta, alpha)

        if self.counts and not for_w7:
            W8.check_counts(triplet=5)

        rotation = np.kron(R_x(beta), IDENTITY)
        return op.rotate_m(w5_1, rotation)
    
    def W8_14(self, theta, alpha, beta, for_w7=False):
        w5_2 = self.W5_2(theta, alpha)

        if self.counts and not for_w7:
            W8.check_counts(triplet=5)

        rotation = np.kron(R_x(beta), IDENTITY)
        return op.rotate_m(w5_2, rotation)
    
    def W8_15(self, theta, alpha, beta, gamma, for_w7=False):
        w5_3 = self.W5_3(theta, alpha, beta)

        if self.counts and not for_w7:
            W8.check_counts(triplet=5)

        rotation = np.kron(R_x(gamma), IDENTITY)
        return op.rotate_m(w5_3, rotation)
    

    ## Triplet 6: Rotate about x, then rotate particle 2 by z ##
    def W8_16(self, theta, alpha, beta, for_w7=False):
        w5_4 = self.W5_4(theta, alpha)

        if self.counts and not for_w7:
            W8.check_counts(triplet=6)

        rotation = np.kron(IDENTITY, R_z(beta))
        return op.rotate_m(w5_4, rotation)
    
    def W8_17(self, theta, alpha, beta, for_w7=False):
        w5_5 = self.W5_5(theta, alpha)

        if self.counts and not for_w7:
            W8.check_counts(triplet=6)

        rotation = np.kron(IDENTITY, R_z(beta))
        return op.rotate_m(w5_5, rotation)
    
    def W8_18(self, theta, alpha, beta, gamma, for_w7=False):
        w5_6 = self.W5_6(theta, alpha, beta)

        if self.counts and not for_w7:
            W8.check_counts(triplet=6)

        rotation = np.kron(IDENTITY, R_z(gamma))
        return op.rotate_m(w5_6, rotation)
    

    #############################################
    ## SET 4: EXCLUDES ZX
    #############################################

    ## Triplet 7: Rotate particle 1 by x then z ##
    def W8_19(self, theta, alpha, beta, for_w7=False):
        w5_4 = self.W5_4(theta, alpha)

        if self.counts and not for_w7:
            W8.check_counts(triplet=7)

        rotation = np.kron(R_z(beta), IDENTITY)
        return op.rotate_m(w5_4, rotation)
    
    def W8_20(self, theta, alpha, beta, for_w7=False):
        w5_5 = self.W5_5(theta, alpha)

        if self.counts and not for_w7:
            W8.check_counts(triplet=7)

        rotation = np.kron(R_z(beta), IDENTITY)
        return op.rotate_m(w5_5, rotation)
    
    def W8_21(self, theta, alpha, beta, gamma, for_w7=False):
        w5_6 = self.W5_6(theta, alpha, beta)

        if self.counts and not for_w7:
            W8.check_counts(triplet=7)

        rotation = np.kron(R_z(gamma), IDENTITY)
        return op.rotate_m(w5_6, rotation)
    

    ## Triplet 8: Rotate about z, then rotate particle 2 by x ##
    def W8_22(self, theta, alpha, beta, for_w7=False):
        w5_1 = self.W5_1(theta, alpha)

        if self.counts and not for_w7:
            W8.check_counts(triplet=8)

        rotation = np.kron(IDENTITY, R_x(beta))
        return op.rotate_m(w5_1, rotation)
    
    def W8_23(self, theta, alpha, beta, for_w7=False):
        w5_2 = self.W5_2(theta, alpha)

        if self.counts and not for_w7:
            W8.check_counts(triplet=8)

        rotation = np.kron(IDENTITY, R_x(beta))
        return op.rotate_m(w5_2, rotation)
    
    def W8_24(self, theta, alpha, beta, gamma, for_w7=False):
        w5_3 = self.W5_3(theta, alpha, beta)

        if self.counts and not for_w7:
            W8.check_counts(triplet=8)

        rotation = np.kron(IDENTITY, R_x(gamma))
        return op.rotate_m(w5_3, rotation)
    
    
    #############################################
    ## SET 5: EXCLUDES YZ
    #############################################

    ## Triplet 9: Rotate particle 1 by z then y ##
    def W8_25(self, theta, alpha, beta, for_w7=False):
        w5_1 = self.W5_1(theta, alpha)

        if self.counts and not for_w7:
            W8.check_counts(triplet=9)

        rotation = np.kron(R_y(beta), IDENTITY)
        return op.rotate_m(w5_1, rotation)
    
    def W8_26(self, theta, alpha, beta, for_w7=False):
        w5_2 = self.W5_2(theta, alpha)

        if self.counts and not for_w7:
            W8.check_counts(triplet=9)

        rotation = np.kron(R_y(beta), IDENTITY)
        return op.rotate_m(w5_2, rotation)
    
    def W8_27(self, theta, alpha, beta, gamma, for_w7=False):
        w5_3 = self.W5_3(theta, alpha, beta)

        if self.counts and not for_w7:
            W8.check_counts(triplet=9)

        rotation = np.kron(R_y(gamma), IDENTITY)
        return op.rotate_m(w5_3, rotation)
    

    ## Triplet 10: Rotate about y, then rotate particle 2 by z ##
    def W8_28(self, theta, alpha, beta, for_w7=False):
        w5_7 = self.W5_7(theta, alpha)

        if self.counts and not for_w7:
            W8.check_counts(triplet=10)

        rotation = np.kron(IDENTITY, R_z(beta))
        return op.rotate_m(w5_7, rotation)
    
    def W8_29(self, theta, alpha, beta, for_w7=False):
        w5_8 = self.W5_8(theta, alpha)

        if self.counts and not for_w7:
            W8.check_counts(triplet=10)

        rotation = np.kron(IDENTITY, R_z(beta))
        return op.rotate_m(w5_8, rotation)
    
    def W8_30(self, theta, alpha, beta, gamma, for_w7=False):
        w5_9 = self.W5_9(theta, alpha, beta)

        if self.counts and not for_w7:
            W8.check_counts(triplet=10)

        rotation = np.kron(IDENTITY, R_z(gamma))
        return op.rotate_m(w5_9, rotation)
    

    #############################################
    ## SET 6: EXCLUDES ZY
    #############################################

    ## Triplet 11: Rotate particle 1 by y then z ##
    def W8_31(self, theta, alpha, beta, for_w7=False):
        w5_7 = self.W5_7(theta, alpha)

        if self.counts and not for_w7:
            W8.check_counts(triplet=11)

        rotation = np.kron(R_z(beta), IDENTITY)
        return op.rotate_m(w5_7, rotation)
    
    def W8_32(self, theta, alpha, beta, for_w7=False):
        w5_8 = self.W5_8(theta, alpha)

        if self.counts and not for_w7:
            W8.check_counts(triplet=11)

        rotation = np.kron(R_z(beta), IDENTITY)
        return op.rotate_m(w5_8, rotation)
    
    def W8_33(self, theta, alpha, beta, gamma, for_w7=False):
        w5_9 = self.W5_9(theta, alpha, beta)

        if self.counts and not for_w7:
            W8.check_counts(triplet=11)

        rotation = np.kron(R_z(gamma), IDENTITY)
        return op.rotate_m(w5_9, rotation)
    

    ## Triplet 12: Rotate about z, then rotate particle 2 by y ##
    def W8_34(self, theta, alpha, beta, for_w7=False):
        w5_1 = self.W5_1(theta, alpha)

        if self.counts and not for_w7:
            W8.check_counts(triplet=12)

        rotation = np.kron(IDENTITY, R_y(beta))
        return op.rotate_m(w5_1, rotation)
    
    def W8_35(self, theta, alpha, beta, for_w7=False):
        w5_2 = self.W5_2(theta, alpha)

        if self.counts and not for_w7:
            W8.check_counts(triplet=12)

        rotation = np.kron(IDENTITY, R_y(beta))
        return op.rotate_m(w5_2, rotation)
    
    def W8_36(self, theta, alpha, beta, gamma, for_w7=False):
        w5_3 = self.W5_3(theta, alpha, beta)

        if self.counts and not for_w7:
            W8.check_counts(triplet=12)

        rotation = np.kron(IDENTITY, R_y(gamma))
        return op.rotate_m(w5_3, rotation)


    def get_witnesses(self, vals=False, theta=None, alpha=None, beta=None, gamma=None):
        w8s = [self.W8_1, self.W8_2, self.W8_3, self.W8_4, self.W8_5, self.W8_6,
               self.W8_7, self.W8_8, self.W8_9, self.W8_10, self.W8_11, self.W8_12,
               self.W8_13, self.W8_14, self.W8_15, self.W8_16, self.W8_17, self.W8_18,
               self.W8_19, self.W8_20, self.W8_21, self.W8_22, self.W8_23, self.W8_24,
               self.W8_25, self.W8_26, self.W8_27, self.W8_28, self.W8_29, self.W8_30,
               self.W8_31, self.W8_32, self.W8_33, self.W8_34, self.W8_35, self.W8_36]
        
        # Return operators
        if not vals:
            ws = super().get_witnesses()
            ws += w8s
            return ws
        
        ## Return expectation values
        assert theta is not None, "ERROR: theta not given"
        assert alpha is not None, "ERROR: alpha not given"
        assert beta is not None, "ERROR: beta not given"
        assert gamma is not None, "ERROR: gamma not given"

        values = super().get_witnesses(vals, theta, alpha, beta)

        # Get the W8 values with the given parameters
        for i, W in enumerate(w8s):
            # every 3rd witness needs gamma (NOTE: i is zero-indexed)
            if i % 3 == 2:
                w8s[i] = W(theta, alpha, beta, gamma)
            else:
                w8s[i] = W(theta, alpha, beta)
        
        for w in w8s:
            values += [np.trace(w @ self.rho).real]
        return values
    
    def check_counts(self, triplet):
        """
        Checks to see that the necessary counts have been 
        given when calculating a witness with experimental data

        Params:
        triplet - which triplet the witness belongs to
        """
        ## Triplets 1-2 exclude the xy measurement ##
        if triplet == 1:
            # Rotation from 3rd W5 triplet
            #
            # already considered: xz, zx, xx, yy, zz
            # need to check: zy, yz, yx
            self.check_zy(quiet=True)
            self.check_yz(quiet=True)
            self.check_yx(quiet=True)
            
        elif triplet == 2:
            # Rotation from 2nd W5 triplet
            #
            # already considered: zy, yz, xx, yy, zz
            # need to check: xz, zx, yx
            self.check_xz(quiet=True)
            self.check_zx(quiet=True)
            self.check_yx(quiet=True)


        ## Triplets 3-4 exclude yx ##
        elif triplet == 3:
            # Rotation from 2nd W5 triplet
            # need to check: xz, zx, xy
            self.check_xz(quiet=True)
            self.check_zx(quiet=True)
            self.check_xy(quiet=True)
        
        elif triplet == 4:
            # Rotation from 3rd W5 triplet
            # need to check: zy, yz, xy
            self.check_zy(quiet=True)
            self.check_yz(quiet=True)
            self.check_xy(quiet=True)
        

        ## Triplets 5-6 exclude xz ##
        elif triplet == 5:
            # Rotation from 1st W5 triplet
            #
            # already considered: xy, yx, xx, yy, zz
            # need to check: zy, yz, zx
            self.check_zy(quiet=True)
            self.check_yz(quiet=True)
            self.check_zx(quiet=True)
        
        elif triplet == 6:
            # Rotation from 2nd W5 triplet
            # need to check: xy, yx, zx
            self.check_xy(quiet=True)
            self.check_yx(quiet=True)
            self.check_zx(quiet=True)
        

        ## Triplets 7-8 exclude zx ##
        elif triplet == 7:
            # Rotation from 2nd W5 triplet
            # need to check: xy, yx, xz
            self.check_xy(quiet=True)
            self.check_yx(quiet=True)
            self.check_xz(quiet=True)

        elif triplet == 8:
            # Rotation from 1st W5 triplet
            # need to check: zy, yz, xz
            self.check_zy(quiet=True)
            self.check_yz(quiet=True)
            self.check_xz(quiet=True)
        
        ## Triplets 9-10 exclude yz ##
        elif triplet == 9:
            # Rotation from 1st W5 triplet
            # need to check: zx, xz, zy
            self.check_zx(quiet=True)
            self.check_xz(quiet=True)
            self.check_zy(quiet=True)
        
        elif triplet == 10:
            # Rotation from 3rd W5 triplet
            # need to check: xy, yx, zy
            self.check_xy(quiet=True)
            self.check_yx(quiet=True)
            self.check_zy(quiet=True)
        
        ## Triplets 11-12 exclude zy ##
        elif triplet == 11:
            # Rotation from 3rd W5 triplet
            # need to check: xy, yx, yz
            self.check_xy(quiet=True)
            self.check_yx(quiet=True)
            self.check_yz(quiet=True)
        
        elif triplet == 12:
            # Rotation from 1st W5 triplet
            # need to check: zx, xz, yz
            self.check_zx(quiet=True)
            self.check_xz(quiet=True)
            self.check_yz(quiet=True)

        else:
            assert False, "Invalid triplet specified"


class W7(W8):
    """
    W7 witnesses, calculated with Paco's rotations (section 4.2 of Navarro thesis).
    These use local measurements on 7 Pauli bases.

    Attributes:
    rho (optional)    - the density matrix for the photon state
    counts (optional) - np array of photon counts and uncertainties from experimental data
        
    NOTE: this class inherits from W8, so all methods in that class can be used here, and all notes apply
    """
    def __init__(self, rho=None, counts=None):
        super().__init__(rho=rho, counts=counts)

    
    #############################################
    ## SET 1 (W7_{1-12}): EXCLUDES XY & YX
    #############################################

    # Triplet 1: deletion rotation about y on particle 2
    def W7_1(self, theta, alpha, beta):
        # deletion rotation angle (defined as gamma in Paco's thesis)
        delta = np.arctan(-1/np.tan(alpha))



        return
    

# Wrapper class that includes all witnesses from Pacos' Rotations
NavarroWitness = W7

if __name__ == '__main__':
    print("States, Matrices, and Witnesses Loaded.")