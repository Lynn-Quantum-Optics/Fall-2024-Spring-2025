import numpy as np
import finding_states.states_and_witnesses as states
import finding_states.operations as op


############################
## PAST STATES OF INTEREST
############################

##  The following 2 states inspired the W3s ##
def PHI_P_PHI_M(eta, chi, rho=True):
     """
     The state cos(η)*PHI_P + e^(iχ)sin(η)*PHI_M

     Parameters:
     eta, chi - floats for η and χ respectively
     rho      - boolean, if true, return as a state, otherwise return as
                a density matrix

     NOTE: state is false by default
     NOTE: these parameters are the same for all other states defined in this
           file
     """
     state = np.cos(eta)*states.PHI_P + np.exp(1j*chi)*np.sin(eta)*states.PHI_M
     return make_state(state, rho)

def PSI_P_PSI_M(eta, chi, rho=True):
    """
    The state cos(η)*PSI_P + e^(iχ)sin(η)*PSI_M
    """
    state = np.cos(eta)*states.PSI_P + np.exp(1j*chi)*np.sin(eta)*states.PSI_M
    return make_state(state, rho)

## The following 6 states inspired the W5s ##
def PHI_P_PSI_M(eta, chi, rho=True):
    """
    The state cos(η)*PHI_P + e^(iχ)sin(η)*PSI_M
    """
    state = np.cos(eta)*states.PHI_P + np.exp(1j*chi)*np.sin(eta)*states.PSI_M
    return make_state(state, rho)

def PHI_M_PSI_P(eta, chi, rho=True):
    """
    The state cos(η)*PHI_M + e^(iχ)sin(η)*PSI_P
    """
    state = np.cos(eta)*states.PHI_M + np.exp(1j*chi)*np.sin(eta)*states.PSI_P
    return make_state(state, rho)

def PHI_P_iPSI_P(eta, chi, rho=True):
    """
    The state cos(η)*PHI_P + ie^(iχ)sin(η)*PSI_P
    """
    state = np.cos(eta)*states.PHI_PLUS + 1j*np.exp(1j*chi)*np.sin(eta)*states.PSI_P
    return make_state(state, rho)

def PHI_P_iPHI_M(eta, chi, rho=True):
    """
    The state cos(η)*PHI_P + ie^(iχ)sin(η)*PHI_M
    """
    state = np.cos(eta)*states.PHI_P + 1j*np.exp(1j*chi)*np.sin(eta)*states.PHI_M
    return make_state(state, rho)

def PSI_P_iPSI_M(eta, chi, rho=True):
    """
    The state cos(η)*PSI_P + ie^(iχ)sin(η)*PSI_M
    """
    state = np.cos(eta)*states.PSI_P + 1j*np.exp(1j*chi)*np.sin(eta)*states.PSI_M
    return make_state(state, rho)

def PHI_M_iPSI_M(eta, chi, rho=True):
    """
    The state cos(η)*PHI_M + ie^(iχ)sin(η)*PSI_M
    """
    state = np.cos(eta)*states.PHI_M + 1j*np.exp(1j*chi)*np.sin(eta)*states.PSI_M
    return make_state(state, rho)

## The following states are an attempt to find new states ##
## that give a positive W3 and negative W5 states ##
def HR_VL(chi, rho=True):
    """
    (1+e^(iχ))/2 * HR + (1-e^(iχ))/2 * VL
    """
    state = ((1 + np.exp(1j*chi))/2 * np.kron(states.H, states.R) + 
             (1 - np.exp(1j*chi))/2 * np.kron(states.V, states.L))
    return make_state(state, rho)

def HR_iVL(chi, rho=True):
    """
    (1+e^(iχ))/2 * HR + i*(1-e^(iχ))/2 * VL
    """
    state = ((1 + np.exp(1j*chi))/2 * np.kron(states.H, states.R) + 
              1j*(1 - np.exp(1j*chi))/2 * np.kron(states.V, states.L))
    return make_state(state, rho)

def HL_VR(chi, rho=True):
    """
    (1+e^(iχ))/2 * HL + (1-e^(iχ))/2 * VR
    """
    state = ((1 + np.exp(1j*chi))/2 * np.kron(states.H, states.L) + 
             (1 - np.exp(1j*chi))/2 * np.kron(states.V, states.R))
    return make_state(state, rho)
    
def HL_iVR(chi, rho=True):
    """
    (1+e^(iχ))/2 * HL + i*(1-e^(iχ))/2 * VR
    """
    state = ((1 + np.exp(1j*chi))/2 * np.kron(states.H, states.L) + 
             1j*(1 - np.exp(1j*chi))/2 * np.kron(states.V, states.R))
    return make_state(state, rho)
    
def HD_VA(chi, rho=True):
    """
    (1+e^(iχ))/2 * HD + (1-e^(iχ))/2 * VA
    """
    state = ((1 + np.exp(1j*chi))/2 * np.kron(states.H, states.D) + 
             (1 - np.exp(1j*chi))/2 * np.kron(states.V, states.A))
    return make_state(state, rho)

def HD_iVA(chi, rho=True):
    """
    (1+e^(iχ))/2 * HD + i*(1-e^(iχ))/2 * VA
    """
    state = ((1 + np.exp(1j*chi))/2 * np.kron(states.H, states.D) + 
             1j*(1 - np.exp(1j*chi))/2 * np.kron(states.V, states.A))
    make_state(state, rho)
    
def HA_VD(chi, rho=True):
    """
    (1+e^(iχ))/2 * HA + (1-e^(iχ))/2 * VD
    """
    state = ((1 + np.exp(1j*chi))/2 * np.kron(states.H, states.A) + 
             (1 - np.exp(1j*chi))/2 * np.kron(states.V, states.D))
    return make_state(state, rho)

def HA_iVD(chi, rho=True):
    """
    (1+e^(iχ))/2 * HA + i*(1-e^(iχ))/2 * VD
    """
    state = ((1 + np.exp(1j*chi))/2 * np.kron(states.H, states.A) +
              1j*(1 - np.exp(1j*chi))/2 * np.kron(states.V, states.D))
    return make_state(state, rho)

def HR_minVL(chi, rho=True):
    """
    (1+e^(iχ))/2 * HR + i*(1-e^(iχ))/2 * VD
    """
    state = ((1 + np.exp(1j*chi))/2 * np.kron(states.H, states.R) 
             + 1j*(1 - np.exp(1j*chi))/2 * np.kron(states.V, states.L))
    return make_state(state, rho)
    
def cosHD_sinVA(chi, rho=True):
    """
    cos(χ)*HD + sin(χ)*VA
    """
    state = (np.cos(chi) * np.kron(states.H, states.D) + 
             np.sin(chi)*np.kron(states.V, states.A))
    return make_state(state, rho)

def cosHR_minussinVL(chi, rho=True):
    """
    cos(χ/2)*HR - sin(χ/2)*VL
    """
    state = (np.cos(chi/2) * np.kron(states.H, states.R) - 
             np.sin(chi/2) * np.kron(states.V, states.L))
    return make_state(state, rho)

def cosHR_sinVL(chi, rho=True):
    """
    cos(χ/2)*HR + sin(χ/2)*VL
    """
    state = (np.cos(chi/2) * np.kron(states.H, states.R) + 
           np.sin(chi/2) * np.kron(states.V, states.L))
    return make_state(state, rho)

def cosHR_minusisinVL(chi, rho=True):
    """
    cos(χ/2)*HR - i*sin(χ/2)*VL
    """
    state = (np.cos(chi/2) * np.kron(states.H, states.R) - 
             1j * np.sin(chi/2) * np.kron(states.V, states.L))
    return make_state(state, rho)
        
def cosHL_sinVR(chi, rho=True):
    """
    cos(χ/2)*HL + sin(χ/2)*VR
    """
    state = (np.cos(chi/2) * np.kron(states.H, states.L) + 
             np.sin(chi/2) * np.kron(states.V, states.R))
    return make_state(state, rho)
    
def cosHD_minussinVA(chi, rho=True):
    """
    cos(χ/2)*HD - sin(χ/2)*VA
    """
    state = (np.cos(chi/2) * np.kron(states.H, states.D) - 
             np.sin(chi/2) * np.kron(states.V, states.A))
    return make_state(state, rho)
    
def cosHA_sinVD(chi, rho=True):
    """
    cos(χ/2)*HA + sin(χ/2)*VD
    """
    state = (np.cos(chi/2) * np.kron(states.H, states.A) + 
             np.sin(chi/2) * np.kron(states.V, states.D))
    return make_state(state, rho)
    
def cosHA_isinVD(chi, rho=True):
    """
    cos(χ/2)*HA + i*sin(χ/2)*VD
    """
    state = (np.cos(chi/2) * np.kron(states.H, states.A) + 
             1j * np.sin(chi/2) * np.kron(states.V, states.D))
    return make_state(state, rho)

def cosHA_minussinVD(chi, rho=True):
    """
    cos(χ/2)*HA - sin(χ/2)*VD
    """
    state = (np.cos(chi/2) * np.kron(states.H, states.A) - 
             np.sin(chi/2) * np.kron(states.V, states.D))
    return make_state(state, rho)
    
def cosHA_minusisinVD(chi, rho=True):
    """
    cos(χ/2)*HA - i*sin(χ/2)*VD
    """
    state = (np.cos(chi/2) * np.kron(states.H, states.A) - 
             1j * np.sin(chi/2) * np.kron(states.V, states.D))
    return make_state(state, rho)

def testof_hrivl_expt_match(chi, rho=True):
    """
    cos(χ/2)*HR + e^(i2π/3)sin(χ/2)*VL
    """
    state = (np.cos(chi/2) * np.kron(states.H, states.R) + 
             np.exp(1j* 2*np.pi/3)*np.sin(chi/2) * np.kron(states.V, states.L))
    return make_state(state, rho)

def testof_hdiva_expt_match(chi, rho=True):
    """
    cos(χ/2)*HD + e^(-iπ/3)sin(χ/2)*VA
    """
    state = (np.cos(chi/2) * np.kron(states.H, states.D) + 
             np.exp(-1j* np.pi/3)*np.sin(chi/2) * np.kron(states.V, states.A))
    return make_state(state, rho)
    
def HD(rho=True):
    state = np.kron(states.H, states.D)
    return make_state(state, rho)

def HA(rho=True):
    state = np.kron(states.H, states.A)
    return make_state(state, rho)    

def VD(rho=True):
    state = np.kron(states.V, states.D)
    return make_state(state, rho)
    
def VA(rho=True):
    state = np.kron(states.V, states.A)
    return make_state(state, rho)
    
def HR(rho=True):
    state = np.kron(states.H, states.R)
    return make_state(state, rho)
    
def VL(rho=True):
    state = np.kron(states.V, states.L)
    return make_state(state, rho)
    
### The following states are states we can potentially create  ###
### in our apparatus w/ +W, -W' without post-processing mixing ###
# NOTE:  eta is defined differently for these states
def phase_test_hdva(eta, chi, rho=True):
    """
    cos(χ/2)*HD + e^(-iη)sin(χ/2)*VA
    """
    state = (np.cos(chi/2) * np.kron(states.H, states.D) 
             + np.exp(-1j * eta) * np.sin(chi/2) * np.kron(states.V, states.A))
    return make_state(state, rho)

def phase_test_hhvv(eta, chi, rho=True):
    """
    cos(χ/2)*HH + e^(-iη)sin(χ/2)*VV
    """
    state = (np.cos(chi/2) * states.HH + 
             np.exp(-1j * eta) * np.sin(chi/2) * states.VV)
    return make_state(state, rho)

def phase_test_hvvh(eta, chi, rho=True):
    """
    cos(χ/2)*HV + e^(-iη)sin(χ/2)*VH
    """
    state = (np.cos(chi/2) * states.HV + 
             np.exp(-1j * eta) * np.sin(chi/2) * states.VH)
    return make_state(state, rho)

def phase_test_hrvl(eta, chi, rho=True):
    """
    cos(χ/2)*HR + e^(-iη)sin(χ/2)*VL
    """
    state = (np.cos(chi/2) * np.kron(states.H, states.R) + 
             np.exp(-1j * eta) * np.sin(chi/2) * np.kron(states.V, states.L))
    return make_state(state, rho)
    
def cosHA_minusphasesinVD(chi, rho=True):
    """
    cos(χ/2)*HA + e^(-i*1.27)sin(χ/2)*VD
    """
    state = (np.cos(chi/2) * np.kron(states.H, states.A) + 
             np.exp(-1j * 1.27) * np.sin(chi/2) * np.kron(states.V, states.D))
    return make_state(state, rho)
    
def cosHL_minussinVR(chi, rho=True):
    """
    cos(χ/2)*HL - sin(χ/2)*VR
    """
    state = (np.cos(chi/2) * np.kron(states.H, states.L) - 
             np.sin(chi/2) * np.kron(states.V, states.R))
    return make_state(state, rho)


#####################
## HELPER FUNCTIONS
#####################
def make_state(state, rho):
    """
    Given a state, returns it either as a ket or density matrix

    Paramters:
    rho - if true, return as density matrix, otherwise return as a ket
    """
    # Return as a density matrix
    if rho:
        return op.get_rho(state)
    
    # Return as a ket
    return state

