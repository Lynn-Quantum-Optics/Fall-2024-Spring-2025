import numpy as np
from scipy.optimize import minimize, approx_fprime
from uncertainties import unumpy as unp
from states_and_gates import *
import operations as op

def rotate(W):
    M = W @ W

    # NOTE: np.kron does tensor product

    return M

class InitGuess:
    """
    An object class that handles initial guesses

    Params:
        x0: an array that stores the guess, which is a vector
        w_val: the w0 associated with x0
        w_params: the w0 parameters associated with x0
    """
    def __init__(self, bounds, params_arr):
        self.x0 = params_arr
        self.w_val = min_W_val(self.x0, bounds)
        self.w_params = min_W_params(self.x0, bounds)
    # "magic methods" help us compare instances of a class
    def __eq__(self, other):
        return self.w_val == other.w_val
    def __ne__(self, other):
        return self.w_val != other.w_val
    def __lt__(self, other):
        return self.w_val < other.w_val
    def __le__(self, other):
        return self.w_val <= other.w_val
    def __gt__(self, other):
        return self.w_val > other.w_val
    def __ge__(self, other):
        return self.w_val >= other.w_val

def compute_witnesses(rho, counts = None, expt = False, verbose = True, do_counts = False, 
                      expt_purity = None, model=None, optimize = True, gd=True, ads_test=False, return_all=False, 
                      return_params=False):
    ''' Computes the minimum of the 6 Ws and the minimum of the 3 triples of the 9 W's. 
        Params:
            rho: the density matrix
            counts: raw np array of photon counts and uncertainties
            -------expt: bool, whether to compute the Ws assuming input is experimental data
            verbose: Whether to return which W/W' are minimal.
            do_stokes: bool, whether to compute stokes parameters
            -------do_counts: use the raw definition in terms of counts 
            expt_purity: the experimental purity of the state, which defines the noise level: 1 - purity.
            ?model: which model to correct for noise; see det_noise in process_expt.py for more info
            model_path: path to noise model csvs.

            # rewrite
            optimize: bool, whether to optimize the Ws or to just check bounds
            gd: bool, whether to use gradient descent or brute random search

            ?ads_test: bool, whether to return w2 expec and sin (theta) for the amplitude damped states
            return_all: bool, whether to return all the Ws or just the min of the 6 and the min of the 3 triples
            return_params: bool, whether to return the params that give the min of the 6 and the min of the 3 triples
    '''
    # check if experimental data
    if expt and counts is not None:
        do_counts = True
    # if wanting to account for experimental purity, add noise to the density matrix for adjusted theoretical purity calculation

    # With experimental data
    if do_counts:
        counts = np.reshape(counts, (36,1))
        def get_W1(params, counts):
            a, b = np.cos(params), np.sin(params)
            # each element in counts is of form: count +/ uncertainty
            HH, HV, HD, HA, HR, HL, VH, VV, VD, VA, VR, VL, DH, DV, DD, DA, DR, DL, AH, AV, AD, AA, AR, AL, RH, RV, RD, RA, RR, RL, LH, LV, LD, LA, LR, LL  = counts
            return np.real(0.25*(1 + ((HH - HV - VH + VV) / (HH + HV + VH + VV)) + (a**2 - b**2)*((DD - DA - AD + AA) / (DD + DA + AD + AA)) + (a**2 - b**2)*((RR - RL - LR + LL) / (RR + RL + LR + LL)) + 2*a*b*(((HH + HV - VH - VV) / (HH + HV + VH + VV)) + ((HH - HV + VH - VV) / (HH + HV + VH + VV)))))
        def get_W2(params, counts):
            a, b = np.cos(params), np.sin(params)
            HH, HV, HD, HA, HR, HL, VH, VV, VD, VA, VR, VL, DH, DV, DD, DA, DR, DL, AH, AV, AD, AA, AR, AL, RH, RV, RD, RA, RR, RL, LH, LV, LD, LA, LR, LL  = counts
            return np.real(0.25*(1 - ((HH - HV - VH + VV) / (HH + HV + VH + VV)) + (a**2 - b**2)*((DD - DA - AD + AA) / (DD + DA + AD + AA)) - (a**2 - b**2)*((RR - RL - LR + LL) / (RR + RL + LR + LL)) + 2*a*b*(((HH + HV - VH - VV) / (HH + HV + VH + VV)) - ((HH - HV + VH - VV) / (HH + HV + VH + VV)))))
        def get_W3(params, counts):
            a, b = np.cos(params), np.sin(params)
            HH, HV, HD, HA, HR, HL, VH, VV, VD, VA, VR, VL, DH, DV, DD, DA, DR, DL, AH, AV, AD, AA, AR, AL, RH, RV, RD, RA, RR, RL, LH, LV, LD, LA, LR, LL  = counts
            return np.real(0.25*(1 + ((DD - DA - AD + AA) / (DD + DA + AD + AA)) + (a**2 - b**2)*((HH - HV - VH + VV) / (HH + HV + VH + VV)) + (a**2 - b**2)*((RR - RL - LR + LL) / (RR + RL + LR + LL)) + 2*a*b*(((DD + DA - AD - AA) / (DD + DA + AD + AA)) + ((DD - DA + AD - AA) / (DD + DA + AD + AA)))))
        def get_W4(params, counts):
            a, b = np.cos(params), np.sin(params)
            HH, HV, HD, HA, HR, HL, VH, VV, VD, VA, VR, VL, DH, DV, DD, DA, DR, DL, AH, AV, AD, AA, AR, AL, RH, RV, RD, RA, RR, RL, LH, LV, LD, LA, LR, LL  = counts
            return np.real(0.25*(1 - ((DD - DA - AD + AA) / (DD + DA + AD + AA)) + (a**2 - b**2)*((HH - HV - VH + VV) / (HH + HV + VH + VV)) - (a**2 - b**2)*((RR - RL - LR + LL) / (RR + RL + LR + LL)) - 2*a*b*(((DD + DA - AD - AA) / (DD + DA + AD + AA)) - ((DD - DA + AD - AA) / (DD + DA + AD + AA)))))
        def get_W5(params, counts):
            a, b = np.cos(params), np.sin(params)
            HH, HV, HD, HA, HR, HL, VH, VV, VD, VA, VR, VL, DH, DV, DD, DA, DR, DL, AH, AV, AD, AA, AR, AL, RH, RV, RD, RA, RR, RL, LH, LV, LD, LA, LR, LL  = counts
            return np.real(0.25*(1 + ((RR - RL - LR + LL) / (RR + RL + LR + LL)) + (a**2 - b**2)*((HH - HV - VH + VV) / (HH + HV + VH + VV)) + (a**2 - b**2)*((DD - DA - AD + AA) / (DD + DA + AD + AA)) - 2*a*b*(((RR - LR + RL - LL) / (RR + LR + RL + LL)) + ((RR + LR - RL - LL) / (RR + LR + RL + LL)))))
        def get_W6(params, counts):
            a, b = np.cos(params), np.sin(params)
            HH, HV, HD, HA, HR, HL, VH, VV, VD, VA, VR, VL, DH, DV, DD, DA, DR, DL, AH, AV, AD, AA, AR, AL, RH, RV, RD, RA, RR, RL, LH, LV, LD, LA, LR, LL  = counts
            return np.real(0.25*(1 - ((RR - RL - LR + LL) / (RR + RL + LR + LL)) + (a**2 - b**2)*((HH - HV - VH + VV) / (HH + HV + VH + VV)) - (a**2 - b**2)*((DD - DA - AD + AA) / (DD + DA + AD + AA)) + 2*a*b*(((RR - LR + RL - LL) / (RR + LR + RL + LL)) - ((RR + LR - RL - LL) / (RR + LR + RL + LL)))))
        
        ## W' from summer 2022 ##
        def get_Wp1(params, counts):
            theta, alpha = params[0], params[1]
            HH, HV, HD, HA, HR, HL, VH, VV, VD, VA, VR, VL, DH, DV, DD, DA, DR, DL, AH, AV, AD, AA, AR, AL, RH, RV, RD, RA, RR, RL, LH, LV, LD, LA, LR, LL  = counts
            return np.real(.25*(1 + ((HH - HV - VH + VV) / (HH + HV + VH + VV)) + np.cos(2*theta)*(((DD - DA - AD + AA) / (DD + DA + AD + AA))+((RR - RL - LR + LL) / (RR + RL + LR + LL)))+np.sin(2*theta)*np.cos(alpha)*(((HH + HV - VH - VV) / (HH + HV + VH + VV)) + ((HH - HV + VH - VV) / (HH + HV + VH + VV))) + np.sin(2*theta)*np.sin(alpha)*(((DR - DL - AR + AL) / (DR + DL + AR + AL)) - ((RD - RA - LD + LA) / (RD + RA + LD + LA)))))
        def get_Wp2(params, counts):
            theta, alpha = params[0], params[1]
            HH, HV, HD, HA, HR, HL, VH, VV, VD, VA, VR, VL, DH, DV, DD, DA, DR, DL, AH, AV, AD, AA, AR, AL, RH, RV, RD, RA, RR, RL, LH, LV, LD, LA, LR, LL  = counts
            return np.real(.25*(1 - ((HH - HV - VH + VV) / (HH + HV + VH + VV)) + np.cos(2*theta)*(((DD - DA - AD + AA) / (DD + DA + AD + AA))-((RR - RL - LR + LL) / (RR + RL + LR + LL)))+np.sin(2*theta)*np.cos(alpha)*(((HH + HV - VH - VV) / (HH + HV + VH + VV)) - ((HH - HV + VH - VV) / (HH + HV + VH + VV))) - np.sin(2*theta)*np.sin(alpha)*(((DR - DL - AR + AL) / (DR + DL + AR + AL)) + ((RD - RA - LD + LA) / (RD + RA + LD + LA)))))
        def get_Wp3(params, counts):
            theta, alpha, beta = params[0], params[1], params[2]
            HH, HV, HD, HA, HR, HL, VH, VV, VD, VA, VR, VL, DH, DV, DD, DA, DR, DL, AH, AV, AD, AA, AR, AL, RH, RV, RD, RA, RR, RL, LH, LV, LD, LA, LR, LL  = counts
            return np.real(.25 * (np.cos(theta)**2*(1 + ((HH - HV - VH + VV) / (HH + HV + VH + VV))) + np.sin(theta)**2*(1 - ((HH - HV - VH + VV) / (HH + HV + VH + VV))) + np.cos(theta)**2*np.cos(beta)*(((DD - DA - AD + AA) / (DD + DA + AD + AA)) + ((RR - RL - LR + LL) / (RR + RL + LR + LL))) + np.sin(theta)**2*np.cos(2*alpha - beta)*(((DD - DA - AD + AA) / (DD + DA + AD + AA)) - ((RR - RL - LR + LL) / (RR + RL + LR + LL))) + np.sin(2*theta)*np.cos(alpha)*((DD + DA - AD - AA) / (DD + DA + AD + AA)) + np.sin(2*theta)*np.cos(alpha - beta)*((DD - DA + AD - AA) / (DD + DA + AD + AA)) + np.sin(2*theta)*np.sin(alpha)*((RR - LR + RL - LL) / (RR + LR + RL + LL)) + np.sin(2*theta)*np.sin(alpha - beta)*((RR + LR - RL - LL) / (RR + LR + RL + LL))+np.cos(theta)**2*np.sin(beta)*(((RD - RA - LD + LA) / (RD + RA + LD + LA)) - ((DR - DL - AR + AL) / (DR + DL + AR + AL))) + np.sin(theta)**2*np.sin(2*alpha - beta)*(((RD - RA - LD + LA) / (RD + RA + LD + LA)) + ((DR - DL - AR + AL) / (DR + DL + AR + AL)))))
        def get_Wp4(params, counts):
            theta, alpha = params[0], params[1]
            HH, HV, HD, HA, HR, HL, VH, VV, VD, VA, VR, VL, DH, DV, DD, DA, DR, DL, AH, AV, AD, AA, AR, AL, RH, RV, RD, RA, RR, RL, LH, LV, LD, LA, LR, LL  = counts
            return np.real(.25*(1+((DD - DA - AD + AA) / (DD + DA + AD + AA))+np.cos(2*theta)*(((HH - HV - VH + VV) / (HH + HV + VH + VV)) + ((RR - RL - LR + LL) / (RR + RL + LR + LL))) + np.sin(2*theta)*np.cos(alpha)*(((DD - DA + AD - AA) / (DD + DA + AD + AA)) + ((DD + DA - AD - AA) / (DD + DA + AD + AA))) + np.sin(2*theta)*np.sin(alpha)*(((RH - RV - LH + LV) / (RH + RV + LH + LV)) - ((HR - HL - VR + VL) / (HR + HL + VR + VL)))))
        def get_Wp5(params, counts):
            theta, alpha = params[0], params[1]
            HH, HV, HD, HA, HR, HL, VH, VV, VD, VA, VR, VL, DH, DV, DD, DA, DR, DL, AH, AV, AD, AA, AR, AL, RH, RV, RD, RA, RR, RL, LH, LV, LD, LA, LR, LL  = counts
            return np.real(.25*(1-((DD - DA - AD + AA) / (DD + DA + AD + AA))+np.cos(2*theta)*(((HH - HV - VH + VV) / (HH + HV + VH + VV)) - ((RR - RL - LR + LL) / (RR + RL + LR + LL))) + np.sin(2*theta)*np.cos(alpha)*(((DD - DA + AD - AA) / (DD + DA + AD + AA)) - ((DD + DA - AD - AA) / (DD + DA + AD + AA))) - np.sin(2*theta)*np.sin(alpha)*(((RH - RV - LH + LV) / (RH + RV + LH + LV)) - ((HR - HL - VR + VL) / (HR + HL + VR + VL)))))
        def get_Wp6(params, counts):
            theta, alpha, beta = params[0], params[1], params[2]
            HH, HV, HD, HA, HR, HL, VH, VV, VD, VA, VR, VL, DH, DV, DD, DA, DR, DL, AH, AV, AD, AA, AR, AL, RH, RV, RD, RA, RR, RL, LH, LV, LD, LA, LR, LL  = counts
            return np.real(.25*(np.cos(theta)**2*np.cos(alpha)**2*(1 + ((HH - HV - VH + VV) / (HH + HV + VH + VV)) + ((HH + HV - VH - VV) / (HH + HV + VH + VV)) + ((HH - HV + VH - VV) / (HH + HV + VH + VV))) + np.cos(theta)**2*np.sin(alpha)**2*(1 - ((HH - HV - VH + VV) / (HH + HV + VH + VV)) + ((HH + HV - VH - VV) / (HH + HV + VH + VV)) - ((HH - HV + VH - VV) / (HH + HV + VH + VV))) + np.sin(theta)**2*np.cos(beta)**2*(1 + ((HH - HV - VH + VV) / (HH + HV + VH + VV)) - ((HH + HV - VH - VV) / (HH + HV + VH + VV)) - ((HH - HV + VH - VV) / (HH + HV + VH + VV))) + np.sin(theta)**2*np.sin(beta)**2*(1 - ((HH - HV - VH + VV) / (HH + HV + VH + VV)) - ((HH + HV - VH - VV) / (HH + HV + VH + VV)) + ((HH - HV + VH - VV) / (HH + HV + VH + VV))) + np.sin(2*theta)*np.cos(alpha)*np.cos(beta)*(((DD - DA - AD + AA) / (DD + DA + AD + AA)) + ((RR - RL - LR + LL) / (RR + RL + LR + LL))) + np.sin(2*theta)*np.sin(alpha)*np.sin(beta)*(((DD - DA - AD + AA) / (DD + DA + AD + AA)) - ((RR - RL - LR + LL) / (RR + RL + LR + LL))) + np.sin(2*theta)*np.cos(alpha)*np.sin(beta)*(((RH - RV - LH + LV) / (RH + RV + LH + LV)) + ((RR - LR + RL - LL) / (RR + LR + RL + LL))) + np.sin(2*theta)*np.sin(alpha)*np.cos(beta)*(((RH - RV - LH + LV) / (RH + RV + LH + LV)) - ((RR - LR + RL - LL) / (RR + LR + RL + LL))) - np.cos(theta)**2*np.sin(2*alpha)*(((HR - HL - VR + VL) / (HR + HL + VR + VL)) + ((RR + LR - RL - LL) / (RR + LR + RL + LL))) - np.sin(theta)**2*np.sin(2*beta)*(((HR - HL - VR + VL) / (HR + HL + VR + VL)) - ((RR + LR - RL - LL) / (RR + LR + RL + LL)))))
        def get_Wp7(params, counts):
            theta, alpha = params[0], params[1]
            HH, HV, HD, HA, HR, HL, VH, VV, VD, VA, VR, VL, DH, DV, DD, DA, DR, DL, AH, AV, AD, AA, AR, AL, RH, RV, RD, RA, RR, RL, LH, LV, LD, LA, LR, LL  = counts
            return np.real(.25*(1 + ((RR - RL - LR + LL) / (RR + RL + LR + LL))+np.cos(2*theta)*(((HH - HV - VH + VV) / (HH + HV + VH + VV)) + ((DD - DA - AD + AA) / (DD + DA + AD + AA))) + np.sin(2*theta)*np.cos(alpha)*(((HD - HA - VD + VA) / (HD + HA + VD + VA)) - ((DH - DV - AH + AV) / (DH + DV + AH + AV))) - np.sin(2*theta)*np.sin(alpha)*(((RR - LR + RL - LL) / (RR + LR + RL + LL))+((RR + LR - RL - LL) / (RR + LR + RL + LL)))))
        def get_Wp8(params, counts):
            theta, alpha = params[0], params[1]
            HH, HV, HD, HA, HR, HL, VH, VV, VD, VA, VR, VL, DH, DV, DD, DA, DR, DL, AH, AV, AD, AA, AR, AL, RH, RV, RD, RA, RR, RL, LH, LV, LD, LA, LR, LL  = counts
            return np.real(.25*(1 - ((RR - RL - LR + LL) / (RR + RL + LR + LL)) + np.cos(2*theta)*(((HH - HV - VH + VV) / (HH + HV + VH + VV))-((DD - DA - AD + AA) / (DD + DA + AD + AA))) + np.sin(2*theta)*np.cos(alpha)*(((HD - HA - VD + VA) / (HD + HA + VD + VA))+((DH - DV - AH + AV) / (DH + DV + AH + AV)))+np.sin(2*theta)*np.sin(alpha)*(((RR - LR + RL - LL) / (RR + LR + RL + LL)) - ((RR + LR - RL - LL) / (RR + LR + RL + LL)))))
        def get_Wp9(params, counts):
            theta, alpha, beta = params[0], params[1], params[2]
            HH, HV, HD, HA, HR, HL, VH, VV, VD, VA, VR, VL, DH, DV, DD, DA, DR, DL, AH, AV, AD, AA, AR, AL, RH, RV, RD, RA, RR, RL, LH, LV, LD, LA, LR, LL  = counts
            return np.real(.25*(np.cos(theta)**2*np.cos(alpha)**2*(1 + ((HH - HV - VH + VV) / (HH + HV + VH + VV)) + ((HH + HV - VH - VV) / (HH + HV + VH + VV)) + ((HH - HV + VH - VV) / (HH + HV + VH + VV))) + np.cos(theta)**2*np.sin(alpha)**2*(1 - ((HH - HV - VH + VV) / (HH + HV + VH + VV)) + ((HH + HV - VH - VV) / (HH + HV + VH + VV)) - ((HH - HV + VH - VV) / (HH + HV + VH + VV))) + np.sin(theta)**2*np.cos(beta)**2*(1 + ((HH - HV - VH + VV) / (HH + HV + VH + VV)) - ((HH + HV - VH - VV) / (HH + HV + VH + VV)) - ((HH - HV + VH - VV) / (HH + HV + VH + VV))) + np.sin(theta)**2*np.sin(beta)**2*(1 - ((HH - HV - VH + VV) / (HH + HV + VH + VV)) - ((HH + HV - VH - VV) / (HH + HV + VH + VV)) + ((HH - HV + VH - VV) / (HH + HV + VH + VV))) + np.sin(2*theta)*np.cos(alpha)*np.cos(beta)*(((DD - DA - AD + AA) / (DD + DA + AD + AA)) + ((RR - RL - LR + LL) / (RR + RL + LR + LL))) + np.sin(2*theta)*np.sin(alpha)*np.sin(beta)*(((DD - DA - AD + AA) / (DD + DA + AD + AA)) - ((RR - RL - LR + LL) / (RR + RL + LR + LL))) + np.cos(theta)**2*np.sin(2*alpha)*(((DD - DA + AD - AA) / (DD + DA + AD + AA)) + ((HD - HA - VD + VA) / (HD + HA + VD + VA))) + np.sin(theta)**2*np.sin(2*beta)*(((DD - DA + AD - AA) / (DD + DA + AD + AA)) - ((HD - HA - VD + VA) / (HD + HA + VD + VA))) + np.sin(2*theta)*np.cos(alpha)*np.sin(beta)*(((DD + DA - AD - AA) / (DD + DA + AD + AA)) + ((DH - DV - AH + AV) / (DH + DV + AH + AV)))+ np.sin(2*theta)*np.sin(alpha)*np.cos(beta)*(((DD + DA - AD - AA) / (DD + DA + AD + AA)) - ((DH - DV - AH + AV) / (DH + DV + AH + AV)))))

        def get_nom(params, expec_vals, func):
            '''For use in error propagation; returns the nominal value of the function'''
            w = func(params, expec_vals)
            return unp.nominal_values(w)


        #### TODO: Rewrite optimization
        # 
        # 1) use a set of random initial guess (x0,x1,x2) as starting points (for scipy minimze)
        # 2) choose best random initial guess (via scipy minimization)
        # 3) use gradient descent optimization (look into tensorflow or scipy)

        # now perform optimization; break into three groups based on the number of params to optimize
        all_W = [get_W1,get_W2, get_W3, get_W4, get_W5, get_W6, get_Wp1, get_Wp2, get_Wp3, get_Wp4, get_Wp5, get_Wp6, get_Wp7, get_Wp8, get_Wp9]
        W_expec_vals = []
        min_params = []

        # defining these min_W functions outside of the for loop should improve efficiency
        def min_W(x0, expt, bounds):
            '''
            Returns a scipy object that has the function that gets minimized
            and the params used to minimize
            '''
            if expt:
                arr = get_nom
                args = (counts, W)
            else:
                arr = W
                args = (counts,)
            return minimize(arr, x0=x0, args=args, bounds=bounds)
        
        def min_W_val(x0, bounds):
                    # returns minimum expectation value of W
                    return min_W(x0, expt, bounds).fun
        def min_W_params(x0, bounds):
                    # returns the parameters that got minimized
                    return min_W(x0, expt, bounds).x
        
        # Look into sklearn SGDRegressor or tensorflow
        def gradient_descent(guess, params_arr, bounds, zeta=0.7, num_reps = 50):
            """
            Does gradient descent optimization

            Params:
                guess: an InitGuess object that is the best guess from the scipy minimization step
                params_arr: generates a new random guess based on how many params the W has
                bounds: ensures min_W_val is called with the right bounds based on number of params for W
                zeta: learning rate 
                num_reps: int, number of times to run the optimization
            Returns:
                an InitGuess object that represents the optimized W
            """
            x0 = guess.x0
            w_min_val = guess.w_val
            w_min_params = guess.w_params
            isi = 0 # index since last improvement
            for _ in range(num_reps): # repeat 10 times and take the minimum
                if isi == num_reps//2: # if isi hasn't improved in a while, reset to random initial guess
                    x0 = params_arr
                else:
                    grad = approx_fprime(guess.x0, min_W_val, 1e-6)
                    if np.all(grad < 1e-5*np.ones(len(grad))):
                        break
                    else:
                        x0 = x0 - zeta*grad

                w_val = min_W_val(x0, bounds)
                w_params = min_W_params(x0, bounds)
                                
                if w_val < w_min_val:
                    w_min_val = w_val
                    w_min_params = w_params
                    isi=0
                else:
                    isi+=1
        
        for i, W in enumerate(all_W):
            if i <= 5: # These Ws only have theta, so just optimize theta
                bounds_t = [(0, np.pi)]
                params_arr_t = [np.random.rand()*np.pi]
                # Try three different starting conditions (initial guesses)
                # These are all initial guesses
                x0 = InitGuess(bounds_t, params_arr_t)
                x1 = InitGuess(bounds_t, params_arr_t)
                x2 = InitGuess(bounds_t, params_arr_t)
                
                # Using scipy to minimize the Ws and choose the best initial guess
                best_guess = min(x0, x1, x2)
                x0 = best_guess.x0
                w_min_val = best_guess.w_val
                w_min_params = best_guess.w_params

                # Use best initial guess in Gradient descent to minimze Ws further
                gradient_descent(best_guess, params_arr_t, bounds_t)
            
            # These witnesses have three parameters to be minimized (theta, alpha, and beta)
            elif i==8 or i==11 or i==14:
                bounds_tab = [(0, np.pi/2),(0, np.pi*2), (0, np.pi*2)]
                params_arr_tab = [np.random.rand()*np.pi/2, np.random.rand()*2*np.pi, np.random.rand()*2*np.pi]
                
                # Two different starting conditions (init. guesses)
                # Process is the same as the one before (just with different parameters)
                x0 = InitGuess(bounds_tab, params_arr_tab)
                x1 = InitGuess(bounds_tab, params_arr_tab)

                best_guess = min(x0, x1)
                gradient_descent(best_guess, params_arr_tab, bounds_tab)
                
            # The rest of the witnesses have 2 parameters to minimize (theta and alpha)
            else:
                bounds_ta = [(0, np.pi/2),(0, np.pi*2)]
                params_arr_ta = [np.random.rand()*np.pi/2, np.random.rand()*2*np.pi]
                    
                # Two init. guesses
                x0 = InitGuess(bounds_ta, params_arr_ta)
                x1 = InitGuess(bounds_ta, params_arr_ta)

                best_guess = min(x0, x1)
                gradient_descent(best_guess, params_arr_ta, bounds_ta)

            if expt: # automatically calculate uncertainty
                W_expec_vals.append(W(w_min_params, counts))
            if return_params:
                min_params.append(w_min_params)
            else:
                W_expec_vals.append(w_min_val)
        W_min = np.real(min(W_expec_vals[:6]))
        try:
            Wp_t1 = np.real(min(W_expec_vals[6:9])[0])
            Wp_t2 = np.real(min(W_expec_vals[9:12])[0])
            Wp_t3 = np.real(min(W_expec_vals[12:15])[0])
        except TypeError:
            Wp_t1 = np.real(min(W_expec_vals[6:9]))
            Wp_t2 = np.real(min(W_expec_vals[9:12]))
            Wp_t3 = np.real(min(W_expec_vals[12:15]))
        
        # For testing and for specific cases
        if verbose:
            #print('i got to verbosity')
            # Define dictionary to get name of
            all_W = ['W1','W2', 'W3', 'W4', 'W5', 'W6', 'Wp1', 'Wp2', 'Wp3', 'Wp4', 'Wp5', 'Wp6', 'Wp7', 'Wp8', 'Wp9']
            index_names = {i: name for i, name in enumerate(all_W)}
           
            W_param = [x for _,x in sorted(zip(W_expec_vals[:6], min_params[:6]))][0]
            Wp_t1_param = [x for _,x in sorted(zip(W_expec_vals[6:9], min_params[6:9]))][0]
            Wp_t2_param = [x for _,x in sorted(zip(W_expec_vals[9:12], min_params[9:12]))][0]
            Wp_t3_param = [x for _,x in sorted(zip(W_expec_vals[12:15], min_params[12:15]))][0]
           
           
            W_exp_val_ls = []
            for val in W_expec_vals:
                W_exp_val_ls.append(unp.nominal_values(val))
            
            W_min_name = [x for _,x in sorted(zip(W_exp_val_ls[:6], all_W[:6]))][0]
            Wp1_min_name = [x for _,x in sorted(zip(W_exp_val_ls[6:9], all_W[6:9]))][0]
            Wp2_min_name = [x for _,x in sorted(zip(W_exp_val_ls[9:12], all_W[9:12]))][0]
            Wp3_min_name = [x for _,x in sorted(zip(W_exp_val_ls[12:15], all_W[12:15]))][0]
            
            print('Wp2 and its params are:', W_expec_vals[7], min_params[7])
            print('The found W and param are:', Wp_t1, Wp1_min_name, Wp_t1_param)

            if not return_params:
                return W_min, Wp_t1, Wp_t2, Wp_t3, W_min_name, Wp1_min_name, Wp2_min_name, Wp3_min_name
            else:
                # return same as above but with the minimum params list at end
                return W_min, Wp_t1, Wp_t2, Wp_t3, W_min_name, Wp1_min_name, Wp2_min_name, Wp3_min_name, W_param, Wp_t1_param, Wp_t2_param, Wp_t3_param
                
        else:
            return W_min, Wp_t1, Wp_t2, Wp_t3
        
        # return W_expec_vals

    # Using theoretical data
    else: # use operators instead like in eritas's matlab code
        # column vectors
        HH = np.array([1, 0, 0, 0]).reshape((4,1))
        HV = np.array([0, 1, 0, 0]).reshape((4,1))
        VH = np.array([0, 0, 1, 0]).reshape((4,1))
        VV = np.array([0, 0, 0, 1]).reshape((4,1))

        # get the operators
        # take rank 1 projector and return witness
        def get_witness(phi):
            ''' Helper function to compute the witness operator for a given state and return trace(W*rho) for a given state rho.'''
            W = phi * op.adjoint(phi)
            W = op.partial_transpose(W) # take partial transpose
            return np.real(np.trace(W @ rho)) # minimizing this gives the expectation value of witness
        
        
        def get_W_matrix(state):
            return op.partial_transpose(state * op.adjoint(state))

        
        # Only difference for witnesses is how they are calculated

        ## ------ for W ------ ##
        def get_W1(param):
            a,b = np.cos(param), np.sin(param)
            phi1 = a*PHI_P + b*PHI_M
            return get_W_matrix(phi1)
        def get_W2(param):
            a,b = np.cos(param), np.sin(param)
            phi2 = a*PSI_P + b*PSI_M
            return get_witness(phi2)
        def get_W3(param):
            a,b = np.cos(param), np.sin(param)
            phi3 = a*PHI_P + b*PSI_P
            return get_witness(phi3)
        def get_W4(param):
            a,b = np.cos(param), np.sin(param)
            phi4 = a*PHI_M + b*PSI_M
            return get_witness(phi4)
        def get_W5(param):
            a,b = np.cos(param), np.sin(param)
            phi5 = a*PHI_P + 1j*b*PSI_M
            return get_witness(phi5)
        def get_W6(param):
            a,b = np.cos(param), np.sin(param)
            phi6 = a*PHI_M + 1j*b*PSI_P
            return get_witness(phi6)

        ## ------ for W' ------ ##
        def get_Wp1(params):
            theta, alpha = params[0], params[1]
            phi1_p = np.cos(theta)*PHI_P + np.exp(1j*alpha)*np.sin(theta)*PHI_M
            return get_witness(phi1_p)
        def get_Wp2(params):
            theta, alpha = params[0], params[1]
            phi2_p = np.cos(theta)*PSI_P + np.exp(1j*alpha)*np.sin(theta)*PSI_M
            return get_witness(phi2_p)
        def get_Wp3(params):
            theta, alpha, beta = params[0], params[1], params[2]
            phi3_p = 1/np.sqrt(2) * (np.cos(theta)*HH + np.exp(1j*(beta - alpha))*np.sin(theta)*HV + np.exp(1j*alpha)*np.sin(theta)*VH + np.exp(1j*beta)*np.cos(theta)*VV)
            return get_witness(phi3_p)
        def get_Wp4(params):
            theta, alpha = params[0], params[1]
            phi4_p = np.cos(theta)*PHI_P + np.exp(1j*alpha)*np.sin(theta)*PSI_P
            return get_witness(phi4_p)
        def get_Wp5(params):
            theta, alpha = params[0], params[1]
            phi5_p = np.cos(theta)*PHI_M + np.exp(1j*alpha)*np.sin(theta)*PSI_M
            return get_witness(phi5_p)
        def get_Wp6(params):
            theta, alpha, beta = params[0], params[1], params[2]
            phi6_p = np.cos(theta)*np.cos(alpha)*HH + 1j * np.cos(theta)*np.sin(alpha)*HV + 1j * np.sin(theta)*np.sin(beta)*VH + np.sin(theta)*np.cos(beta)*VV
            return get_witness(phi6_p)
        def get_Wp7(params):
            theta, alpha = params[0], params[1]
            phi7_p = np.cos(theta)*PHI_P + np.exp(1j*alpha)*np.sin(theta)*PSI_M
            return get_witness(phi7_p)
        def get_Wp8(params):
            theta, alpha = params[0], params[1]
            phi8_p = np.cos(theta)*PHI_M + np.exp(1j*alpha)*np.sin(theta)*PSI_P
            return get_witness(phi8_p)
        def get_Wp9(params):
            theta, alpha, beta = params[0], params[1], params[2]
            phi9_p = np.cos(theta)*np.cos(alpha)*HH + np.cos(theta)*np.sin(alpha)*HV + np.sin(theta)*np.sin(beta)*VH + np.sin(theta)*np.cos(beta)*VV
            return get_witness(phi9_p)
        
        # W''
        def get_w_pp_a1(params):
            """
            Witness includes szx, syz, and szx
            params - list of parameters to optimize, note a^2 + b^2 + c^2 + d^2 = 1 and a,b,c,d > 0 and real.
            returns - the expectation value of the witness with the input state, rho
            """
            theta, alpha, beta = params[0], params[1], params[2] #optimizing parameters
            #Witness constraints
            a= np.cos(theta)*np.cos(alpha)
            b= np.sin(theta)*np.sin(alpha)
            c= np.cos(theta)*np.sin(alpha)
            d= -a*b/c
            #constructs the witness
            phi = a*HH + b*np.exp(1j*beta)*HV + c*np.exp(1j*beta)*VH + d*VV
            return get_witness(phi)
        
        def get_w_pp_a2(params):
            """
            Witness includes szx, syz, and szx
            params - list of parameters to optimize, note a^2 + b^2 + c^2 + d^2 = 1 and a,b,c,d > 0 and real.
            returns - the expectation value of the witness with the input state, rho
            """
            theta, alpha, beta = params[0], params[1], params[2] #optimizing parameters
            #Witness constraints
            a= np.cos(theta)*np.cos(alpha)
            b= np.sin(theta)*np.sin(alpha)
            c= np.cos(theta)*np.sin(alpha)
            d= -a*c/b
            #constructs the witness
            phi = a*HH + b*np.exp(1j*beta)*HV + c*np.exp(1j*beta)*VH + d*VV
            return get_witness(phi)

        def get_w_pp_b1(params):
            """
            Witness includes szx, syz, and szy
            params - list of parameters to optimize, note a^2 + b^2 + c^2 + d^2 = 1 and a,b,c,d > 0 and real.
            returns - the expectation value of the witness with the input state, rho
            """
            theta, alpha, beta = params[0], params[1], params[2] #optimizing parameters
            #Witness constraints
            a= np.cos(theta)*np.cos(alpha)
            b= np.sin(theta)*np.sin(alpha)
            c= np.cos(theta)*np.sin(alpha)
            d= a*b/c
            #constructs the witness
            phi = a*HH + b*np.exp(1j*beta)*HV + c*np.exp(1j*beta)*VH + d*VV
            return get_witness(phi)
        
        def get_w_pp_b2(params):
            """
            Witness includes szx, syz, and szy
            params - list of parameters to optimize, note a^2 + b^2 + c^2 + d^2 = 1 and a,b,c,d > 0 and real.
            returns - the expectation value of the witness with the input state, rho
            """
            theta, alpha, beta = params[0], params[1], params[2] #optimizing parameters
            #Witness constraints
            a= np.cos(theta)*np.cos(alpha)
            b= np.sin(theta)*np.sin(alpha)
            c= np.cos(theta)*np.sin(alpha)
            d= a*c/b
            #constructs the witness
            phi = a*HH + b*np.exp(1j*beta)*HV + c*np.exp(1j*beta)*VH + d*VV
            return get_witness(phi)
        

        if not(ads_test): 
            all_W = [get_W1,get_W2, get_W3, get_W4, get_W5, get_W6, get_Wp1, get_Wp2, get_Wp3, get_Wp4, get_Wp5, get_Wp6, get_Wp7, get_Wp8, get_Wp9, get_w_pp_a1, get_w_pp_a2, get_w_pp_b1,  get_w_pp_b2]
            W_expec_vals = []
            if return_params: # to log the params
                min_params = []
            
            # Same optimization (essentially) as for experimental
            for i, W in enumerate(all_W):
                if i <= 5: # just theta optimization
                    # get initial guess at boundary
                    def min_W(x0):
                        do_min = minimize(W, x0=x0, bounds=[(0, np.pi)])
                        return do_min['fun']
                    x0 = [np.random.rand()*np.pi]
                    w0 = min_W(x0)
                    x1 = [np.random.rand()*np.pi]
                    w1 = min_W(x1)
                    if w0 < w1:
                        w_min = w0
                        x0_best = x0
                    else:
                        w_min = w1
                        x0_best = x1
                    if optimize:
                        isi = 0 # index since last improvement
                        for _ in range(num_reps): # repeat 10 times and take the minimum
                            if gd:
                                if isi == num_reps//2: # if isi hasn't improved in a while, reset to random initial guess
                                    x0 = [np.random.rand()*np.pi]
                                else:
                                    grad = approx_fprime(x0, min_W, 1e-6)
                                    if np.all(grad < 1e-5*np.ones(len(grad))):
                                        break
                                    else:
                                        x0 = x0 - zeta*grad
                            else:
                                x0 = [np.random.rand()*np.pi]

                            w = min_W(x0)
                            
                            if w < w_min:
                                w_min = w
                                x0_best = x0
                                isi=0
                            else:
                                isi+=1
                    # print('------------------')
                elif i==8 or i==11 or i==14: # theta, alpha, and beta
                    def min_W(x0):
                        do_min = minimize(W, x0=x0, bounds=[(0, np.pi/2),(0, np.pi*2), (0, np.pi*2)])
                        return do_min['fun']

                    x0 = [np.random.rand()*np.pi/2, np.random.rand()*2*np.pi, np.random.rand()*2*np.pi]
                    w0 = min_W(x0)
                    x1 = [np.random.rand()*np.pi/2, np.random.rand()*2*np.pi, np.random.rand()*2*np.pi]
                    w1 = min_W(x1)
                    if w0 < w1:
                        w_min = w0
                        x0_best = x0
                    else:
                        w_min = w1
                        x0_best = x1
                    if optimize:
                        isi = 0 # index since last improvement
                        for _ in range(num_reps): # repeat 10 times and take the minimum
                            if gd:
                                if isi == num_reps//2: # if isi hasn't improved in a while, reset to random initial guess
                                    x0 = [np.random.rand()*np.pi/2, np.random.rand()*2*np.pi, np.random.rand()*2*np.pi]
                                else:
                                    grad = approx_fprime(x0, min_W, 1e-6)
                                    if np.all(grad < 1e-5*np.ones(len(grad))):
                                        x0 = [np.random.rand()*np.pi/2, np.random.rand()*2*np.pi, np.random.rand()*2*np.pi]
                                    else:
                                        x0 = x0 - zeta*grad
                            else:
                                x0 = [np.random.rand()*np.pi/2, np.random.rand()*2*np.pi, np.random.rand()*2*np.pi]

                            w = min_W(x0)
                            
                            if w < w_min:
                                w_min = w
                                x0_best = x0
                                isi=0
                            else:
                                isi+=1
                elif i == 15 or i==16 or i==17 or i==18: # the W'' witness
                    def min_W(x0):
                        # print(x0)
                        do_min = minimize(W, x0=x0, bounds=[(-np.pi/2 + 0.01,np.pi/2-0.01), (0.01,np.pi-0.01), (0,2*np.pi)])
                        # print(do_min['x'])
                        return do_min['fun']

                    #Begin with two random states
                    x0 = [np.random.rand()*np.pi/2,np.random.rand()*np.pi,np.random.rand()*2*np.pi] #generates a random set of parameters based on its relationship to beta
                    w0 = min_W(x0)
                    x1 = [np.random.rand()*np.pi/2,np.random.rand()*np.pi,np.random.rand()*2*np.pi]
                    w1 = min_W(x1)
                    
                    if w0 < w1: #choose the better one 
                        w_min = w0
                        x0_best = x0
                    else:
                        w_min = w1
                        x0_best = x1
                    if optimize: #Optimize the witness based on the previous best
                        isi = 0 # index since last improvement
                        count = 0
                        for _ in range(num_reps): # repeat numsteps times and take the minimum
                            count += 1
                            if gd:
                                if isi == num_reps//2: # if isi hasn't improved in a while, reset to random initial guess
                                    x0 = [np.random.rand()*np.pi/2,np.random.rand()*np.pi,np.random.rand()*2*np.pi]
                                else:
                                    grad = approx_fprime(x0, min_W, 1e-6) #Error here assk oscar why it might be doing this>
                                    if np.all(grad < 1e-5*np.ones(len(grad))):
                                        x0 = [np.random.rand()*np.pi/2, np.random.rand()*2*np.pi, np.random.rand()*2*np.pi]
                                    else:
                                        x0 = x0 - zeta*grad          
                            else:
                                x0 = [np.random.rand()*np.pi/2,np.random.rand()*np.pi,np.random.rand()*2*np.pi]
                            w = min_W(x0)
                            
                            if w < w_min:
                                w_min = w
                                x0_best = x0
                                isi=0
                            else:
                                isi+=1
                else:# theta and alpha
                    def min_W(x0, return_params = False):
                        if return_params == False:
                            return minimize(W, x0=x0, bounds=[(0, np.pi/2),(0, np.pi*2)])['fun']
                        else:
                            return minimize(W, x0=x0, bounds=[(0, np.pi/2),(0, np.pi*2)])
                        
                    x0 = [np.random.rand()*np.pi/2, np.random.rand()*2*np.pi]
                    w0 = min_W(x0)
                    x1 = [np.random.rand()*np.pi/2, np.random.rand()*2*np.pi]
                    w1 = min_W(x1)
                    if w0 < w1:
                        w_min = w0
                        x0_best = x0
                    else:
                        w_min = w1
                        x0_best = x1
                    if optimize:
                        isi = 0 # index since last improvement
                        for _ in range(num_reps): # repeat 10 times and take the minimum
                            if gd:
                                if isi == num_reps//2: # if isi hasn't improved in a while, reset to random initial guess
                                    x0 = [np.random.rand()*np.pi/2, np.random.rand()*2*np.pi]
                                else:
                                    grad = approx_fprime(x0, min_W, 1e-6)
                                    if np.all(grad < 1e-5*np.ones(len(grad))):
                                        x0 = [np.random.rand()*np.pi/2, np.random.rand()*2*np.pi]
                                    else:
                                        x0 = x0 - zeta*grad
                            else:
                                x0 = [np.random.rand()*np.pi/2, np.random.rand()*2*np.pi, np.random.rand()*2*np.pi]

                            w = min_W(x0)
                            
                            if w < w_min:
                                w_min = w
                                x0_best = x0
                                isi=0
                            else:
                                isi+=1
                if return_params:
                    ### Note that these are not the correct parameters!! This must be fixed ###
                    min_params.append(x0_best)
                W_expec_vals.append(w_min)
            # print('W', np.round(W_expec_vals[:6], 3))
            # print('W\'', np.round(W_expec_vals[6:], 3))
            # find min witness expectation values
            W_min = min(W_expec_vals[:6])
            Wp_t1 = min(W_expec_vals[6:9])
            Wp_t2 = min(W_expec_vals[9:12])
            Wp_t3 = min(W_expec_vals[12:15])
            # get the corresponding parameters
            if return_params:
                W_expec_vals_ls = []
                for val in W_expec_vals:
                    W_expec_vals_ls.append(unp.nominal_values(val))
                # sort by witness value; want the most negative, so take first element in sorted
                W_param = [x for _,x in sorted(zip(W_expec_vals_ls[:6], min_params[:6]))][0]
                Wp_t1_param = [x for _,x in sorted(zip(W_expec_vals_ls[6:9], min_params[6:9]))][0]
                Wp_t2_param = [x for _,x in sorted(zip(W_expec_vals_ls[9:12], min_params[9:12]))][0]
                Wp_t3_param = [x for _,x in sorted(zip(W_expec_vals_ls[12:15], min_params[12:15]))][0]


            if not(return_all):
                if verbose:
                    #print('i got to verbosity')
                    # Define dictionary to get name of
                    all_W = ['W1','W2', 'W3', 'W4', 'W5', 'W6', 'Wp1', 'Wp2', 'Wp3', 'Wp4', 'Wp5', 'Wp6', 'Wp7', 'Wp8', 'Wp9', 'W_pp_a1', 'W_pp_a2', 'W_pp_b1',  'W_pp_b2']
                    index_names = {i: name for i, name in enumerate(all_W)}
                
                    W_exp_val_ls = []
                    for val in W_expec_vals:
                        W_exp_val_ls.append(unp.nominal_values(val))
                    
                   
                    W_min_name = [x for _,x in sorted(zip(W_expec_vals[:6], all_W[:6]))][0]
                    Wp1_min_name = [x for _,x in sorted(zip(W_expec_vals[6:9], all_W[6:9]))][0]
                    Wp2_min_name = [x for _,x in sorted(zip(W_expec_vals[9:12], all_W[9:12]))][0]
                    Wp3_min_name = [x for _,x in sorted(zip(W_expec_vals[12:15], all_W[12:15]))][0]
                    Wpp_min_name = [x for _, x in sorted(zip(W_expec_vals[15:], all_W[15:]))][0]

                    if not return_params:
                        # Find names from dictionary and return them and their values
                        return W_min, Wp_t1, Wp_t2, Wp_t3, W_min_name, Wp1_min_name, Wp2_min_name, Wp3_min_name
                    else:
                        return W_min, Wp_t1, Wp_t2, Wp_t3, W_min_name, Wp1_min_name, Wp2_min_name, Wp3_min_name, W_param, Wp_t1_param, Wp_t2_param, Wp_t3_param
                if return_params:
                    return W_min, Wp_t1, Wp_t2, Wp_t3, W_param, Wp_t1_param, Wp_t2_param, Wp_t3_param
                else:
                    return W_min, Wp_t1, Wp_t2, Wp_t3
            else:
                if return_params:
                    return W_expec_vals, min_params
                else:
                    return W_expec_vals
        else: 
            print('i went to the 2nd else')
            W2_main= minimize(get_W2, x0=[0], bounds=[(0, np.pi)])
            W2_val = W2_main['fun']
            W2_param = W2_main['x']

            return W2_val, W2_param[0]


def test_witnesses():
    '''Calculate witness vals for select experimental states'''
    r1 = np.load("../../framework/decomp_test/rho_('E0', (45.0, 18.0))_32.npy", allow_pickle=True)
    counts1 = unp.uarray(r1[3], r1[4])
    print('45, 18, 32')
    print(counts1)
    print(compute_witnesses(r1[0], counts1, expt=True))
    print('------')
    r2 = np.load("../../framework/decomp_test/rho_('E0', (59.99999999999999, 72.0))_32.npy", allow_pickle=True)
    counts2 = unp.uarray(r2[3], r2[4])
    print('60, 72, 32')
    print(counts2)
    print(compute_witnesses(r2[0], counts2, expt=True))
    print('------')