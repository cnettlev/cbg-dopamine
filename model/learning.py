#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Original Code from:
# https://github.com/rougier/Neurosciences/tree/master/basal-ganglia/guthrie-et-al-2013
# Copyright (c) 2014, Nicolas P. Rougier, Meropi Topalidou
# Distributed under the (new) BSD License.
#
# Contributors: Nicolas P. Rougier (Nicolas.Rougier@inria.fr)
#               Meropi Topalidou (Meropi.Topalidou@inria.fr)
# -----------------------------------------------------------------------------
# Dopamine effects extension by:
#               Cristobal J. Nettle (cristobal.nettle@gmail.com)
# -----------------------------------------------------------------------------
# References:
#
# * Interaction between cognitive and motor cortico-basal ganglia loops during
#   decision making: a computational study. M. Guthrie, A. Leblois, A. Garenne,
#   and T. Boraud. Journal of Neurophysiology, 109:3025–3040, 2013.
#
# * A long Journey into Reproducible Computational Neurosciences (submitted)
#   Meropi Topalidou, Arthur Leblois, Thomas Boraud, Nicolas P. Rougier
#
# * Tonic Dopamine Effects in a Bio-inspired Robot Controller Enhance Expected 
#   Lifetime. Cristobal J. Nettle, Maria-Jose Escobar, Arthur Leblois. 
#   International Conference on Developmental Learning and on Epigenetic 
#   Robotics, ICDL-EpiRob 2016.
# -----------------------------------------------------------------------------
#
# Testing:
# python learning.py -t 200 -r100 -b 12 --ltd-constant --correlatedNoise --relativeValue --zeroValues -P --flashPlots --storePlots 10 --debug
# 
# Evaluating noise effects:
# python learning.py -t 150 -r100 -b 12 -R --ltd-constant --correlatedNoise --relativeValue --zeroValues -P --flashPlots --storePlots 10 -F noiseCrash/allBy10 --debug --ltd 0.00001 --ltp 0.0000625
#
# Proposals:
#    Use a measure as "advantage" (how good was the selected option agains what would 
#    have happend) -> This would show clearly that when the selection fails, it doesn't 
#    matter too much cause it is comparing bad cues. But it does not commit mistakes when
#    a good cue is compared with a bad one (not influencing then too much in the actual
#    performance)
#    

from dana import *
import matplotlib
import matplotlib.pyplot as plt
plt.ion()
# from matplotlib import rc
# rc('font',**{'family':'serif','serif':['cmr10'],'size':13})
# Customize matplotlib
matplotlib.rcParams.update(
    {
        'text.usetex': False,
        'font.family': 'stixgeneral',
        'mathtext.fontset': 'stix',
    }
)

#import sys                          # Added to use command line arguments
from datetime import datetime       # Added to create a store file with time data
from learningOptions import *       # Added argument options parser
import csv
from collections import OrderedDict
import itertools

# Parameters    
#!! Commented parameters are set in learningOptions.py (except if it's said 
# explicitely). Uncomment for overwriting the command line argument values (and 
# default values)
# -----------------------------------------------------------------------------
# Population size
# n = 4

# Number of trials
# nbTrials = 200

# Default trial duration (max duration)
duration = 3.5*second
afterSelectionTime = 1.5*second
stopAfterSelection = True

# Default Time resolution
dt = 1.0*millisecond

# Cortical inputs amplitude [sp/s]
cues_amplitude = 22 if options.GM2008 else ( 16 if options.indirectLoop else 22 )

# Sigmoid parameter
Vmin       =  1.0 if options.indirectLoop else  0.0
Vmax       = 20.0
Vh         = 10   if options.indirectLoop else 20.0
Vc         =  3.0 if options.indirectLoop else  6.0

# Thresholds
Cortex_h   =  -4.0 if options.indirectLoop else -3.0
Striatum_h =   0.0
STN_h      = -10.0
GPi_h      = -40.0 if options.indirectLoop else 10.0
GPe_h      = -20.0
Thalamus_h = -10.0 if options.indirectLoop else -40.0
SNc_h_base =  -DA

# Time constants
Cortex_tau   = 0.01
Striatum_tau = 0.01
STN_tau      = 0.01
GPi_tau      = 0.01
Thalamus_tau = 0.01
STR_N_tau    = 0.03
SNc_tau      = 0.01
SNc_N_tau    = 0.03
arD2_tau     = 0.5

# DA D2-autoreceptors
# D2-Autoreceptors delay [ms]
arD2_lag    = 50 * millisecond
# DA strenght on D2-Autoreceptors 
alpha_DA_arD2 = 0.1

# DA burst height produced by a reward (modulated by the prediction error)
alpha_Rew_DA = 15 # [sp/s]
# DA strenght on striatal inputs
gamma_DAth        = 1
gamma_DAstrenght  = 0.025#.25
gamma_DAbySuccess = options.DAbySuccess # default is 2 [sp/s]
alpha_SucessEMA   = 0.8
gamma_DA_LTD      = 0.025 # (1+gamma_DA_LTD * DA) -> 1.1 (4.0) - 1.15 (6.0) - 1.2 (8.0)
gamma_mGluR_LTD   = 0.01 # 60 * (1+DA) -> 60 * 5 - 60*9 -> 300-700
gamma_eCB1_LTD    = 0.1 # (1-20)

# Noise level (%)
Cortex_N        =   0.3 if options.indirectLoop else ( 0.01 * aux_X )
Striatum_N      =   0.1 if options.indirectLoop else ( 0.01 * aux_X )
Striatum_corr_N =   0.2 if options.indirectLoop else ( 0.7  * aux_Y )
STN_N           =   0.1 if options.indirectLoop else ( 0.01 * aux_X )
GPi_N           =   0.3 if options.indirectLoop else ( 0.03 * aux_X )
Thalamus_N      =   0.1 if options.indirectLoop else ( 0.01 * aux_X )
SNc_N           =   0.1 if options.indirectLoop else ( 0.01 * aux_X )

# DA buffer for a time delay on arD2
DA_buffSize = int(round(arD2_lag/dt))
DA_buff = np.zeros((DA_buffSize,))
DA_buffIndex = 0

# Learning parameters
decision_threshold = 20 if options.indirectLoop else 30
alpha_c     = 0.2  # 0.05
Wmin, Wmax = 0.5, 0.6#0.45, 0.55
#Wmin, Wmax = 0.4, 0.6

# Reward extension [ms]
delayReward = 300 * millisecond
reward_ext = 50 * millisecond + delayReward

# Data saving configuration
avoidUnsavedPlots = True
# STORE_DATA = True
# STORE_FAILED = False
# FAILED_FILE = folder+'failed_'+str(DA)

alpha_SuccessEMA = 0.95

if STORE_DATA:
    DATA = OrderedDict([('SNc',0),('SNc_h',0),('P-buff',0),('choice',0),('nchoice',0),('motTime',0),('weights',''),('mot_weights',''),('cogTime',0),
                        ('failedTrials',0),('persistence',0),('values',''),('P',0),('P-3',0),('R',0),('R-buff',0),('pA',0),
                        ('P-3',0),('LTD-LTP',''),('A',0),('A-buff',0),('pchoice',''),('Regret',0),('cumRegret',0),('c_rewards',''),('trial',0)])
    with open(filename,"w") as records:
        wtr = csv.DictWriter(records, DATA.keys(),delimiter=',')
        wtr.writeheader()

# Include learning
# learn = True 
# applyInMotorLoop = True

# Compute all the trial (true) or just until a motor decision has been made
completeTrials = True
# Force selection in motor loop for every trial (discards trials with no decision)
forceSelection = False # True

#if regPattern:
#	completeTrials = False
#	forceSelection = False

# Maximum number of failed trials. Available if forceSelection is enable
nbFailedTrials = .5*nbTrials
# Maximum number of continuous failed trials
nbContFailedTrials = 20

# Use a regular pattern for the cues selection
# regPattern = False
nbTrialsPerPattern = 50
if options.GM2008:
    pattern = np.array([0,1,2])
    nPatterns = len(pattern)
    n_availableOptions = 3
else:
    pattern = np.array([[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]])
    nPatterns = len(pattern)
    n_availableOptions = 2

probChoiceBuffSize = 10
probChoice = 0.5*np.ones((nPatterns,probChoiceBuffSize))

# Enable debug messages
# DA = 1.0
# doPrint = False
# invertAt = 100


# Initialization of the random generator
if randomInit >= 0:
    np.random.seed(randomInit)

# Helper functions
# -----------------------------------------------------------------------------
def sigmoid(V,Vmin=Vmin,Vmax=Vmax,Vh=Vh,Vc=Vc):
    return  Vmin + (Vmax-Vmin)/(1.0+np.exp((Vh-V)/Vc))

def flip(items, ncol):
    return itertools.chain(*[items[i::ncol] for i in range(ncol)])

# def rewardShape(t,rewardTime,dev=0.1):
#     return np.exp(-(t-(rewardTime+3*dev))**2/(2*dev**2))

def strThreshold(DA):
    return gamma_DAth * DA + Vh
    # return (gamma_DAstrenght * DA + 1)*Vh # Factor was 19, but considering gamma_DAstrenght, dropped to 4.75 - testing 5.5
    # return 4.75 * DA + Vh # Factor was 19, but considering gamma_DAstrenght, dropped to 4.75 - testing 5.5

def strSlope(DA):
    return Vc #* (1.0 + gamma_DAstrenght * DA)

def setDAlevels(DA):
    Striatum_cog['DA'] = DA
    Striatum_mot['DA'] = DA
    Striatum_ass['DA'] = DA

def noise(Z, level):
    Z = np.random.normal(0,level,Z.shape)*Z
    return Z

def correlatedNoise(Z, pastValue, noiseScale, base=Vmax, tau=STR_N_tau):
    noise = np.random.normal(0,noiseScale,Z.shape)*base
    delta = dt*(-pastValue + noise)/tau
    noise = pastValue + delta
    return noise

def striatalNoise(Z,pastV):
    if useCorrelatedNoise:
        return correlatedNoise(Z,pastV,Striatum_corr_N)
    else:
        return noise(Z,Striatum_N)

def init_weights(L, gain=1):
    # Wmin, Wmax = 0.25, 0.75
    W = L._weights
    N = np.random.normal(Wmean, 0.005, W.shape)
    N = np.minimum(np.maximum(N, 0.0),1.0)
    L._weights = gain*W*(Wmin + (Wmax - Wmin)*N)

def tonicDA(minSmooth = options.minSmooth):
    global smoothA, smoothR
    if options.dynamicDAoverA: # base on advantage
        if minSmooth >= 0:
            tonic = SNc_h_base - gamma_DAbySuccess * np.max((smoothA-minSmooth,0))/(1-minSmooth)
        else:
            tonic = sigmoid(smoothA,Vmin=SNc_h_base,Vmax=SNc_h_base-gamma_DAbySuccess,Vh=tonicDA_h,Vc=-minSmooth)
    else: # base on reward
        if minSmooth >= 0:
            tonic = SNc_h_base - gamma_DAbySuccess * np.max((smoothR-minSmooth,0))/(1-minSmooth)
        else:
            tonic = sigmoid(smoothR,Vmin=SNc_h_base,Vmax=SNc_h_base-gamma_DAbySuccess,Vh=tonicDA_h,Vc=-minSmooth)

    return tonic

def reset():
    for group in network.__default_network__._groups:
        # group['U'] = 0
        group['V'] = 0
        group['I'] = 0
    Cortex_mot['Iext'] = 0
    Cortex_cog['Iext'] = 0
    Cortex_ass['Iext'] = 0
    Cortex_cog['n'] = 0
    SNc_dop['Irew'] = 0
    SNc_dop['Ie_rew'] = 0
    SNc_dop['u'] = -SNc_dop['SNc_h']
    SNc_dop['V_lag'] = -SNc_dop['SNc_h']
    SNc_dop['V'] = -SNc_dop['SNc_h']
    SNc_dop['DAtoD2c'] = 0
    DA_buff[:] = -SNc_dop['SNc_h']

def partialReset():
    global motDecisionTime, cogDecisionTime, trialLtdLtp
    Cortex_mot['Iext'] = 0
    Cortex_cog['Iext'] = 0
    Cortex_ass['Iext'] = 0
    cogDecision = False
    cogDecisionTime = 3500.0
    motDecision = False
    motDecisionTime = 3500.0
    trialLtdLtp *= 0


#@after(clock.tick)
# @clock.every(100*millisecond)
# def printVh(t):
#     print(Striatum_cog['Vh'])

def resetPlot():
    global pastW,cogWDiff,cogMarker,motMarker,movMarker
    W = W_cortex_cog_to_striatum_cog
    dw = np.diag(W._weights)-np.diag(pastW)
    # print "Past Cognitive weights: ", np.diag(pastW)
    # print "Diff Cognitive weights: ", dw
    pastW = np.copy(W._weights)
    # print cogWDiff

    for p in N:
        if dw[p]>0:
            cogWDiff[p][2] = (cogWDiff[p][2]*(currentTrial)+dw[p])/(currentTrial+1)
            if cogWDiff[p][3] < dw[p]:
                cogWDiff[p][3] = dw[p]
        elif dw[p]<0:
            cogWDiff[p][0] = (cogWDiff[p][0]*(currentTrial)+dw[p])/(currentTrial+1)
            if cogWDiff[p][1] > dw[p]:
                cogWDiff[p][1] = dw[p]
        # if doPrint:
        #     print 'Cog',p,dw[p],'\t min',cogWDiff[p][1],'\t meanm',cogWDiff[p][0],'   \t max',cogWDiff[p][3],'\t meanM',cogWDiff[p][2]

    if flashPlots > 0 and (currentTrial % flashPlots) == 0:
        drawPlots()
    if pausePlots:
        while not plt.waitforbuttonpress():
            continue

    if len(cogMarker):
        cogMarker.pop(0).remove()
    if len(motMarker):
        motMarker.pop(0).remove()
    if len(movMarker):
        movMarker.pop(0).remove()

    neuralData_th.fill(None)
    neuralData_ctx.fill(None)
    neuralData_ctx_ass.fill(None)
    neuralData_snc.fill(None)
    neuralData_str.fill(None)
    neuralData_str_ass.fill(None)
    neuralData_gpi.fill(None)
    neuralData_stn.fill(None)
    if options.indirectLoop:
        neuralData_gpe.fill(None)
        neuralData_str_in.fill(None)
    setXlim_d()

def clip(V, Vmin, Vmax):
    return np.minimum(np.maximum(V, Vmin), Vmax)

def positiveClip(V):
    return np.maximum(V, 0.0)

def fillData():
    DATA['SNc']          = SNc_dop['V'][0][0]
    DATA['SNc_h']        = SNc_dop['SNc_h'][0]
    DATA['P-buff']       = np.array(P[-perfBuff:]).mean()
    DATA['choice']       = choice
    DATA['nchoice']      = nchoice if not options.GM2008 else -1
    DATA['motTime']      = motDecisionTime
    DATA['weights']      = '\t'.join(['{:.5f}'.format(i) for i in W_cortex_cog_to_striatum_cog.weights.diagonal()])
    DATA['mot_weights']  = '\t'.join(['{:.5f}'.format(i) for i in W_cortex_mot_to_striatum_mot.weights.diagonal()])
    DATA['cogTime']      = cogDecisionTime
    DATA['failedTrials'] = failedTrials
    DATA['persistence']  = 1 if np.max(Cortex_cog['V']) > 30 else 0
    DATA['values']       = '\t'.join(['{:.5f}'.format(i) for i in cog_cues_value])
    DATA['P']            = P[-1:][0]
    DATA['P-3']          = np.array(P[-3:]).mean()
    DATA['R']            = R[-1:][0]
    DATA['R-buff']       = np.array(R[-perfBuff:]).mean()
    DATA['LTD-LTP']      = '\t'.join(['{:.5f}'.format(float(i)) for i in trialLtdLtp.reshape((n*4,1))])
    DATA['A']            = A[-1:][0]
    DATA['pA']           = perceived_advantage
    DATA['A-buff']       = np.array(A[-perfBuff:]).mean()
    DATA['pchoice']      = '\t'.join(['{:.2f}'.format(np.nanmean(i)) for i in probChoice])
    DATA['Regret']       = Regret[-1:][0]
    DATA['cumRegret']    = cumRegret
    DATA['c_rewards']    = '\t'.join(['{:.2f}'.format(i) for i in cues_reward])
    DATA['trial']        = currentTrial

def printData():
    # if applyInMotorLoop:
    #     with open(filename,'a') as f:
    #         f.write("%f\t%f\t%f\t%d\t%d\t%d\t%s\t%s\t%d\n" % (DA, np.array(P).mean(), np.array(R).mean(), choice, nchoice, motDecisionTime,
    #             '\t'.join(['{:.2f}'.format(i) for i in W_cortex_cog_to_striatum_cog.weights.diagonal()]),
    #             '\t'.join(['{:.2f}'.format(i) for i in W_cortex_mot_to_striatum_mot.weights.diagonal()]),cogDecisionTime))
    # else:
    with open(filename,'a') as records:
        wtr = csv.DictWriter(records, DATA.keys(),delimiter=',')
        fillData()
        wtr.writerow(DATA)
    # with open(filename,'a') as f:
    #     finalActivity = np.max(Cortex_cog['V']) 
    #     persistence = 1 if finalActivity > 30 else 0
    #     f.write("%f\t%f\t%d\t%s\t%d\t%d\t%d\t%s\t%f\t%f\t%f\t%f\t%s\t%f\t%f\n" % (SNc_dop['V'], np.array(P[-perfBuff:]).mean(), choice,
    #         '\t'.join(['{:.5f}'.format(i) for i in W_cortex_cog_to_striatum_cog.weights.diagonal()]),cogDecisionTime,
    #         failedTrials,persistence,'\t'.join(['{:.5f}'.format(i) for i in cog_cues_value]),P[-1:][0],np.array(P[-3:]).mean(),R[-1:][0], np.array(R[-perfBuff:]).mean(),
    #         '\t'.join(['{:.5f}'.format(float(i)) for i in trialLtdLtp.reshape((n*4,1))]), A[-1:][0], np.array(A[-perfBuff:]).mean()) )

def D2_IPSC_kernel(t,t_DelRew):
    if t < t_DelRew:
        return 0
    return np.exp(-(t-t_DelRew)/arD2_tau)

def convolv_D2Kernel(t,currentValue,input):
    return currentValue + input * D2_IPSC_kernel(t,motDecisionTime/1000+arD2_lag)

SNc_dop   = zeros((1,1), """ D2_IPSC = - alpha_DA_arD2 * DAtoD2c;
                             Ir = np.maximum(Irew, Ie_rew);
                             I = Ir + D2_IPSC;
                             n = noise(I,SNc_N); 
                             It = I + n;
                             SNc_h = tonicDA();
                             u = positiveClip(It - SNc_h);
                             dV/dt = (-V + u)/SNc_tau; Irew; Ie_rew; V_lag; DAtoD2c""")
                             # n = correlatedNoise(I,n,SNc_N,alpha_Rew_DA,SNc_N_tau);
                             # du_DA/dt = (-u_DA + (It - SNc_h))/SNc_tau;
                             # V = positiveClip(u_DA); Irew; Ie_rew; SNc_h; fDA_del; DAtoD2c""")

Cortex_cog   = zeros((n,1), """Is = I + Iext; 
                             n = noise(Is,Cortex_N);
                             It = Is + n;
                             u = positiveClip(It - Cortex_h);
                             dV/dt = (-V + u)/Cortex_tau; I; Iext""")
Cortex_mot   = zeros((1,n), """Is = I + Iext; 
                             n = noise(Is,Cortex_N);
                             It = Is + n;
                             u = positiveClip(It - Cortex_h);
                             dV/dt = (-V + u)/Cortex_tau; I; Iext""")
Cortex_ass   = zeros((n,n), """Is = I + Iext; 
                             n = noise(Is,Cortex_N);
                             It = Is + n;
                             u = positiveClip(It - Cortex_h);
                             dV/dt = (-V + u)/Cortex_tau; I; Iext""")

Striatum_cog = zeros((n,1), """Is = I*(1 + 1.0*gamma_DAstrenght*DA);
                             n = striatalNoise(Is,n);
                             It = Is + n;
                             Vh = strThreshold(DA);
                             Vc = strSlope(DA);
                             u = sigmoid(It - Striatum_h,Vmin,Vmax,Vh,Vc);
                             dV/dt = (-V + u)/Striatum_tau; I; DA""")

Striatum_mot = zeros((1,n), """Is = I*(1 + 1.0*gamma_DAstrenght*DA);
                             n = striatalNoise(Is,n);
                             It = Is + n;
                             Vh = strThreshold(DA);
                             Vc = strSlope(DA);
                             u = sigmoid(It - Striatum_h,Vmin,Vmax,Vh,Vc);
                             dV/dt = (-V + u)/Striatum_tau; I; DA""")

Striatum_ass = zeros((n,n), """Is = I*(1 + 1.0*gamma_DAstrenght*DA);
                             n = striatalNoise(Is,n);
                             It = Is + n;
                             Vh = strThreshold(DA);
                             Vc = strSlope(DA);
                             u = sigmoid(It - Striatum_h,Vmin,Vmax,Vh,Vc);
                             dV/dt = (-V + u)/Striatum_tau; I; DA""")

STN_cog      = zeros((n,1), """Is = I;
                             n = noise(Is,STN_N);
                             It = Is + n;
                             u = positiveClip(It - STN_h);
                             dV/dt = (-V + u)/STN_tau; I""")
STN_mot      = zeros((1,n), """Is = I;
                             n = noise(Is,STN_N);
                             It = Is + n;
                             u = positiveClip(It - STN_h);
                             dV/dt = (-V + u)/STN_tau; I""")
GPi_cog      = zeros((n,1), """Is = I;
                             n = noise(Is,GPi_N);
                             It = Is + n;
                             u = positiveClip(It - GPi_h);
                             dV/dt = (-V + u)/GPi_tau; I""")
GPi_mot      = zeros((1,n), """Is = I;
                             n = noise(Is,GPi_N);
                             It = Is + n;
                             u = positiveClip(It - GPi_h);
                             dV/dt = (-V + u)/GPi_tau; I""")
Thalamus_cog = zeros((n,1), """Is = I;
                             n = noise(Is,Thalamus_N);
                             It = Is + n;
                             u = positiveClip(It - Thalamus_h);
                             dV/dt = (-V + u)/Thalamus_tau; I""")
Thalamus_mot = zeros((1,n), """Is = I;
                             n = noise(Is,Thalamus_N);
                             It = Is + n;
                             u = positiveClip(It - Thalamus_h);
                             dV/dt = (-V + u)/Thalamus_tau; I""")

if options.indirectLoop:

    Striatum_ind_cog = zeros((n,1), """Is = I*(1 + 1.0*gamma_DAstrenght*DA);
                                     n = striatalNoise(Is,n);
                                     It = Is + n;
                                     Vh = strThreshold(DA);
                                     Vc = strSlope(DA);
                                     u = sigmoid(It - Striatum_h,Vmin,Vmax,Vh,Vc);
                                     dV/dt = (-V + u)/Striatum_tau; I; DA""")

    Striatum_ind_mot = zeros((1,n), """Is = I*(1 + 1.0*gamma_DAstrenght*DA);
                                     n = striatalNoise(Is,n);
                                     It = Is + n;
                                     Vh = strThreshold(DA);
                                     Vc = strSlope(DA);
                                     u = sigmoid(It - Striatum_h,Vmin,Vmax,Vh,Vc);
                                     dV/dt = (-V + u)/Striatum_tau; I; DA""")
    GPe_cog      = zeros((n,1), """Is = I;
                                 n = noise(Is,GPi_N);
                                 It = Is + n;
                                 u = positiveClip(It - GPe_h);
                                 dV/dt = (-V + u)/GPi_tau; I""")
    GPe_mot      = zeros((1,n), """Is = I;
                                 n = noise(Is,GPi_N);
                                 It = Is + n;
                                 u = positiveClip(It - GPe_h);
                                 dV/dt = (-V + u)/GPi_tau; I""")


cues_mot = np.arange(n)
cues_cog = np.arange(n)
labels = ['Cog'+str(i+1) for i in cues_cog]+['Mot'+str(i+1) for i in cues_mot]
if zeroValues:
    cog_cues_value = np.zeros(n)
else:
    cog_cues_value = np.ones(n) * 0.5
mot_cues_value = np.ones(n) * 0.5
if options.GM2008:
    cues_reward = np.array([0.5,0.4,0.3])
else:
    cues_reward = np.array([3.0,2.0,1.0,0.0])/3.0


# Connectivity
# -----------------------------------------------------------------------------
W = DenseConnection( Cortex_cog('V'),   Striatum_cog('I'), 1.0)
if len(cogInitialWeights) == 0:
    init_weights(W)
else:
    W._weights = cogInitialWeights
W_cortex_cog_to_striatum_cog = W

if doPrint:
    print "Cognitive weights: ", np.diag(W._weights)

W = DenseConnection( Cortex_mot('V'),   Striatum_mot('I'), 1.0)#0.9) #1.0)
if len(motInitialWeights) == 0:
    init_weights(W)
else:
    W._weights = motInitialWeights
W_cortex_mot_to_striatum_mot = W

W = DenseConnection( Cortex_ass('V'),   Striatum_ass('I'), 1.0)
init_weights(W)
W = DenseConnection( Cortex_cog('V'),   Striatum_ass('I'), np.ones((1,2*n+1)))
init_weights(W,0.2) #0.15)
W = DenseConnection( Cortex_mot('V'),   Striatum_ass('I'), np.ones((2*n+1,1)))
init_weights(W,0.2) #0.15)
DenseConnection( Cortex_cog('V'),   STN_cog('I'),       0.5 if options.indirectLoop else  1.0 )
DenseConnection( Cortex_mot('V'),   STN_mot('I'),       0.5 if options.indirectLoop else  1.0 )
DenseConnection( Striatum_cog('V'), GPi_cog('I'),      -2.5 if options.indirectLoop else -2.0 )
DenseConnection( Striatum_mot('V'), GPi_mot('I'),      -2.5 if options.indirectLoop else -2.0 )
DenseConnection( Striatum_ass('V'), GPi_cog('I'),    ( -2.5 if options.indirectLoop else -2.0 )*np.ones((1,2*n+1)))
DenseConnection( Striatum_ass('V'), GPi_mot('I'),    ( -2.5 if options.indirectLoop else -2.0 )*np.ones((2*n+1,1)))
DenseConnection( STN_cog('V'),      GPi_cog('I'),    (  2.0 if options.indirectLoop else  1.0 )*np.ones((2*n+1,1)))
DenseConnection( STN_mot('V'),      GPi_mot('I'),    (  2.0 if options.indirectLoop else  1.0 )*np.ones((1,2*n+1)))
DenseConnection( GPi_cog('V'),      Thalamus_cog('I'), -0.2 if options.indirectLoop else -0.5 )
DenseConnection( GPi_mot('V'),      Thalamus_mot('I'), -0.2 if options.indirectLoop else -0.5 )
DenseConnection( Thalamus_cog('V'), Cortex_cog('I'),    0.5 if options.indirectLoop else  1.0 )
DenseConnection( Thalamus_mot('V'), Cortex_mot('I'),    0.5 if options.indirectLoop else  1.0 )
DenseConnection( Cortex_cog('V'),   Thalamus_cog('I'),  0.4 )
DenseConnection( Cortex_mot('V'),   Thalamus_mot('I'),  0.4 )
DenseConnection( SNc_dop('V'),      Striatum_cog('DA'), 1.0 )
DenseConnection( SNc_dop('V'),      Striatum_mot('DA'), 1.0 )
DenseConnection( SNc_dop('V'),      Striatum_ass('DA'), 1.0 )

if options.striatostriatalConnections:
    DenseConnection( Striatum_cog('V'), Striatum_cog('I'), -0.2*np.ones((2*n+1,1))) #-1.5*np.ones((1,2*n+1)))
    DenseConnection( Striatum_mot('V'), Striatum_mot('I'), -0.2*np.ones((1,2*n+1))) #-1.5*np.ones((1,2*n+1)))
    if options.indirectLoop:
        DenseConnection( Striatum_ind_cog('V'), Striatum_ind_cog('I'), -0.2*np.ones((2*n+1,1))) #-1.5*np.ones((1,2*n+1)))            
        DenseConnection( Striatum_ind_mot('V'), Striatum_ind_mot('I'), -0.2*np.ones((1,2*n+1))) #-1.5*np.ones((1,2*n+1)))            

if options.indirectLoop:
    W = DenseConnection( Cortex_cog('V'),   Striatum_ind_cog('I'), 1.0)
    # if cogInitialWeights == None:
    init_weights(W)
    # else:
    #     W._weights = cogInitialWeights

    W = DenseConnection( Cortex_mot('V'),   Striatum_ind_mot('I'), 1.0)
    # if motInitialWeights != None:
    #     W._weights = motInitialWeights
    # else:
    init_weights(W)

    DenseConnection( Striatum_ind_cog('V'), GPe_cog('I'),      -2.0 )
    DenseConnection( Striatum_ind_mot('V'), GPe_mot('I'),      -2.0 )
    DenseConnection( STN_cog('V'),      GPe_cog('I'),           0.5*np.ones((2*n+1,1)))
    DenseConnection( STN_mot('V'),      GPe_mot('I'),           0.5*np.ones((1,2*n+1)))
    DenseConnection( GPe_cog('V'),      STN_cog('I'),          -0.5 ) # -1.0*np.ones((2*n+1,1)))
    DenseConnection( GPe_mot('V'),      STN_mot('I'),          -0.5 ) # -1.0*np.ones((1,2*n+1)))
    # DenseConnection( GPe_cog('V'),      GPi_cog('I'),          -1.0 )
    # DenseConnection( GPe_mot('V'),      GPi_mot('I'),          -1.0 )
    DenseConnection( SNc_dop('V'),      Striatum_ind_cog('DA'), 1.0 )
    DenseConnection( SNc_dop('V'),      Striatum_ind_mot('DA'), 1.0 )


# Initial DA levels
# -----------------------------------------------------------------------------
# Unnecessary now, DA comes from SNc
# setDAlevels(DA)

# Trial setup
# -----------------------------------------------------------------------------
inputCurrents_noise = np.zeros(3*n_availableOptions)
c1,c2,c3,m1,m2,m3 = 0,1,2,0,1,2
SNc_dop['SNc_h'] = SNc_h_base
wnoise = []

@clock.at(500*millisecond)
def set_trial(t):
    # global cues_cog, cogDecision, cogDecisionTime, motDecision, motDecisionTime, inputCurrents_noise, flag#, c1, c2, m1, m2
    # global daD, corticalLegend

    global cues_cog, cogDecision, cogDecisionTime, motDecision, motDecisionTime, c1, c2, m1, m2
    global inputCurrents_noise, flag, ctxLabels, wnoise

    if n_availableOptions == 2:
        if regPattern:
            index = int(currentTrial/nbTrialsPerPattern)
            if doPrint:
                print 'Current trial: ',currentTrial,'/',nbTrials,'. Then: index(cT/',nbTrialsPerPattern,') = ',index
            cues_cog = np.concatenate([pattern[index,:],pattern[index,:]])
        else:
            np.random.shuffle(cues_cog)
        np.random.shuffle(cues_mot)
        c1,c2 = cues_cog[:2]
        m1,m2 = cues_mot[:2]
    
    v = cues_amplitude
    Cortex_mot['Iext'] = 0
    Cortex_cog['Iext'] = 0
    Cortex_ass['Iext'] = 0
    
    cogDecision = False
    cogDecisionTime = 3500.0
    motDecision = False
    motDecisionTime = 3500.0

    if neuralPlot and n_availableOptions == 2:
        ctxLabels = labels[:]
        ctxLabels[c1] += ' *'
        ctxLabels[m1+n] += ' *'
        ctxLabels[c2] += ' ^'
        ctxLabels[m2+n] += ' ^'
        addLegend(axctx,lines_ctx,ctxLabels,loc='upper right')

    if Weights_N:
        W = W_cortex_cog_to_striatum_cog
        pwnoise = np.copy(wnoise)
        wnoise = np.random.normal(0,(np.diag(W.weights)-Wmin)*Weights_N)
        for w in range(n):
            if currentTrial > 0:
                noise = wnoise[w] - pwnoise[w]
            else:
                noise = wnoise[w]
            W.weights[w,w] = clip(W.weights[w,w] + noise, Wmin, Wmax)



@before(clock.tick)
def computeSoftInput(t):
    if motDecisionTime < 3500:
        inputsOffAt =  motDecisionTime/1000+reward_ext+0.2#delayReward+0.2
    else:
        inputsOffAt = 3.2
    v = sigmoid(t,0,cues_amplitude,.725,.042) - sigmoid(t,0,cues_amplitude,inputsOffAt,.084)

    inputCurrents_noise = np.random.normal(0,v*Cortex_N,3*n_availableOptions)
    for i in range(n_availableOptions):
        Cortex_mot['Iext'][0          ,cues_mot[i]] = v + inputCurrents_noise[i*3]
        Cortex_cog['Iext'][cues_cog[i],0          ] = v + inputCurrents_noise[i*3+1]
        Cortex_ass['Iext'][cues_cog[i],cues_mot[i]] = v + inputCurrents_noise[i*3+2]

@clock.at(3000*millisecond)
def unset_trial(t):
    Cortex_mot['Iext'] = 0
    Cortex_cog['Iext'] = 0
    Cortex_ass['Iext'] = 0

## @clock.at(duration-dt)
## def storeResults(t):
##     if completeTrials and STORE_DATA and motDecision:
##         printData()

@before(clock.tick)
def convolvD2(t):
    baseActivity = -SNc_dop['SNc_h']
    relativeFiringRate = SNc_dop['V_lag']-baseActivity
    SNc_dop['DAtoD2c'] = positiveClip(convolv_D2Kernel(t,SNc_dop['DAtoD2c'],relativeFiringRate))

@before(clock.tick)
def fillDopamineBuffer(t):
    global DA_buffIndex
    buffPos = DA_buffIndex % DA_buffSize
    DA_buff[buffPos] = SNc_dop['V']
    SNc_dop['V_lag'] = DA_buff[(buffPos+1) % DA_buffSize]
    DA_buffIndex += 1

@before(clock.tick)
def deliverReward(t):
    if (motDecisionTime/1000 + delayReward) < t:
        if t < (motDecisionTime/1000 + reward_ext) and SNc_dop['Irew'] == 0:
            SNc_dop['Irew'] = pError * alpha_Rew_DA
        elif t >= (motDecisionTime/1000 + reward_ext):
            SNc_dop['Irew'] = 0

    if cogReward and cogDecisionTime/1000 < t:
        if t < (cogDecisionTime/1000 + reward_ext - delayReward) and SNc_dop['Ie_rew'] == 0:
            # print 'Value for current:',cog_cues_value[np.argmax(Cortex_cog['V'])],float(cog_cues_value[np.argmax(Cortex_cog['V'])]),float(cog_cues_value[np.argmax(Cortex_cog['V'])])*alpha_Rew_DA
            SNc_dop['Ie_rew'] = float(cog_cues_value[np.argmax(Cortex_cog['V'])]) * alpha_Rew_DA
        elif t >= (cogDecisionTime/1000 + reward_ext - delayReward):
            SNc_dop['Ie_rew'] = 0

# @after(clock.tick)
# def resetReward(t):
#     if motDecisionTime/1000 + reward_ext < t: # Comparison between [ms] 
#         SNc_dop['Irew'] = 0
# 
#     if cogReward and cogDecisionTime/1000 + reward_ext - delayReward < t: # Comparison between [ms] 
#         SNc_dop['Ie_rew'] = 0

# @before(clock.tick)
# def propagateDA(t): # This should be implemented as a connection! but for quick testing...
#     setDAlevels(SNc_dop['V'])

@after(clock.tick)
def earlyStop(t):
    global currentTrial, continuousFailedTrials
    if stopAfterSelection and motDecisionTime < 3500 and cogDecisionTime < 3500:
        if t > motDecisionTime / 1000 + afterSelectionTime:
            if neuralPlot:
                resetPlot()
                if storePlotsEvery > 0 and (currentTrial % storePlotsEvery) == 0:
                    setXlim_d(t)
                    storePlots()
            if STORE_DATA:
                printData()
            partialReset()
            clock.reset()
            currentTrial += 1
            continuousFailedTrials = 0

# Learning
# -----------------------------------------------------------------------------
P, R, A, Regret = [], [], [], []
cumRegret = 0
smoothR = 0
smoothA = 0

cogDecision = False
motDecision = False
cogDecisionTime = 3500.0
motDecisionTime = 3500.0
pError = 0.0
choice = -1
mchoice = -1
pastW = np.copy(W_cortex_cog_to_striatum_cog._weights)
nW = np.copy(np.diag(pastW))
cogWDiff = np.zeros([n,4])
trialLtdLtp = np.zeros([n,4])

# LTP-LTD in corticostriatal connections
@after(clock.tick)
def corticostriatal_learning(t):
    if learn:
        # if choice < 0:
        #     return
        W = W_cortex_cog_to_striatum_cog
        dDA = SNc_dop['V']-(-SNc_dop['SNc_h'])  # current DA activity w.r.t. base level
        if (constantLTD or oneStepConstantLTD) and dDA < 0:
            return
        if SNc_dop['Ir'] > 0 and dDA > 0: # LTP
            alpha = alpha_LTP
        elif SNc_dop['Ir'] < 0: # LTD
            alpha = alpha_LTD
        else:
            return

        for c in N:
            dw = dDA * alpha * Striatum_cog['V'][c][0] / ((reward_ext - delayReward)/millisecond)
            w = clip(W.weights[c, c] + dw, Wmin, Wmax)
            trialLtdLtp[c,2] += dw
            trialLtdLtp[c,3] += w - W.weights[c, c]
            W.weights[c,c] = w

        if applyInMotorLoop:
            W = W_cortex_mot_to_striatum_mot
            for c in N:
                dw = dDA * alpha * Striatum_mot['V'][0][c] / ((reward_ext - delayReward)/millisecond)
                w = clip(W.weights[c, c] + dw, Wmin, Wmax)
                W.weights[c,c] = w


@before(clock.tick)
def corticostriatalLTD(t):
    if learn and constantLTD:
        W = W_cortex_cog_to_striatum_cog
        for pop in N:
            # print pop,n
            # LTD is produced by the combined activation of postsynaptic mGluR and presynaptic endocannabinoid receptors (produced at the postynaptic neurons)
            # The endocannabinoid release is enhanced by D2 receptors -> DA promotes LTD

            # -> mGluR -> Striatum_cog['It'] (activation of receptors from cortical inputs, after DA enhancing effect)
            # -> eCB1  -> CB release is produced by voltage dependent calcium channels (so dependent in membrane potential, assumed directly related with firing rate)
            #             and enhanced by D2 (SNc_dop['V']). 
            # As Shen et al. (2008) shown that D2 sped LTD but not blocked it, SNc_dop['V'] is introduced as a weighted additive term.
            # if pop == 1:
            #     print "Glu:",gamma_mGluR_LTD * Striatum_cog['It'][pop][0],"  CB:",gamma_eCB1_LTD * Striatum_cog['V'][pop][0],"  DA:",(0.1 + gamma_DA_LTD * SNc_dop['V']), "   factor:",(gamma_mGluR_LTD * Striatum_cog['It'][pop][0]) * (gamma_eCB1_LTD * Striatum_cog['V'][pop][0]) * (0.1 + gamma_DA_LTD * SNc_dop['V'])


            # if pop == 1:
            #     print "Glu:",gamma_mGluR_LTD * Striatum_cog['It'][pop][0],"  CB1_release:",(gamma_eCB1_LTD * (Striatum_cog['V'][pop][0] + gamma_DA_LTD * SNc_dop['V'])),"CB1_dopa:",gamma_eCB1_LTD*gamma_DA_LTD * SNc_dop['V']

            # W.weights[pop][pop] -= 0.05 * alpha_LTD * (gamma_mGluR_LTD * Striatum_cog['It'][pop][0]) * (gamma_eCB1_LTD * (Striatum_cog['V'][pop][0] + gamma_DA_LTD * SNc_dop['V']))
            dw = - 0.001 * alpha_LTD * Striatum_cog['It'][pop][0] * (1 + gamma_DA_LTD * SNc_dop['V'])
            w = clip(W.weights[pop][pop] + dw,Wmin,Wmax)
            trialLtdLtp[pop,0] += dw
            trialLtdLtp[pop,1] += w - W.weights[pop, pop]
            W.weights[pop][pop] = w

        if applyInMotorLoop:
            W = W_cortex_mot_to_striatum_mot
            for pop in N:
                dw = - 0.001 * alpha_LTD * Striatum_mot['It'][0][pop] * (1 + gamma_DA_LTD * SNc_dop['V'])
                w = clip(W.weights[pop][pop] + dw,Wmin,Wmax)
                W.weights[pop][pop] = w

#@after(clock.tick)
@clock.every(50*millisecond)
def register(t):
    global currentTrial, continuousFailedTrials, selectionError, cogDecision
    global cogDecisionTime, motDecision, motDecisionTime
    global choice, nchoice, mchoice, pError, smoothR, smoothA, pastW, cog_cues_value
    global cumRegret, perceived_advantage
    
    if not cogDecision:
        U = np.sort(Cortex_cog['V'].ravel())
        if abs(U[-1] - U[-2]) >= decision_threshold and t <= duration-500*millisecond and t >= 500*millisecond:
            cogDecisionTime = 1000.0*clock.time
            cogDecision = True
            if cogReward:
                SNc_dop['Ie_rew'] = float(cog_cues_value[np.argmax(Cortex_cog['V'])]) * alpha_Rew_DA

    U = np.sort(Cortex_mot['V']).ravel()

    # No motor decision yet
    if motDecision or abs(U[-1] - U[-2]) < decision_threshold or t > duration-500*millisecond or t < 500*millisecond: 
        return

    motDecision = True
    motDecisionTime = 1000.0*clock.time

    # The motor selection is the executed one,
    # defining the selected cue.
    # The cognitive selection might differ.
    

    mot_choice = np.argmax(Cortex_mot['V'])
    cog_choice = np.argmax(Cortex_cog['V'])

    if options.GM2008:
        choice = mot_choice
        mchoice = mot_choice

        if choice == np.argmax(cues_reward):
            P.append(1.0)
        else:
            P.append(0.0)

    else:
        c1, c2 = cues_cog[:2]
        m1, m2 = cues_mot[:2]
        if mot_choice == m1:
            choice = c1
            nchoice = c2
            mchoice = m1
        elif mot_choice == m2:
            choice = c2
            nchoice = c1
            mchoice = m2
        else:
            if doPrint:
                print "! Failed trial: selected a non-presented cue (not ",m1,"nor",m2,"but",mot_choice,")"
            reset()
            selectionError += 1
            if not forceSelection:
                currentTrial += 1
                continuousFailedTrials = 0
            end()
            return

        if cues_reward[choice] > cues_reward[nchoice]:
            P.append(1.0)
        else:
            P.append(0.0)

    
    # actualizing choice statistics:
    if options.GM2008:
        for c in cues_cog:
            if c == choice:
                probChoice[c][currentTrial%probChoiceBuffSize] = 1
            else:
                probChoice[c][currentTrial%probChoiceBuffSize] = 0
    else:
        sPair = [(choice < nchoice) * choice + (nchoice < choice) * nchoice ,
                 (choice < nchoice) * nchoice + (nchoice < choice) * choice ]
        pairPos = 0
        for i in range(nPatterns):
            if pattern[i][0]==sPair[0] and pattern[i][1]==sPair[1]:
                pairPos = i

        probChoice[pairPos][currentTrial%probChoiceBuffSize] = choice<nchoice

    
    # Compute reward
    reward = np.random.uniform(0,1) < cues_reward[choice]
    R.append(reward)

    if options.GM2008:
        # How good was the selection compared to any choice
        regret = np.max(cues_reward) - cues_reward[choice]
        perceived_regret = np.max(cog_cues_value) - reward # cog_cues_value[choice]
        relative_regret = np.mean(cog_cues_value) - reward 
    else:
        # How good was the selection compared to the best choice presented
        regret = np.max([cues_reward[choice],cues_reward[nchoice]]) - cues_reward[choice]
        perceived_regret = np.max([cog_cues_value[choice],cog_cues_value[nchoice]]) - reward # cog_cues_value[choice]
        relative_regret = np.mean([cog_cues_value[choice],cog_cues_value[nchoice]]) - reward # cog_cues_value[choice]

    advantage = 1 - regret
    perceived_advantage = 1 - perceived_regret
    A.append(advantage)
    Regret.append(regret)
    cumRegret += regret

    if smoothR == -1: # first trial
        smoothR = reward
        smoothA = perceived_advantage if options.usePerception else advantage
    else:
        smoothR = alpha_SuccessEMA * smoothR + (1-alpha_SuccessEMA) * reward
        smoothA = alpha_SuccessEMA * smoothA + (1-alpha_SuccessEMA) * (perceived_advantage if options.usePerception else advantage)

    W = W_cortex_cog_to_striatum_cog
    Wm = W_cortex_mot_to_striatum_mot
    if learn:
        # Compute prediction error
        pError = reward - cog_cues_value[choice]
        # Update cues values
        cog_cues_value[choice] += pError* alpha_c

        ## Error regulates PPNg activity, the prediction error encoding signal
        if relativeValue:
            if options.GM2008:
                pError = cog_cues_value[choice] - np.mean(cog_cues_value)
            else:
                pError = cog_cues_value[choice] - (cog_cues_value[choice]+cog_cues_value[nchoice])/2.0

        # Constant depression over values
        # cog_cues_value = clip(cog_cues_value - alpha_c /20.0,0,1)

        if oneStepConstantLTD:
            W_cortex_cog_to_striatum_cog._weights -= alpha_LTD

    if doPrint:
        # Just for displaying ordered cue
        oc1,oc2 = min(c1,c2), max(c1,c2)
        if P[-1]:
            if options.GM2008:
                print "Choice:            [%d]  (good)" % (choice)
            else:
                print "Choice:            [%d] / %d  (good)" % (choice,nchoice)
        else:
            if options.GM2008:
                print "Choice:              %d   (bad)" % (choice)
                print "Choice:              %d / [%d] (bad)" % (nchoice,choice)
        if learn:
            print "Reward (%3d%%) :   %d / %f" % (int(100*cues_reward[choice]),reward, smoothR)
            print "SNc_h :           ",SNc_dop['SNc_h']
            print "Mean performance: ",np.array(P[-perfBuff:]).mean()
            print "Mean reward:       %.3f" % np.array(R).mean()
            print "Response time:     %d ms" % motDecisionTime
            print "Cognitive weights:", np.diag(W._weights)
            print "Motor weights:", np.diag(Wm._weights)
            print "Cognitive values: ", cog_cues_value
            # print "Motor weights:     ", np.diag(Wm._weights)

    # if STORE_DATA and not completeTrials:
    #     printData()
# 
#     # if completeTrials:
#     #     return
# 
#     # # In case that there is no interest in continuing this trial, 'completeTrials == False',
#     # # the model is set to its initial state and the clock is reset, continuing with a new
#     # # trial.
#     # currentTrial += 1
#     # continuousFailedTrials = 0
#     # partialReset()
    # clock.reset()

if neuralPlot:
    cogMarker = motMarker = movMarker = []
    plotSteps = 1*millisecond

    ### Neural activity

    axctx = axctxass = axstr = axstrass = axgpi = axstn = axth = axsnc = axgpe = axstr_in = None


    if options.indirectLoop:
        fig, ((axctx,axstn),(axstr,axstr_in),(axgpe,axgpi),(axsnc,axth)) = plt.subplots(4,2,figsize=(20,8),num="DA: "+str(DA)+" "+'% '.join(str(r) for r in cues_reward)+'% ')
    else:
        fig, ((axctx,axstn),(axstr,axgpi),(axsnc,axth)) = plt.subplots(3,2,figsize=(20,6),num="DA: "+str(DA)+" "+'% '.join(str(r) for r in cues_reward)+'% ')
    fig_ass, (axctxinputs, axctxass,axstrass) = plt.subplots(3,1,figsize=(10,6),num="ASS - DA: "+str(DA)+" "+'% '.join(str(r) for r in cues_reward)+'% ')#+" X_"+str(aux_X)+" Y_"+str(aux_Y))

    xBySteps = np.arange(0,duration+plotSteps,plotSteps)
    xLen = len(xBySteps)

    axctx.set_ylim(-2,70)
    axctx.set_title('Cortical activity', fontsize=10)
    neuralData_ctx = np.full((xLen,2*n),None,dtype=float)
    lines_ctx = axctx.plot(xBySteps,neuralData_ctx, alpha=0.7)

    axctxinputs.set_ylim(-2,70)
    axctxinputs.set_title('Cortical inputs from sensory populations', fontsize=10)
    neuralData_ctxinputs = np.full((xLen,2*n),None,dtype=float)
    lines_ctxinputs = axctxinputs.plot(xBySteps,neuralData_ctxinputs, alpha=0.7)

    axctxass.set_ylim(-2,40)
    axctxass.set_title('Associative cortical activity', fontsize=10)
    neuralData_ctx_ass = np.full((xLen,n*n),None,dtype=float)
    lines_ctx_ass = axctxass.plot(xBySteps,neuralData_ctx_ass, alpha=0.7)

    axstr.set_ylim(-2,23)
    axstr.set_title('Striatal (direct) activity', fontsize=10)
    neuralData_str = np.full((xLen,2*n),None,dtype=float)
    lines_str = axstr.plot(xBySteps,neuralData_str, alpha=0.7)

    if options.indirectLoop:
        axstr_in.set_ylim(-2,18)
        axstr_in.set_title('Striatal (indirect) activity', fontsize=10)
        neuralData_str_in = np.full((xLen,2*n),None,dtype=float)
        lines_str_in = axstr_in.plot(xBySteps,neuralData_str_in, alpha=0.7)

        axgpe.set_ylim(-2,40)
        axgpe.set_title('GPe activity', fontsize=10)
        neuralData_gpe = np.full((xLen,2*n),None,dtype=float)
        lines_gpe = axgpe.plot(xBySteps,neuralData_gpe, alpha=0.7)

    axstrass.set_ylim(-2,18)
    axstrass.set_title('Associative striatal activity', fontsize=10)
    neuralData_str_ass = np.full((xLen,n*n),None,dtype=float)
    lines_str_ass = axstrass.plot(xBySteps,neuralData_str_ass, alpha=0.7)

    axsnc.set_ylim(-10,20)
    axsnc.set_title('SNc and PPTN activity', fontsize=10)
    neuralData_snc = np.full((xLen,2),None,dtype=float)
    lines_snc = axsnc.plot(xBySteps,neuralData_snc, alpha=0.7)

    axgpi.set_ylim(40,130)
    axgpi.set_title('GPi activity', fontsize=10)
    neuralData_gpi = np.full((xLen,2*n),None,dtype=float)
    lines_gpi = axgpi.plot(xBySteps,neuralData_gpi, alpha=0.7)

    axstn.set_ylim(10,80)
    axstn.set_title('STN activity', fontsize=10)
    neuralData_stn = np.full((xLen,2*n),None,dtype=float)
    lines_stn = axstn.plot(xBySteps,neuralData_stn, alpha=0.7)

    axth.set_ylim(-2,45)
    axth.set_title('Thalamus activity', fontsize=10)
    neuralData_th = np.full((xLen,2*n),None,dtype=float)
    lines_th = axth.plot(xBySteps,neuralData_th, alpha=0.7)

    # Legends

    # corticalLegend = axctx.legend(flip(lines_ctx,n),flip(labels,n),loc='upper right', ncol=n, fontsize='x-small',framealpha=0.6, # bbox_to_anchor= (1.08, 0.5), #ncol=2, # bbox_to_anchor= (1.2, 0.5), 
    #         borderaxespad=0, frameon=False)
# 
    # axstrass.legend(flip(lines_ctx_ass,n),flip(['Ass'+str(i+1)+'_'+str(j+1) for j in N for i in N],n),loc='upper right', fontsize='x-small', #bbox_to_anchor= (1.1, 0.5), 
    #     borderaxespad=0, frameon=False, ncol=n)
# 
#     # axsnc.legend(flip(lines_snc,2),flip(['SNc','PPTN'],2),loc='upper right', ncol=1, fontsize='x-small',framealpha=0.6, # bbox_to_anchor= (1.08, 0.5), # (1.12, 0.5), 
#     #         borderaxespad=0, frameon=False)
# 
    ### Behavioral outcomes and internal ponderations

    axv = axw = axsnct = axper = axprob = axentr = None

    fig2, ((axper,axsnct),(axentr,axprob),(axw,axv)) = plt.subplots(3,2,figsize=(20,6),num="[Behavior] DA: "+str(DA)+" "+'% '.join(str(r) for r in cues_reward)+'% ')#+" X_"+str(aux_X)+" Y_"+str(aux_Y))

    xByTrials = np.arange(nbTrials)
    xLen = len(xByTrials)

    axper.set_ylim(0,1)
    axper.set_title('Behavioral outcomes', fontsize=10)
    neuralData_per = np.full((xLen,3),None,dtype=float)
    lines_per = axper.plot(xByTrials,neuralData_per, alpha=0.7)

    axv.set_ylim(0,1)
    axv.set_title('Learned cognitive values', fontsize=10)
    neuralData_v = np.full((xLen,n+2),None,dtype=float)
    lines_v = axv.plot(xByTrials,neuralData_v, alpha=0.7)

    axw.set_ylim(0.39,.61)
    axw.set_title('Cognitive corticostriatal weights', fontsize=10)
    neuralData_w = np.full((xLen,n),None,dtype=float)
    lines_w = axw.plot(xByTrials,neuralData_w, alpha=0.7)

    axsnct.set_ylim(0,20)
    axsnct.set_title('SNc tonic activity', fontsize=10)
    neuralData_snct = np.full((xLen,1),None,dtype=float)
    lines_snct = axsnct.plot(xByTrials,neuralData_snct, alpha=0.7,color='magenta',linewidth=3)

    axprob.set_ylim(0,1.1)
    axprob.set_title('Choice probability', fontsize=10)
    neuralData_prob = np.full((xLen,nPatterns),None,dtype=float)
    lines_prob = axprob.plot(xByTrials,neuralData_prob, alpha=0.7, linewidth=3)

    axentr.set_ylim(0,1)
    axentr.set_title('Choice entropy', fontsize=10)
    neuralData_entr = np.full((xLen,1),None,dtype=float)
    lines_entr = axentr.plot(xByTrials,neuralData_entr, alpha=0.7)


    # Legends


    def addLegend(axs,signals,labels=labels,n=n,doflip=True,loc='upper right'):
        if doflip:
            axs.legend(flip(signals,n),flip(labels,n),loc=loc, ncol=n, fontsize='x-small',
                borderaxespad=0, framealpha=0.4) #frameon=False)
        else:
            axs.legend(signals,loc=loc, ncol=n, fontsize='x-small',borderaxespad=0, framealpha=0.4) #frameon=False)

    # axsnct.legend(flip(lines_snct,n),flip(['SNc'],n),loc='upper right', ncol=n, fontsize='x-small', # bbox_to_anchor= (1.08, 0.1), # (1.18, 0.5), 
    #         borderaxespad=0, frameon=False)
# 
#     # axw.legend(flip(lines_w,n),flip(['Wcog['+str(i)+']' for i in cues_cog],n),loc='upper right', ncol=n, fontsize='x-small',framealpha=0.6, # bbox_to_anchor= (1.08, 0.5), #(1.14, 0.5), 
#     #         borderaxespad=0, frameon=False)
#     # 
#     # axv.legend(flip(lines_v,n),flip(['Vcog['+str(i)+']' for i in cues_cog]+['Selection']+['Not selected'],n),loc='upper right', ncol=n, fontsize='x-small',framealpha=0.6, # bbox_to_anchor= (1.08, 0.5), # bbox_to_anchor= (1.15, 0.5), 
    #         borderaxespad=0, frameon=False)


    addLegend(axper,lines_per,['Performance','Advantage','Regret'],loc='upper left')
    if not options.GM2008:
        addLegend(axprob,lines_prob,[str(x[0])+' - '+str(x[1]) for x in pattern])
    else:
        addLegend(axprob,lines_prob,[str(x) for x in pattern])
    addLegend(axctxass,lines_ctx_ass,['ASS-'+str(i+1)+'_'+str(j+1) for j in range(n) for i in range(n)])
    addLegend(axsnc,lines_snc,['SNc','PPTN'],n=3)
    addLegend(axw,lines_w,['Wcog['+str(i)+']' for i in cues_cog])
    addLegend(axv,lines_v,['Vcog['+str(i)+']' for i in cues_cog]+['Selection']+['Not selected'])

    ### Linestyles

    assMarkers = ['1', '2', '3', '4']

    for l in N:
        plt.setp(lines_ctx[l+n],'color',plt.getp(lines_ctx[l],'color'),'ls','--')
        plt.setp(lines_ctxinputs[l+n],'color',plt.getp(lines_ctx[l],'color'),'ls','--')
        plt.setp(lines_str[l+n],'color',plt.getp(lines_str[l],'color'),'ls','--')
        plt.setp(lines_gpi[l+n],'color',plt.getp(lines_str[l],'color'),'ls','--')
        plt.setp(lines_stn[l+n],'color',plt.getp(lines_str[l],'color'),'ls','--')
        plt.setp(lines_th[l+n],'color',plt.getp(lines_str[l],'color'),'ls','--')
        plt.setp(lines_str_ass[l*n+l],'color',plt.getp(lines_str[l],'color'),'ls','--')
        plt.setp(lines_ctx_ass[l*n+l],'color',plt.getp(lines_str[l],'color'),'ls','--')
        for l2 in N:
            if l == l2:
                continue
            plt.setp(lines_str_ass[l+l2*n],'color',plt.getp(lines_str[l],'color'),'marker',assMarkers[l2],'markersize',12)
            plt.setp(lines_ctx_ass[l+l2*n],'color',plt.getp(lines_str[l],'color'),'marker',assMarkers[l2],'markersize',12)

        if options.indirectLoop:
            plt.setp(lines_str_in[l+n],'color',plt.getp(lines_str[l],'color'),'ls','--')
            plt.setp(lines_gpe[l+n],'color',plt.getp(lines_str[l],'color'),'ls','--')

    plt.setp(lines_v[n:], 'marker', 'D','linestyle','','alpha',0.2,'markersize',5)
    plt.setp(lines_v[n], 'color','magenta')


    # Setting X limits

    axd = [axstr,axctx, axctxinputs,axctxass,axsnc,axstrass,axgpi,axstn,axth]+([axstr_in, axgpe] if options.indirectLoop else [])
    axt = [axsnct,axw,axv,axprob,axper,axentr]

    def setXlim_d(t=duration):
        for axis in axd:
            axis.set_xlim(0,t)
    setXlim_d()
    for axis in axt:
        axis.set_xlim(0,nbTrials)
    for axis in axd+axt:
        axis.tick_params(axis='both', which='major', labelsize=10)

    fig.tight_layout()
    fig2.tight_layout()
    fig_ass.tight_layout()

    def storePlots():
        fig.savefig(plotname+'_neuralActivity_'+str(currentTrial)+".pdf",bbox_inches='tight')
        fig.savefig(plotname+'_neuralActivity_'+str(currentTrial)+".png",bbox_inches='tight')
        fig2.savefig(plotname+'_behavioralOutcomes_'+str(currentTrial)+".pdf",bbox_inches='tight')
        fig2.savefig(plotname+'_behavioralOutcomes_'+str(currentTrial)+".png",bbox_inches='tight')

    def drawPlots():
        fig.canvas.draw()
        fig.canvas.flush_events()

    @clock.every(plotSteps)
    def plotDecisionMarkers(t):
        global cogMarker, motMarker, movMarker, ctxLabels
        reloadLabel = False
        if len(cogMarker) == 0 and cogDecisionTime/1000 < t:
            reloadLabel = True
            cogMarker = axctx.plot(t,40,marker='^',alpha=0.4,markersize=10,color='r',linestyle='None')
        if len(motMarker) == 0 and motDecisionTime/1000 < t:
            reloadLabel = True
            motMarker = axctx.plot(t,40,marker='^',alpha=0.4,markersize=10,color='b',linestyle='None')
            neuralData_v[currentTrial][n] = float((n-1)-choice)/float(n-1)
            if not options.GM2008:
                neuralData_v[currentTrial][n+1] = float((n-1)-nchoice)/float(n-1)
            for i in np.arange(2)+n:
                lines_v[i].set_ydata(neuralData_v[:,i])
        if len(movMarker) == 0 and (motDecisionTime/1000 + delayReward) < t:
            reloadLabel = True
            if R[-1]:
                movMarker = axctx.plot(t,40,marker='.',alpha=0.4,markersize=20,color='g',linestyle='None')
            else:
                movMarker = axctx.plot(t,40,marker='x',alpha=0.4,markersize=10,color='r',linestyle='None')

        if reloadLabel:
            addLegend(axctx,lines_ctx + \
                (len(cogMarker) != 0) * cogMarker + \
                (len(motMarker) != 0) * motMarker + \
                (len(movMarker) != 0) * movMarker, 
                ctxLabels + (len(cogMarker) != 0)*['C-choise'] +
                (len(motMarker) != 0) * ['M-choise'] +
                (len(movMarker) != 0) * ['Reward'], loc='upper right')

            if len(motMarker) != 0:
                actualizePerformance()


    def actualizePerformance():
        neuralData_per[currentTrial]   = [np.array(P[-perfBuff:]).mean(),
                                          np.array(A[-perfBuff:]).mean(),
                                          np.array(Regret[-perfBuff:]).mean()]

        for i,line in enumerate(lines_per):
            line.set_ydata(neuralData_per[:,i])

    @clock.at(500*millisecond)
    def addCuesLines(t):
        axctx.plot([t, t], axctx.get_ylim(),'gray',ls='--',alpha=.2,lw=2)

    @clock.at(dt)
    def plotTrial_data(t):
        neuralData_snct[currentTrial]  = SNc_dop['SNc_h']
        neuralData_w[currentTrial]     = np.diag(W_cortex_cog_to_striatum_cog._weights)
        neuralData_v[currentTrial][:n] = cog_cues_value
        neuralData_prob[currentTrial]  = np.nanmean(probChoice,axis=1)
        neuralData_entr[currentTrial] = np.sum([- prob * np.log2(prob) for prob in neuralData_prob[currentTrial]]) # / len(neuralData_prob[currentTrial])

        # if currentTrial > 0:
        #     neuralData_per[currentTrial-1]   = [np.array(P[-perfBuff:]).mean(),
        #                                       np.array(A[-perfBuff:]).mean(),
        #                                       np.array(Regret[-perfBuff:]).mean()]
# 
#         #     for i,line in enumerate(lines_per):
        #         line.set_ydata(neuralData_per[:,i])

        for i,line in enumerate(lines_w):
            line.set_ydata(neuralData_w[:,i])
        for i,line in enumerate(lines_v):
            line.set_ydata(neuralData_v[:,i])
        for i,line in enumerate(lines_prob):
            line.set_ydata(neuralData_prob[:,i])
        for i,line in enumerate(lines_entr):
            line.set_ydata(neuralData_entr[:,i])
        for i,line in enumerate(lines_snct):
            line.set_ydata(neuralData_snct[:,i])
    
    @clock.every(plotSteps)
    def plotNeural_data(t):
        if flashPlots>0 and (currentTrial % flashPlots) != 0:
            return
        index = int(round(t/plotSteps))

        # Neural activity
        neuralData_str[index] = np.hstack(([Striatum_cog['V'][i][0] for i in N],Striatum_mot['V'][0][:]))
        neuralData_ctx[index] = np.hstack(([Cortex_cog['V'][i][0] for i in N],Cortex_mot['V'][0][:]))
        neuralData_ctxinputs[index] = np.hstack(([Cortex_cog['Iext'][i][0] for i in N],Cortex_mot['Iext'][0][:]))
        neuralData_str_ass[index] = [Striatum_ass['V'][i][j] for j in N for i in N]
        neuralData_ctx_ass[index] = [Cortex_ass['V'][i][j] for j in N for i in N]
        neuralData_gpi[index] = np.hstack(([GPi_cog['V'][i][0] for i in N],GPi_mot['V'][0][:]))
        neuralData_stn[index] = np.hstack(([STN_cog['V'][i][0] for i in N],STN_mot['V'][0][:]))
        neuralData_th[index]  = np.hstack(([Thalamus_cog['V'][i][0] for i in N],Thalamus_mot['V'][0][:]))
        neuralData_snc[index] = [SNc_dop['V'][0],SNc_dop['Ir'][0]]

        for i,line in enumerate(lines_str):
            line.set_ydata(neuralData_str[:,i])
        for i,line in enumerate(lines_ctx):
            line.set_ydata(neuralData_ctx[:,i])
        for i,line in enumerate(lines_ctxinputs):
            line.set_ydata(neuralData_ctxinputs[:,i])
        for i,line in enumerate(lines_ctx_ass):
            line.set_ydata(neuralData_ctx_ass[:,i])
        for i,line in enumerate(lines_snc):
            line.set_ydata(neuralData_snc[:,i])
        for i,line in enumerate(lines_str_ass):
            line.set_ydata(neuralData_str_ass[:,i])
        for i,line in enumerate(lines_gpi):
            line.set_ydata(neuralData_gpi[:,i])
        for i,line in enumerate(lines_stn):
            line.set_ydata(neuralData_stn[:,i])
        for i,line in enumerate(lines_th):
            line.set_ydata(neuralData_th[:,i])


        if options.indirectLoop:
            neuralData_str_in[index] = np.hstack(([Striatum_ind_cog['V'][i][0] for i in N],Striatum_ind_mot['V'][0][:]))
            neuralData_gpe[index] = np.hstack(([GPe_cog['V'][i][0] for i in N],GPe_mot['V'][0][:]))

            for i,line in enumerate(lines_str_in):
                line.set_ydata(neuralData_str_in[:,i])
            for i,line in enumerate(lines_gpe):
                line.set_ydata(neuralData_gpe[:,i])

        if not flashPlots:
            drawPlots()


# Simulation
# -----------------------------------------------------------------------------

currentTrial = 0
failedTrials = 0
selectionError = 0
continuousFailedTrials = 0
reset()

while currentTrial < nbTrials:
    if doPrint:
        print "Trial: ", currentTrial,". Failed trials: ",failedTrials

    partialReset()

    if len(invertAt) and currentTrial in invertAt:
        if invertAt.index(currentTrial) == 0:
          cues_reward[2] = 0.9
        else:
          cues_reward[2] = 0.3
    run(time=duration, dt=dt)

    if clock.time >= duration:
        # If the 'completeTrial' flag is 'True' or there was no selection
        # during the experiment, the current trial's lenght is 'duration' seconds

        # The model is set to its initial state
        # reset()
        if neuralPlot:
            if storePlotsEvery > 0 and (currentTrial % storePlotsEvery) == 0:
                storePlots()
            if currentTrial < nbTrials-1:
                resetPlot()
        if not motDecision:
            # Detection and counting of failed trials
            if doPrint:
                print "Failed trial!! ",cues_cog[:2]
            failedTrials += 1
            continuousFailedTrials += 1
        elif STORE_DATA:
            printData()

        if motDecision or not forceSelection:
            currentTrial += 1
            continuousFailedTrials = 0
    if doPrint:
        print
    if forceSelection and (failedTrials >= nbFailedTrials or 
        continuousFailedTrials >= nbContFailedTrials):
        break

if STORE_FAILED and FAILED_FILE:
    with open(FAILED_FILE,'a') as f:
        f.write("%d\n" % failedTrials)
elif STORE_FAILED:
    with open(filename,'a') as f:
        if STORE_DATA:
            if applyInMotorLoop:
                st =  '\t'.join(['{:.0f}'.format(-1) for i in range(2*n)])
            else:
                st = '\t'.join(['{:.0f}'.format(-1) for i in N])
            f.write("%d\t%d\t%d\t%d\t%d\t%d\t%s\n" % (currentTrial,failedTrials,selectionError,-1,-1,-1,
                st))
        else:
            f.write("%d\t%d\t%d\n" % (currentTrial,failedTrials,selectionError))

if STORE_DATA and forceSelection and \
   (failedTrials >= nbFailedTrials or continuousFailedTrials >= nbContFailedTrials):
    filename += '_error'
    with open(filename,'a') as f:
        f.write("\n")

if neuralPlot:
    plt.ioff()
    plt.show(block=False)
