#!/usr/bin/env python
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
# python learning.py --ltd-constant --relativeValue -P --flashPlots 1 --debug
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

from datetime import datetime       # Added to create a store file with time data
from socket import gethostname      # Added to specify different output folders
                                    # deppending on the computer where the code is
                                    # running
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
stopAfterSelecion = False

# Default Time resolution
dt = 1.0*millisecond

# Cortical inputs amplitude [sp/s]
cues_amplitude = 22 if garivierMoulines else 16

# Thresholds
Cortex_h   =  -3.0
Striatum_h =   0.0
STN_h      = -10.0
GPi_h      =  10.0
Thalamus_h = -40.0
SNc_h_base =  -DA

# Time constants
Cortex_tau   = 0.01
Striatum_tau = 0.01
STN_tau      = 0.01
GPi_tau      = 0.01
Thalamus_tau = 0.01
STR_N_tau    = 0.035
SNc_tau   = 0.01
SNc_N_tau    = 0.03
arD2_tau = .5

# DA D2-autoreceptors
# D2-Autoreceptors delay [ms]
arD2_lag    = 50 * millisecond
# DA strenght on D2-Autoreceptors 
alpha_DA_arD2 = .1

# DA burst height produced by a reward (modulated by the prediction error)
alpha_Rew_DA = 15 # [sp/s]
# DA strenght on striatal inputs
gamma_DAth        = 1
gamma_DAstrenght  = .025
gamma_DAbySuccess = 4 # [sp/s]
alpha_SuccessEMA   = .8
gamma_DA_LTD      = 0.025 # (1+gamma_DA_LTD * DA) -> 1.8 (4.0) - 2.2 (6.0) - 2.6 (8.0)
gamma_mGluR_LTD   = 0.01 # 60 * (1+DA) -> 60 * 5 - 60*9 -> 300-700
gamma_eCB1_LTD    = 0.1 # (1-20)

# Noise level (%)
Cortex_N        =   0.01
Striatum_N      =   0.01
Striatum_corr_N =   N_factor * 0.5
STN_N           =   0.01
GPi_N           =   0.03
Thalamus_N      =   0.01
SNc_N           =   0.1
# if Weights_N:
#     Weights_N *= aux_X
# Cortex_N        =   0.01
# Striatum_N      =   0.01
# Striatum_corr_N =   0.1
# STN_N           =   0.01
# GPi_N           =   0.03
# Thalamus_N      =   0.01
# SNc_N           =   0.1

# DA buffer for a time delay on arD2
DA_buffSize = int(round(arD2_lag/dt))
DA_buff = np.zeros((DA_buffSize,))
DA_buffIndex = 0

# Sigmoid parameter
Vmin       =  0.0
Vmax       = 20.0
Vh0         = 18.18 # 18.38
Vc         =  6.0

# Learning parameters
decision_threshold = 30
alpha_c     = 0.2  # 0.05
Wmin, Wmax = 0.45, 0.55
#Wmin, Wmax = 0.4, 0.6

# Reward extension [ms]
delayReward = 300 * millisecond
reward_ext = 50 * millisecond + delayReward

# Data saving configuration
avoidUnsavedPlots = True
# STORE_DATA = True
STORE_FAILED = False
FAILED_FILE = folder+'failed_'+str(DA)

file_base =  'CBG-DA_'

if garivierMoulines:
    file_base += 'gM_'
if useCorrelatedNoise:
    file_base += 'c_'
if constantLTD:
    file_base += 'cLTD_' #+ str(constantLTD)
if dynamicDA:
    file_base += 'd_'
    if dynamicDAoverA:
        file_base += 'pAdv_'
if relativeValue:
    file_base += 'rv_'
if invertAt:
    for rAt in invertAt:
        file_base += 'r'+str(rAt)+'_'
if N_factor != 1.0:
    file_base += 'N'+str(N_factor)+'_'
if aux_X != 1:
    file_base += 'x'+str(aux_X)+'_'
if aux_Y != 1:
    file_base += 'y'+str(aux_Y)+'_'
if staticThreshold:
    file_base += 'sTh_'
if staticCtxStr:
    file_base += 'sCxtStr_'

file_base += str(DA)+'_'+str(alpha_LTP)+'_'+str(alpha_LTD)+'_'

if nFile:
    file_base += nFile
else:
    file_base += datetime.now().strftime("%Y%m%d")

if STORE_DATA or STORE_FAILED:
    
    if 'corcovado' == gethostname() and  not folder:
        filename = '/mnt/Data/Cristobal/tonicDA_effects/connections/Exp/'+file_base+''
    else:
        filename = dataFolder+file_base+''

if STORE_DATA:
    DATA = OrderedDict([('SNc',0),('P-buff',0),('choice',0),('nchoice',0),('motTime',0),('weights',''),('cogTime',0),
                        ('failedTrials',0),('persistence',0),('values',''),('P',0),('P-3',0),('R',0),('R-buff',0),
                        ('P-3',0),('LTD-LTP',''),('A',0),('A-buff',0),('pchoice',''),('Regret',0),('cumRegret',0),('c_rewards','')])
    with open(filename,"w") as records:
        wtr = csv.DictWriter(records, DATA.keys(),delimiter=',')
        wtr.writeheader()

# Include learning
# learn = True 
# applyInMotorLoop = True

# Compute all the trial (true) or just until a motor decision has been made
completeTrials = True

# Force selection in motor loop for every trial (discards trials with no decision)
forceSelection = True

# Maximum number of failed trials. Available if forceSelection is enable
nbFailedTrials = .5*nbTrials
# Maximum number of continuous failed trials
nbContFailedTrials = 20

# Use a regular pattern for cues selection
# regPattern = False
nbTrialsPerPattern = 100

if garivierMoulines:
    invertAt = [300,500]
    nbTrials = 1000
    n = 3
    pattern = np.array([0,1,2])
    nPatterns = len(pattern)
    n_availableOptions = 3
else:
    pattern = np.array([[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]])
    nPatterns = len(pattern)
    n_availableOptions = 2

probChoiceBuffSize = 10
probChoice = np.zeros((nPatterns,probChoiceBuffSize))
for p in range(nPatterns):
    for i in range(probChoiceBuffSize/2):
        probChoice[p][i] = 1

# Enable debug messages
# doPrint = False

if invertAt and not (type(invertAt).__module__ == 'numpy' ):
    invertAt = np.array(invertAt)


# Initialization of the random generator
if randomInit:
    np.random.seed(randomInit)

# Helper functions
# -----------------------------------------------------------------------------
def sigmoid(V,Vmin=Vmin,Vmax=Vmax,Vh=Vh0,Vc=Vc):
    return  Vmin + (Vmax-Vmin)/(1.0+np.exp((Vh-V)/Vc))

def flip(items, ncol):
    return itertools.chain(*[items[i::ncol] for i in range(ncol)])

# def rewardShape(t,rewardTime,dev=0.1):
#     return np.exp(-(t-(rewardTime+3*dev))**2/(2*dev**2))

def strThreshold(DA):
    if staticThreshold:
        return aux_Y*4*Vh0 # Considers DA = 4 sp/s => (0.5 * 4 + 1)*Vh
    return gamma_DAth * DA + Vh0 # 

def strSlope(DA):
    return 6.0 #* (1.0 + gamma_DAstrenght * DA)

def ctxStr(DA):
    if staticCtxStr:
        return aux_X*4 #* (1.0 + gamma_DAstrenght * DA)
    # return (1 + aux_X*gamma_DAstrenght*DA)
    return (1 + gamma_DAstrenght*DA)



def setDAlevels(DA):
    Striatum_cog['DA'] = DA
    Striatum_mot['DA'] = DA
    Striatum_ass['DA'] = DA

def noise(Z, level):
    Z = np.random.normal(0,level,Z.shape)*Z
    return Z

def correlatedNoise(Z, pastValue, factorLevel, baseLevel=40, tau=STR_N_tau):
    noise = np.random.normal(0,factorLevel*baseLevel,Z.shape)
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

def resetPlot():
    global pastW,cogWDiff,cogMarker,motMarker,movMarker
    W = W_cortex_cog_to_striatum_cog
    dw = np.diag(W._weights)-np.diag(pastW)
    pastW = np.copy(W._weights)

    for p in range(n):
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

    neuralData_y.fill(None)
    neuralData_y2.fill(None)
    # neuralData_ysnc.fill(None)
    neuralData_ysncpptn.fill(None)
    neuralData_y_ctx_ass.fill(None)

    # neuralData_ypptn.fill(None)
    neuralData_yth.fill(None)
    neuralData_yr1.fill(None)
    neuralData_yr2.fill(None)
    neuralData_yr3.fill(None)
    setXlim_d()

def clip(V, Vmin, Vmax):
    return np.minimum(np.maximum(V, Vmin), Vmax)

def positiveClip(V):
    return np.maximum(V, 0.0)

def fillData():
    DATA['SNc']          = SNc_dop['V'][0][0]
    DATA['P-buff']       = np.array(P[-perfBuff:]).mean()
    DATA['choice']       = choice
    DATA['nchoice']      = nchoice if not garivierMoulines else -1
    DATA['motTime']      = motDecisionTime
    DATA['weights']      = '\t'.join(['{:.5f}'.format(i) for i in W_cortex_cog_to_striatum_cog.weights.diagonal()])
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
    DATA['A-buff']       = np.array(A[-perfBuff:]).mean()
    DATA['pchoice']      = '\t'.join(['{:.2f}'.format(np.nanmean(i)) for i in probChoice])
    DATA['Regret']       = Regret[-1:][0]
    DATA['cumRegret']    = cumRegret
    DATA['c_rewards']    = '\t'.join(['{:.2f}'.format(i) for i in cues_reward])

def printData():
    with open(filename,'a') as records:
        wtr = csv.DictWriter(records, DATA.keys(),delimiter=',')
        fillData()
        wtr.writerow(DATA)

def D2_IPSC_kernel(t,t_DelRew):
    if t < t_DelRew:
        return 0
    return np.exp(-(t-t_DelRew)/arD2_tau)

def convolv_D2Kernel(t,currentValue,input):
    return currentValue + input * D2_IPSC_kernel(t,motDecisionTime/1000+arD2_lag)

SNc_dop   = zeros((1,1), """  D2_IPSC = - alpha_DA_arD2 * DAtoD2c;
                             Ir = np.maximum(Irew, Ie_rew);
                             I = Ir + D2_IPSC;
                             n = correlatedNoise(I,n,SNc_N,alpha_Rew_DA,SNc_N_tau);
                             It = I + n;
                             u = positiveClip(It - SNc_h);
                             dV/dt = (-V + u)/SNc_tau; Irew; Ie_rew; SNc_h; V_lag; DAtoD2c""")
                             # u/dt = (-u + (It - SNc_h))/SNc_tau;
                             # V = positiveClip(u); Irew; Ie_rew; SNc_h; V_lag; DAtoD2c""")

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

Striatum_cog = zeros((n,1), """Is = I*ctxStr(DA);
                             n = striatalNoise(Is,n);
                             It = Is + n;
                             Vh = strThreshold(DA);
                             Vc = strSlope(DA);
                             u = sigmoid(It - Striatum_h,Vmin,Vmax,Vh,Vc);
                             dV/dt = (-V + u)/Striatum_tau; I; DA""")

Striatum_mot = zeros((1,n), """Is = I*ctxStr(DA);
                             n = striatalNoise(Is,n);
                             It = Is + n;
                             Vh = strThreshold(DA);
                             Vc = strSlope(DA);
                             u = sigmoid(It - Striatum_h,Vmin,Vmax,Vh,Vc);
                             dV/dt = (-V + u)/Striatum_tau; I; DA""")

Striatum_ass = zeros((n,n), """Is = I*ctxStr(DA);
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

cues_mot = np.arange(n)
cues_cog = np.arange(n)
labels = ['Cog'+str(i) for i in cues_cog]+['Mot'+str(i) for i in cues_mot]
labels_plusOne = ['Cog-'+str(i+1) for i in cues_cog]+['Mot-'+str(i+1) for i in cues_mot]
ctxLabels = labels_plusOne
if zeroValues:
    cog_cues_value = np.zeros(n)
else:
    cog_cues_value = np.ones(n) * 0.5
mot_cues_value = np.ones(n) * 0.5

if garivierMoulines:
    cues_reward1 = np.array([0.5,0.4,0.3,])
    cues_reward2 = np.array([0.5,0.4,0.9,])
    cues_reward3 = np.array([0.5,0.4,0.3,])
    cues_rewards = [cues_reward1, cues_reward2, cues_reward3]
elif pasquereau:
    cues_reward1 = np.array([3.0,2.0,1.0,0.0])/3.0
    cues_rewards = [cues_reward1]
else:
    cues_reward1 = np.array([3.0,2.0,1.0,0.0])/3.0
    cues_reward2 = np.array([0.0,1.0,2.0,3.0])/3.0
    cues_rewards = [cues_reward1, cues_reward2]

cues_reward = cues_rewards[0]

# Connectivity
# -----------------------------------------------------------------------------
W = DenseConnection( Cortex_cog('V'),   Striatum_cog('I'), 1.0)
if cogInitialWeights == None:
    init_weights(W)
else:
    W._weights = cogInitialWeights
W_cortex_cog_to_striatum_cog = W

if doPrint:
    print "Cognitive weights: ", np.diag(W._weights)

W = DenseConnection( Cortex_mot('V'),   Striatum_mot('I'), 1.0)
if motInitialWeights != None:
    W._weights = motInitialWeights
else:
    init_weights(W)
W_cortex_mot_to_striatum_mot = W

W = DenseConnection( Cortex_ass('V'),   Striatum_ass('I'), 1.0)
init_weights(W)
W = DenseConnection( Cortex_cog('V'),   Striatum_ass('I'), np.ones((1,2*n-1)))
init_weights(W,0.2)
W = DenseConnection( Cortex_mot('V'),   Striatum_ass('I'), np.ones((2*n-1,1)))
init_weights(W,0.2)
DenseConnection( Cortex_cog('V'),   STN_cog('I'),       1.0 )
DenseConnection( Cortex_mot('V'),   STN_mot('I'),       1.0 )
if n_availableOptions == 3:
    DenseConnection( Striatum_cog('V'), GPi_cog('I'),      -2.4 )
    DenseConnection( Striatum_mot('V'), GPi_mot('I'),      -2.4 )
else:
    DenseConnection( Striatum_cog('V'), GPi_cog('I'),      -2.4 )
    DenseConnection( Striatum_mot('V'), GPi_mot('I'),      -2.4 )
DenseConnection( Striatum_ass('V'), GPi_cog('I'),      -2.0*np.ones((1,2*n-1)))
DenseConnection( Striatum_ass('V'), GPi_mot('I'),      -2.0*np.ones((2*n-1,1)))
if n_availableOptions == 3:
    DenseConnection( STN_cog('V'),      GPi_cog('I'),       1.0*np.ones((2*n-1,1)))
    DenseConnection( STN_mot('V'),      GPi_mot('I'),       1.0*np.ones((1,2*n-1)))
else:
    DenseConnection( STN_cog('V'),      GPi_cog('I'),       1.0*np.ones((2*n-1,1)))
    DenseConnection( STN_mot('V'),      GPi_mot('I'),       1.0*np.ones((1,2*n-1)))
DenseConnection( GPi_cog('V'),      Thalamus_cog('I'), -0.5 )
DenseConnection( GPi_mot('V'),      Thalamus_mot('I'), -0.5 )
DenseConnection( Thalamus_cog('V'), Cortex_cog('I'),    1.0 )
DenseConnection( Thalamus_mot('V'), Cortex_mot('I'),    1.0 )
DenseConnection( Cortex_cog('V'),   Thalamus_cog('I'),  0.4 )
DenseConnection( Cortex_mot('V'),   Thalamus_mot('I'),  0.4 )
DenseConnection( SNc_dop('V'),      Striatum_cog('DA'), 1.0 )
DenseConnection( SNc_dop('V'),      Striatum_mot('DA'), 1.0 )
DenseConnection( SNc_dop('V'),      Striatum_ass('DA'), 1.0 )

# Trial setup
# -----------------------------------------------------------------------------
inputCurrents_noise = np.zeros(3*n_availableOptions)
c1,c2,m1,m2 = 0,0,0,0
SNc_dop['SNc_h'] = SNc_h_base
wnoise = 0

@clock.at(500*millisecond)
def set_trial(t):
    global cues_cog, cogDecision, cogDecisionTime, motDecision, motDecisionTime, c1, c2, m1, m2
    global inputCurrents_noise, flag, ctxLabels
    global wnoise

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

    inputCurrents_noise = np.random.normal(0,v*Cortex_N,3*n_availableOptions)
    
    cogDecision = False
    cogDecisionTime = 3500.0
    motDecision = False
    motDecisionTime = 3500.0

    if neuralPlot and n_availableOptions == 2:
        ctxLabels = labels_plusOne[:] #('Cog1','Cog2','Cog3','Cog4','Mot1','Mot2','Mot3','Mot4')
        ctxLabels[c1] += ' *'
        ctxLabels[m1+n] += ' *'
        ctxLabels[c2] += ' ^'
        ctxLabels[m2+n] += ' ^'
        addLegend(axctx,neuralSignals2,ctxLabels,loc='upper left')
        # axctx.legend(flip(neuralSignals2,n),flip(ctxLabels,n),loc='upper right', ncol=n, fontsize='x-small',framealpha=0.6, # bbox_to_anchor= (1.08, 0.5), #ncol=2, # bbox_to_anchor= (1.2, 0.5), 
        #         borderaxespad=0, frameon=False)

    if Weights_N:
        W = W_cortex_cog_to_striatum_cog
        for w in range(n):
            wnoise = np.random.normal(0,(W.weights[w,w]-Wmin)*Weights_N)
            # if useCorrelatedNoise:
            #     delta = dt*(-wnoise + wnoise_t)/STR_N_tau
            #     wnoise = wnoise + delta
            # else:
            #     wnoise = wnoise_t
            W.weights[w,w] = clip(W.weights[w,w] + wnoise, Wmin, Wmax)

@before(clock.tick)
def computeSoftInput(t):
    if motDecisionTime < 3500:
        inputsOffAt =  motDecisionTime/1000+delayReward+0.2
    else:
        inputsOffAt = 3.2
    v = sigmoid(t,0,cues_amplitude,.725,.042) - sigmoid(t,0,cues_amplitude,inputsOffAt,.084)


    # if n_availableOptions == 2:
    #     Cortex_mot['Iext'][0,m1]  = v + inputCurrents_noise[0]
    #     Cortex_mot['Iext'][0,m2]  = v + inputCurrents_noise[1]
    #     Cortex_cog['Iext'][c1,0]  = v + inputCurrents_noise[2]
    #     Cortex_cog['Iext'][c2,0]  = v + inputCurrents_noise[3]
    #     Cortex_ass['Iext'][c1,m1] = v + inputCurrents_noise[4]
    #     Cortex_ass['Iext'][c2,m2] = v + inputCurrents_noise[5]  
    # else if n_availableOptions == n:
    for i in range(n_availableOptions):
        Cortex_mot['Iext'][0          ,cues_mot[i]] = v + inputCurrents_noise[i*3]
        Cortex_cog['Iext'][cues_cog[i],0          ] = v + inputCurrents_noise[i*3+1]
        Cortex_ass['Iext'][cues_cog[i],cues_mot[i]] = v + inputCurrents_noise[i*3+2]

@clock.at(3000*millisecond)
def unset_trial(t):
    Cortex_mot['Iext'] = 0
    Cortex_cog['Iext'] = 0
    Cortex_ass['Iext'] = 0

@clock.at(duration*.9)
def storeResults(t):
    if completeTrials and STORE_DATA and motDecision:
        printData()

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
    if motDecisionTime/1000 + delayReward < t:
        if t < motDecisionTime/1000 + reward_ext and SNc_dop['Irew'] == 0:
            SNc_dop['Irew'] = pError * alpha_Rew_DA
        elif t > motDecisionTime/1000 + reward_ext:
            SNc_dop['Irew'] = 0

    if cogReward and cogDecisionTime/1000 < t:
        if t < (cogDecisionTime/1000 + reward_ext - delayReward) and SNc_dop['Ie_rew'] == 0:
            # print 'Value for current:',cog_cues_value[np.argmax(Cortex_cog['V'])],float(cog_cues_value[np.argmax(Cortex_cog['V'])]),float(cog_cues_value[np.argmax(Cortex_cog['V'])])*alpha_Rew_DA
            SNc_dop['Ie_rew'] = float(cog_cues_value[np.argmax(Cortex_cog['V'])]) * alpha_Rew_DA
        elif t > (cogDecisionTime/1000 + reward_ext - delayReward):
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
    if stopAfterSelecion and motDecisionTime < 3500:
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
smoothR = 0
smoothA = 0
cumRegret = 0
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

        for c in range(n):
            dw = dDA * alpha * Striatum_cog['V'][c][0] / ((reward_ext - delayReward)/millisecond)
            w = clip(W.weights[c, c] + dw, Wmin, Wmax)
            trialLtdLtp[c,2] += dw
            trialLtdLtp[c,3] += w - W.weights[c, c]
            W.weights[c,c] = w

        if applyInMotorLoop:
            W = W_cortex_mot_to_striatum_mot
            for c in range(n):
                dw = dDA * alpha * Striatum_mot['V'][0][c] / ((reward_ext - delayReward)/millisecond)
                w = clip(W.weights[c, c] + dw, Wmin, Wmax)
                W.weights[c,c] = w


@before(clock.tick)
def corticostriatalLTD(t):
    if constantLTD:
        W = W_cortex_cog_to_striatum_cog
        for pop in range(n):
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
            for pop in range(n):
                dw = - 0.001 * alpha_LTD * Striatum_mot['It'][0][pop] * (1 + gamma_DA_LTD * SNc_dop['V'])
                w = clip(W.weights[pop][pop] + dw,Wmin,Wmax)
                W.weights[pop][pop] = w

#@after(clock.tick)
@clock.every(50*millisecond)
def register(t):
    global currentTrial, continuousFailedTrials, selectionError, cogDecision
    global cogDecisionTime, motDecision, motDecisionTime
    global choice, nchoice, mchoice, pError, smoothR, smoothA, pastW, cog_cues_value
    global cumRegret
    
    if not cogDecision:
        U = np.sort(Cortex_cog['V'].ravel())
        if abs(U[-1] - U[-2]) >= decision_threshold and t <= duration-500*millisecond and t >= 500*millisecond:
            cogDecisionTime = 1000.0*clock.time
            cogDecision = True
            if cogReward:
                SNc_dop['Ie_rew'] = float(cog_cues_value[np.argmax(Cortex_cog['V'])]) * alpha_Rew_DA

    U = np.sort(Cortex_mot['V']).ravel()

    # No motor decision yet
    if motDecision or abs(U[-1] - U[-2]) < decision_threshold or t > duration-500*millisecond or t < 500*millisecond: return

    motDecision = True
    motDecisionTime = 1000.0*clock.time

    # A motor decision has been made
    mot_choice = np.argmax(Cortex_mot['V'])
    cog_choice = np.argmax(Cortex_cog['V'])

    # The motor selection is the executed one, then it
    # defines the selected cue in a cognitive domain.
    # The actual cognitive selection might differ.
    if not garivierMoulines:
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


        # How good was the selection compared to the best choice presented
        regret = np.max([cues_reward[choice],cues_reward[nchoice]]) - cues_reward[choice]
        perceived_regret = np.max([cog_cues_value[choice],cog_cues_value[nchoice]]) - cog_cues_value[choice]
        # advantage = 1 + (cues_reward[choice] - np.max([cues_reward[choice],cues_reward[nchoice]]))
        advantage = 1 - regret
        perceived_advantage = 1 - perceived_regret
        A.append(advantage)
        Regret.append(regret)
        cumRegret += regret

        sPair = [(choice < nchoice) * choice + (nchoice < choice) * nchoice ,
                 (choice < nchoice) * nchoice + (nchoice < choice) * choice ]
        pairPos = 0
        for i in range(nPatterns):
            if pattern[i][0]==sPair[0] and pattern[i][1]==sPair[1]:
                pairPos = i

        probChoice[pairPos][currentTrial%probChoiceBuffSize] = choice<nchoice

    else: # garivierMoulines
        choice = mot_choice
        mchoice = mot_choice

        if choice == np.argmax(cues_reward):
            P.append(1.0)
        else:
            P.append(0.0)

        # How good was the selection compared to the best choice presented
        regret = np.max(cues_reward) - cues_reward[choice]
        perceived_regret = np.max(cog_cues_value) - cog_cues_value[choice]
        # advantage = 1 + (cues_reward[choice] - np.max([cues_reward[choice],cues_reward[nchoice]]))
        advantage = 1 - regret
        perceived_advantage = 1 - perceived_regret
        A.append(advantage)
        Regret.append(regret)
        cumRegret += regret
        for c in cues_cog:
            if c == choice:
                probChoice[c][currentTrial%probChoiceBuffSize] = 1
            else:
                probChoice[c][currentTrial%probChoiceBuffSize] = 0


    W = W_cortex_cog_to_striatum_cog
    Wm = W_cortex_mot_to_striatum_mot
    if learn:
        # Compute reward
        reward = np.random.uniform(0,1) < cues_reward[choice]
        R.append(reward)
        smoothR = alpha_SuccessEMA * smoothR + (1-alpha_SuccessEMA) * reward
        smoothA = alpha_SuccessEMA * smoothA + (1-alpha_SuccessEMA) * (perceived_advantage if usePerception else advantage)
        if dynamicDA:
            if dynamicDAoverA:
                sSmoothA = smoothA
                if minSmoothA:
                    sSmoothA = np.max((sSmoothA-minSmoothA,0))/(1-minSmoothA)
                print "Smooth advantage: ",smoothA," [ sat -",sSmoothA,"]"
                SNc_dop['SNc_h'] = SNc_h_base - gamma_DAbySuccess * sSmoothA
            else:
                SNc_dop['SNc_h'] = SNc_h_base - gamma_DAbySuccess * smoothR

        # Compute prediction error
        pError = reward - cog_cues_value[choice]
        # Update cues values
        cog_cues_value[choice] += pError* alpha_c

        ## Error regulates PPNg activity, the prediction error encoding signal
        if relativeValue:
            if not garivierMoulines:
                pError = reward - (cog_cues_value[choice]+cog_cues_value[nchoice])/2.0
            else:
                pError = reward - np.mean(cog_cues_value)


        if oneStepConstantLTD:
            W_cortex_cog_to_striatum_cog._weights -= alpha_LTD

    if doPrint:
        if not garivierMoulines:
            if P[-1]:
                print "Choice:            [%d] / %d  (good)" % (choice,nchoice)
            else:
                print "Choice:              %d / [%d] (bad)" % (nchoice,choice)
        else:
            if P[-1]:
                print "Choice:            [%d]  (good)" % (choice)
            else:
                print "Choice:            [%d]  (bad)" % (choice)
        if learn:
            print "Reward (%3d%%) :     %d / %f" % (int(100*cues_reward[choice]),reward, smoothR)
            print "SNc_h :            ",SNc_dop['SNc_h'][0]
            print "Mean performance:  ",np.array(P[-perfBuff:]).mean()
            print "Mean reward:        %.3f" % np.array(R).mean()
            print "Response time:      %d ms" % motDecisionTime
            print "Cognitive weights: ", np.diag(W._weights)
            print "Motor weights:     ", np.diag(Wm._weights)
            print "Cognitive values:  ", cog_cues_value
            print "Choices prob:      ",'  '.join(['{:.2f}'.format(np.nanmean(i)) for i in probChoice])
            print "Reward  prob:      ",'  '.join(['{:.2f}'.format(i) for i in cues_reward])

    if STORE_DATA and not completeTrials:
        printData()

    if completeTrials:
        return

    # In case that there is no interest in continuing this trial, 'completeTrials == False',
    # the model is set to its initial state and the clock is reset, continuing with a new
    # trial.
    currentTrial += 1
    continuousFailedTrials = 0
    partialReset()
    clock.reset()

if neuralPlot:
    cogMarker = motMarker = movMarker = []
    plotSteps = 1*millisecond
    nData = 2*n
    nData2 = 2*n
    nDataSnc = 1
    nDataPptn = 3
    nData4 = n
    nData4_2 = nData4
    nData5 = n+2 # values plus choiced cue
    nDataPer = 3
    nDataProb = nPatterns

    # Right column plots
    nrData = n*n
    nrData2 = 2*n
    nrData3 = 2*n

    fig, ((axl1,axr1),(axl2,axr2),(axl3,axr3),(axl4,axr4)) = plt.subplots(4,2,figsize=(12,6),num="Trial - DA: "+str(DA))#+" X_"+str(aux_X)+" Y_"+str(aux_Y))

    axctx     = axl1
    axctx_ass = axr1
    axstr     = axl2
    axstr_ass = axr2
    axgpi     = axl3
    # axsnc     = axl3
    axsncPPTN = axl4
    axstn     = axr3
    axth      = axr4
    # axpptn    = axr4


    fig2, ((axl1_2,axr1_2),(axl2_2,axr2_2),(axl3_2,axr3_2)) = plt.subplots(3,2,figsize=(12,6),num="Exp - DA: "+str(DA))#+" X_"+str(aux_X)+" Y_"+str(aux_Y))

    axper  = axl1_2
    axw    = axr1_2
    axprob = axl2_2
    axv    = axr2_2
    axsnct = axl3_2
    axentr = axr3_2

    axstr.set_ylim(-2,25)
    axstr.set_title('Striatal activity', fontsize=10)
    axctx.set_ylim(-2,150)
    axctx.set_title('Cortical activity', fontsize=10)
    axctx_ass.set_ylim(-2,80)
    axctx_ass.set_title('Associative Cortical activity', fontsize=10)
    # axsnc.set_ylim(-2,20)
    # axsnc.set_title('SNc activity', fontsize=10)
    axsncPPTN.set_ylim(-2,20)
    axsncPPTN.set_title('SNc and PPTN activity', fontsize=10)
    # axpptn.set_ylim(-15,15)
    # axpptn.set_title('PPTN activity', fontsize=10)

    #axsnct = axsnc.twiny()
    #plt.setp(axsnct.get_xticklabels(), visible=False)
    #axsnct.xaxis.set_ticks_position('none') 
    axsnct.set_ylim(0,10)
    axsnct.set_title('SNc tonic activity', fontsize=10)

    axentr.set_ylim(0,1)
    axentr.set_title('Entropy', fontsize=10)

    axper.set_ylim(0,1)
    axper.set_title('Performance', fontsize=10)
    axreg = axper.twinx()
    axreg.set_ylim(0,200)
    axprob.set_ylim(0,1.1)
    axprob.set_title('Choice probability', fontsize=10)

    neuralData_x = np.arange(0,duration+plotSteps,plotSteps)
    neuralData_y = np.full((len(neuralData_x),nData),None,dtype=float)
    neuralData_y2 = np.full((len(neuralData_x),nData2),None,dtype=float)
    # neuralData_ysnc = np.full((len(neuralData_x),nDataSnc),None,dtype=float)
    # neuralData_ypptn = np.full((len(neuralData_x),nDataPptn),None,dtype=float)
    neuralData_ysncpptn = np.full((len(neuralData_x),nDataSnc+nDataPptn),None,dtype=float)
    
    neuralSignals = axstr.plot(neuralData_x,neuralData_y, alpha=0.7)
    neuralSignals2 = axctx.plot(neuralData_x,neuralData_y2, alpha=0.7)
    # neuralSignalsSnc = axsnc.plot(neuralData_x,neuralData_ysnc, alpha=0.7)
    neuralSignalsSncPPTN = axsncPPTN.plot(neuralData_x,neuralData_ysncpptn, alpha=0.7)
    # neuralSignalsPptn = axpptn.plot(neuralData_x,neuralData_ypptn, alpha=0.3,linewidth=3)

    axv.set_ylim(0,1)
    axv.set_title('Learned cognitive values', fontsize=10)
    axw.set_ylim(0.39,.61)
    axw.set_title('Cognitive corticostriatal weights', fontsize=10)
    axdw = axw.twinx()
    axdw.set_ylim(-.02,.06)

    axdw.axhline(0, color='black')
    neuralData_x2 = np.arange(nbTrials)
    neuralData_y_2 = np.full((len(neuralData_x2),1),None,dtype=float)
    neuralData_y4 = np.full((len(neuralData_x2),nData4),None,dtype=float)
    neuralData_y4_2 = np.full((len(neuralData_x2),nData4_2),None,dtype=float)
    neuralData_y5 = np.full((len(neuralData_x2),nData5),None,dtype=float)
    neuralData_y_entr = np.full((len(neuralData_x2),1),None,dtype=float)
    neuralSignals_entr = axentr.plot(neuralData_x2,neuralData_y_entr)
    neuralSignals_2 = axsnct.plot(neuralData_x2,neuralData_y_2,alpha=0.3,color='magenta',linewidth=3)
    neuralSignals4 = axw.plot(neuralData_x2,neuralData_y4, alpha=0.7)
    neuralSignals4_2 = axdw.plot(neuralData_x2,neuralData_y4_2, alpha=0.3,linewidth=5)
    neuralSignals5 = axv.plot(neuralData_x2,neuralData_y5, alpha=0.7)
    neuralData_y_perf = np.full((len(neuralData_x2),nDataPer),None,dtype=float)
    neuralData_y_prob = np.full((len(neuralData_x2),nDataProb),None,dtype=float)
    neuralData_y_reg = np.full((len(neuralData_x2),1),None,dtype=float)
    neuralSignals_per = axper.plot(neuralData_x2,neuralData_y_perf, alpha=0.7)
    neuralSignals_prob = axprob.plot(neuralData_x2,neuralData_y_prob, alpha=0.7, linewidth=3)
    neuralSignals_reg = axreg.plot(neuralData_x2,neuralData_y_reg, alpha=0.7, color='purple', linewidth=3)


    axstr_ass.set_ylim(-2,25)
    axstr_ass.set_title('Associative striatal activity', fontsize=10)
    axgpi.set_ylim(-2,150)
    axgpi.set_title('GPi activity', fontsize=10)
    axstn.set_ylim(-2,80)
    axstn.set_title('STN activity', fontsize=10)
    axth.set_ylim(-2,80)
    axth.set_title('Th activity', fontsize=10)

    neuralData_yr1 = np.full((len(neuralData_x),nrData),None,dtype=float)
    neuralData_yr2 = np.full((len(neuralData_x),nrData2),None,dtype=float)
    neuralData_yr3 = np.full((len(neuralData_x),nrData3),None,dtype=float)
    neuralData_yth = np.full((len(neuralData_x),nData2),None,dtype=float)
    neuralData_y_ctx_ass = np.full((len(neuralData_x),nrData),None,dtype=float)
    neuralSignalsr1 = axstr_ass.plot(neuralData_x,neuralData_yr1, alpha=0.7)
    neuralSignalsr2 = axgpi.plot(neuralData_x,neuralData_yr2, alpha=0.7)
    neuralSignalsr3 = axstn.plot(neuralData_x,neuralData_yr3, alpha=0.7)
    neuralSignalsTh = axth.plot(neuralData_x,neuralData_yth, alpha=0.7)
    neuralSignalsCtx_ass = axctx_ass.plot(neuralData_x,neuralData_y_ctx_ass, alpha=0.7)

    plt.setp(neuralSignals5[n:], 'marker', 'D','linestyle','','alpha',0.2,'markersize',5)
    plt.setp(neuralSignals5[n], 'color','magenta')
    
    for l in range(n):
        plt.setp(neuralSignals2[l+n],'color',plt.getp(neuralSignals2[l],'color'),'ls','--')
        plt.setp(neuralSignals[l+n],'color',plt.getp(neuralSignals[l],'color'),'ls','--')
        plt.setp(neuralSignalsr2[l+n],'color',plt.getp(neuralSignals[l],'color'),'ls','--')
        plt.setp(neuralSignalsr3[l+n],'color',plt.getp(neuralSignals[l],'color'),'ls','--')
        plt.setp(neuralSignalsTh[l+n],'color',plt.getp(neuralSignals[l],'color'),'ls','--')
        # plt.setp(neuralSignals_per[l+n],'color',plt.getp(neuralSignals[l],'color'),'ls','--')
        # plt.setp(neuralSignals_str_th[l+n],'color',plt.getp(neuralSignals[l],'color'),'ls','--')
    # for l in range(2):
    #     plt.setp(neuralSignals3_2[l+1],'ls',':')
    

    axd = [axstr,axctx,axstr_ass,axgpi,axstn, axth, axsncPPTN, axctx_ass] # axsnc, axpptn]
    axt = [axsnct,axw,axdw,axv,axper,axprob,axentr]

    def setXlim_d(t=duration):
        for axis in axd:
            axis.set_xlim(0,t)
    setXlim_d()
    for axis in axd:
        axis.grid(color='gray',which='both',alpha=0.3)
    for axis in axt:
        axis.set_xlim(0,nbTrials)
    for axis in axd+axt:
        axis.tick_params(axis='both', which='major', labelsize=10)

    axsncPPTN.set_xlabel('Time [s]')
    axth.set_xlabel('Time [s]')

    def addLegend(axs,signals,labels=labels,n=n,doflip=True,loc='upper right'):
        if doflip:
            axs.legend(flip(signals,n),flip(labels,n),loc=loc, ncol=n, fontsize='x-small',
                borderaxespad=0, frameon=False)
        else:
            axs.legend(signals,loc=loc, ncol=n, fontsize='x-small',borderaxespad=0, frameon=False)

    # addLegend(axstr,neuralSignals)
    addLegend(axper,neuralSignals_per,['Performance','Advantage','Regret'],loc='upper left')
    if not garivierMoulines:
        addLegend(axprob,neuralSignals_prob,[str(x[0])+' - '+str(x[1]) for x in pattern])
    else:
        addLegend(axprob,neuralSignals_prob,[str(x) for x in pattern])
    addLegend(axreg,neuralSignals_reg,'Acc. regret')
    addLegend(axentr,neuralSignals_entr,'Entropy')
    addLegend(axctx,neuralSignals2,labels_plusOne)
    addLegend(axctx_ass,neuralSignalsCtx_ass,['ASS-'+str(i+1)+'_'+str(j+1) for j in range(n) for i in range(n)])
    addLegend(axsncPPTN,neuralSignalsSncPPTN,['SNc','PPTN'],n=3)
    addLegend(axw,neuralSignals4,['Wcog['+str(i)+']' for i in cues_cog])
    addLegend(axv,neuralSignals5,['Vcog['+str(i)+']' for i in cues_cog]+['Selection']+['Not selected'])
    # addLegend(axstr_ass,neuralSignalsr1,['StrA'+str(i)+'_'+str(j) for j in range(n) for i in range(n)])
    # addLegend(axgpi,neuralSignalsr2)
    # addLegend(axstn,neuralSignalsr3)
    # addLegend(axth,neuralSignalsTh)
    # addLegend(axpptn,neuralSignalsPptn,['I_r','I_rew','Ie_rew'])

    fig.tight_layout()
    fig2.tight_layout()

    def storePlots():
        fig.savefig(plotsFolder+file_base+'_'+str(currentTrial)+".pdf",bbox_inches='tight')
        fig.savefig(plotsFolder+file_base+'_'+str(currentTrial)+".png",bbox_inches='tight')
        fig2.savefig(plotsFolder+file_base+'_'+str(currentTrial)+"2.png",bbox_inches='tight')
        fig2.savefig(plotsFolder+file_base+'_'+str(currentTrial)+"2.pdf",bbox_inches='tight')

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
            neuralData_y5[currentTrial][n] = float((n-1)-choice)/float(n-1)
            if not garivierMoulines:
                neuralData_y5[currentTrial][n+1] = float((n-1)-nchoice)/float(n-1)
            for i in np.arange(2)+n:
                neuralSignals5[i].set_ydata(neuralData_y5[:,i])
        if len(movMarker) == 0 and motDecisionTime/1000 + delayReward < t:
            reloadLabel = True
            if R[-1]:
                movMarker = axctx.plot(t,40,marker='.',alpha=0.4,markersize=20,color='g',linestyle='None')
            else:
                movMarker = axctx.plot(t,40,marker='x',alpha=0.4,markersize=10,color='r',linestyle='None')

        if reloadLabel:
            addLegend(axctx,neuralSignals2 + \
                (len(cogMarker) != 0) * cogMarker + \
                (len(motMarker) != 0) * motMarker + \
                (len(movMarker) != 0) * movMarker, 
                ctxLabels + (len(cogMarker) != 0)*['C-choise'] +
                (len(motMarker) != 0) * ['M-choise'] +
                (len(movMarker) != 0) * ['Reward'], loc='upper left')

    @clock.at(dt)
    def plotTrial_data(t):
        neuralData_y_2[currentTrial] = SNc_dop['V']
        neuralData_y4[currentTrial] = np.diag(W_cortex_cog_to_striatum_cog._weights)
        neuralData_y5[currentTrial][:n] = cog_cues_value
        if P:
            neuralData_y_perf[currentTrial] = [np.array(P[-perfBuff:]).mean(),
                                              np.array(A[-perfBuff:]).mean(),
                                              np.array(Regret[-perfBuff:]).mean()]
        neuralData_y_prob[currentTrial] = np.nanmean(probChoice,axis=1)
        neuralData_y_reg[currentTrial] = cumRegret

        entr = 0
        for prob in neuralData_y_prob[currentTrial]:
            entr -= prob * np.log(prob)
        neuralData_y_entr[currentTrial] = entr / len(neuralData_y_prob[currentTrial])

        neuralSignals_reg[0].set_ydata(neuralData_y_reg)
        neuralSignals_2[0].set_ydata(neuralData_y_2)
        neuralSignals_entr[0].set_ydata(neuralData_y_entr)

        if currentTrial>0:
            neuralData_y4_2[currentTrial] = neuralData_y4[currentTrial] - neuralData_y4[currentTrial-1]

        for i in np.arange(nData4):
            neuralSignals4[i].set_ydata(neuralData_y4[:,i])
            neuralSignals4_2[i].set_ydata(neuralData_y4_2[:,i])
        for i in np.arange(nData5-2):
            neuralSignals5[i].set_ydata(neuralData_y5[:,i])
        for i in np.arange(nDataPer):
            neuralSignals_per[i].set_ydata(neuralData_y_perf[:,i])
        for i in np.arange(nDataProb):
            neuralSignals_prob[i].set_ydata(neuralData_y_prob[:,i])
        
    
    @clock.every(plotSteps)
    def plotNeural_data(t):
        if flashPlots>0 and (currentTrial % flashPlots) != 0:
            return
        index = int(round(t/plotSteps))
        neuralData_y[index] = np.hstack(([Striatum_cog['V'][i][0] for i in range(n)],Striatum_mot['V'][0][:]))
        neuralData_y2[index] = np.hstack(([Cortex_cog['V'][i][0] for i in range(n)],Cortex_mot['V'][0][:]))
        # neuralData_ysnc[index][0] = SNc_dop['V']
        # neuralData_y3[index][1] = -SNc_dop['SNc_h']
        # neuralData_y3[index][2] = SNc_dop['D2_IPSC']/5.0
        neuralData_yth[index] = np.hstack(([Thalamus_cog['V'][i][0] for i in range(n)],Thalamus_mot['V'][0][:]))
        neuralData_ysncpptn[index][0] = SNc_dop['V'] 
        neuralData_ysncpptn[index][1] = SNc_dop['Ir']
        # neuralData_ypptn[index][0] = SNc_dop['Ir']
        # neuralData_ypptn[index][1] = SNc_dop['Irew']
        # neuralData_ypptn[index][2] = SNc_dop['Ie_rew']
        neuralData_y_ctx_ass[index] = [Cortex_ass['V'][i][j] for j in range(n) for i in range(n)]

        neuralData_yr1[index] = [Striatum_ass['V'][i][j] for j in range(n) for i in range(n)]
        neuralData_yr2[index] = np.hstack(([GPi_cog['V'][i][0] for i in range(n)],GPi_mot['V'][0][:]))
        neuralData_yr3[index] = np.hstack(([STN_cog['V'][i][0] for i in range(n)],STN_mot['V'][0][:]))

        for i in np.arange(nData):
            neuralSignals[i].set_ydata(neuralData_y[:,i])
        for i in np.arange(nData2):
            neuralSignals2[i].set_ydata(neuralData_y2[:,i])
            neuralSignalsTh[i].set_ydata(neuralData_yth[:,i])
        # for i in np.arange(nDataSnc):
        #     neuralSignalsSnc[i].set_ydata(neuralData_ysnc[:,i])
        # for i in np.arange(nDataPptn):
        #     neuralSignalsPptn[i].set_ydata(neuralData_ypptn[:,i])
        for i in np.arange(nDataSnc+nDataPptn):
            neuralSignalsSncPPTN[i].set_ydata(neuralData_ysncpptn[:,i])
        for i in np.arange(nrData):
            neuralSignalsCtx_ass[i].set_ydata(neuralData_y_ctx_ass[:,i])
            neuralSignalsr1[i].set_ydata(neuralData_yr1[:,i])
        for i in np.arange(nrData2):
            neuralSignalsr2[i].set_ydata(neuralData_yr2[:,i])
        for i in np.arange(nrData3):
            neuralSignalsr3[i].set_ydata(neuralData_yr3[:,i])

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

    if len(invertAt) and (currentTrial+1) in invertAt:
          cues_reward = cues_rewards[np.sum(invertAt<=(currentTrial+1))]

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
                print "Failed trial!! presenting cues:"
                print cues_cog if garivierMoulines else cues_cog[:2]
            failedTrials += 1
            continuousFailedTrials += 1

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
        printData()

if STORE_DATA and forceSelection and failedTrials >= nbFailedTrials:
    fileError = filename.replace('.csv','_error')
    with open(fileError,'a') as f:
        f.write("\n")

if neuralPlot:
    plt.ioff()
    plt.show(block=False)