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
# python learning.py -t 120 -r100 -b 12 -R --ltd-constant --correlatedNoise --relativeValue --zeroValues -P --flashPlots --storePlots 10 --debug
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
from socket import gethostname      # Added to specify different output folders
                                    # deppending on the computer where the code is
                                    # running
from learningOptions import *       # Added argument options parser

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
cues_amplitude = 16

# Sigmoid parameter
Vmin       =  0.0
Vmax       = 20.0
Vh         = 18.5
Vc         =  3.0

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
STR_N_tau    = 0.03
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
gamma_DAstrenght  = .5#.25
gamma_DAbySuccess = 4 # [sp/s]
alpha_SucessEMA   = .8
gamma_DA_LTD      = 0.025 # (1+gamma_DA_LTD * DA) -> 1.8 (4.0) - 2.2 (6.0) - 2.6 (8.0)
gamma_mGluR_LTD   = 0.01 # 60 * (1+DA) -> 60 * 5 - 60*9 -> 300-700
gamma_eCB1_LTD    = 0.1 # (1-20)

# Noise level (%)
Cortex_N        =   0.01   # * aux_X
Striatum_N      =   0.01   # * aux_X
Striatum_corr_N =   0.1    # * aux_Y
STN_N           =   0.01   # * aux_X
GPi_N           =   0.03   # * (aux_X/2.0)
Thalamus_N      =   0.01   # * aux_X
SNc_N           =   0.1    # * aux_Y
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

# Learning parameters
decision_threshold = 40
alpha_c     = 0.2  # 0.05
Wmin, Wmax = 0.45, 0.6
#Wmin, Wmax = 0.4, 0.6

# Reward extension [ms]
delayReward = 300 * millisecond
reward_ext = 50 * millisecond + delayReward

# Data saving configuration
avoidUnsavedPlots = True
# STORE_DATA = True
STORE_FAILED = False
FAILED_FILE = folder+'failed_'+str(DA)

file_base =  'DA_'+str(SNc_N)
if useCorrelatedNoise:
    file_base += 'c_'
if constantLTD:
    file_base += 'cLTD_' #+ str(constantLTD)
if dynamicDA:
    file_base += 'd_'
if invertAt:
    file_base += 'r'+str(invertAt)+'_'

if folder:
    file_base = folder+file_base

if STORE_DATA or STORE_FAILED:
    
    if 'corcovado' == gethostname():
        if not folder:
            filename = '/mnt/Data/Cristobal/tonicDA_effects/connections/Exp/'+file_base+'__'+datetime.now().strftime("%Y%m%d")+'_'+nFile
        else:
            filename = file_base+str(DA)+'_'+str(alpha_LTP)+'_'+str(alpha_LTD)+'_'+nFile
    elif 'lhotse' == gethostname():
        if not folder:
            filename = './Exp/'+file_base+str(DA)+'__'+datetime.now().strftime("%Y%m%d")+'_'+nFile
        else:
            filename = file_base+'_'+str(alpha_LTP)+'_'+str(alpha_LTD)+'_'+str(DA)+'_'+nFile
    else:
        filename = file_base+str(DA)+'_'+str(alpha_LTP)+'_'+str(alpha_LTD)+'_'+nFile

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

# Use a regular pattern for the cues selection
# regPattern = False
nbTrialsPerPattern = 100
pattern = np.array([[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]])

# Enable debug messages
# DA = 1.0
# doPrint = False
# invertAt = 100


# Initialization of the random generator
if randomInit:
    np.random.seed(randomInit)

# Helper functions
# -----------------------------------------------------------------------------
def sigmoid(V,Vmin=Vmin,Vmax=Vmax,Vh=Vh,Vc=Vc):
    return  Vmin + (Vmax-Vmin)/(1.0+np.exp((Vh-V)/Vc))

# def rewardShape(t,rewardTime,dev=0.1):
#     return np.exp(-(t-(rewardTime+3*dev))**2/(2*dev**2))

def strThreshold(DA):
    return (gamma_DAstrenght * DA + 1)*Vh # Factor was 19, but considering gamma_DAstrenght, dropped to 4.75 - testing 5.5
    # return 4.75 * DA + Vh # Factor was 19, but considering gamma_DAstrenght, dropped to 4.75 - testing 5.5

def strSlope(DA):
    return 6.0 #* (1.0 + gamma_DAstrenght * DA)

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
    SNc_dop['u_DA'] = -SNc_dop['SNc_h']
    SNc_dop['fDA_del'] = -SNc_dop['SNc_h']
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
    # print "Past Cognitive weights: ", np.diag(pastW)
    # print "Diff Cognitive weights: ", dw
    pastW = np.copy(W._weights)
    # print cogWDiff

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
    neuralData_y3.fill(None)
    neuralData_y3_2.fill(None)
    ax.set_xlim(0,duration)
    ax2.set_xlim(0,duration)
    ax3.set_xlim(0,duration)
    ax3_2.set_xlim(0,duration)

def clip(V, Vmin, Vmax):
    return np.minimum(np.maximum(V, Vmin), Vmax)

def positiveClip(V):
    return np.maximum(V, 0.0)

def printData():
    # if applyInMotorLoop:
    #     with open(filename,'a') as f:
    #         f.write("%f\t%f\t%f\t%d\t%d\t%d\t%s\t%s\t%d\n" % (DA, np.array(P).mean(), np.array(R).mean(), choice, nchoice, motDecisionTime,
    #             '\t'.join(['{:.2f}'.format(i) for i in W_cortex_cog_to_striatum_cog.weights.diagonal()]),
    #             '\t'.join(['{:.2f}'.format(i) for i in W_cortex_mot_to_striatum_mot.weights.diagonal()]),cogDecisionTime))
    # else:
    with open(filename,'a') as f:
        finalActivity = np.max(Cortex_cog['V']) 
        persistence = 1 if finalActivity > 30 else 0
        f.write("%f\t%f\t%d\t%d\t%d\t%s\t%d\t%d\t%d\t%s\t%f\t%f\t%f\t%f\t%s\t%f\t%f\n" % (SNc_dop['V'], np.array(P[-perfBuff:]).mean(), choice, nchoice, motDecisionTime,
            '\t'.join(['{:.5f}'.format(i) for i in W_cortex_cog_to_striatum_cog.weights.diagonal()]),cogDecisionTime,
            failedTrials,persistence,'\t'.join(['{:.5f}'.format(i) for i in cog_cues_value]),P[-1:][0],np.array(P[-3:]).mean(),R[-1:][0], np.array(R[-perfBuff:]).mean(),
            '\t'.join(['{:.5f}'.format(float(i)) for i in trialLtdLtp.reshape((n*4,1))]), A[-1:][0], np.array(A[-perfBuff:]).mean()) )

def D2_IPSC_kernel(t,t_DelRew):
    if t < t_DelRew:
        return 0
    return np.exp(-(t-t_DelRew)/arD2_tau)

def convolv_D2Kernel(t,currentValue,input):
    return currentValue + input * D2_IPSC_kernel(t,motDecisionTime/1000+arD2_lag)

SNc_dop   = zeros((1,), """  D2_IPSC = - alpha_DA_arD2 * DAtoD2c;
                             Ir = np.maximum(Irew, Ie_rew);
                             I = Ir + D2_IPSC;
                             n = correlatedNoise(I,n,SNc_N,alpha_Rew_DA,SNc_N_tau);
                             It = I + n;
                             du_DA/dt = (-u_DA + (It - SNc_h))/SNc_tau;
                             V = positiveClip(u_DA); Irew; Ie_rew; SNc_h; fDA_del; DAtoD2c""")

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

Striatum_cog = zeros((n,1), """Is = I*(1 + gamma_DAstrenght*DA);
                             n = striatalNoise(Is,n);
                             It = Is + n;
                             Vh = strThreshold(DA);
                             Vc = strSlope(DA);
                             u = sigmoid(It - Striatum_h,Vmin,Vmax,Vh,Vc);
                             dV/dt = (-V + u)/Striatum_tau; I; DA""")

Striatum_mot = zeros((1,n), """Is = I*(1 + gamma_DAstrenght*DA);
                             n = striatalNoise(Is,n);
                             It = Is + n;
                             Vh = strThreshold(DA);
                             Vc = strSlope(DA);
                             u = sigmoid(It - Striatum_h,Vmin,Vmax,Vh,Vc);
                             dV/dt = (-V + u)/Striatum_tau; I; DA""")

Striatum_ass = zeros((n,n), """Is = I*(1 + gamma_DAstrenght*DA);
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

cues_mot = np.array([0,1,2,3])
cues_cog = np.array([0,1,2,3])
if zeroValues:
    cog_cues_value = np.zeros(n)
else:
    cog_cues_value = np.ones(4) * 0.5
mot_cues_value = np.ones(4) * 0.5
cues_reward = np.array([3.0,2.0,1.0,0.0])/3.0


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

W = DenseConnection( Cortex_mot('V'),   Striatum_mot('I'), 1.0)#0.9) #1.0)
if motInitialWeights != None:
    W._weights = motInitialWeights
else:
    init_weights(W)
W_cortex_mot_to_striatum_mot = W

W = DenseConnection( Cortex_ass('V'),   Striatum_ass('I'), 1.0)
init_weights(W)
W = DenseConnection( Cortex_cog('V'),   Striatum_ass('I'), np.ones((1,2*n+1)))
init_weights(W,0.2)
W = DenseConnection( Cortex_mot('V'),   Striatum_ass('I'), np.ones((2*n+1,1)))
init_weights(W,0.2)
DenseConnection( Cortex_cog('V'),   STN_cog('I'),       1.0 )#1.0 )
DenseConnection( Cortex_mot('V'),   STN_mot('I'),       1.0 )#1.0 )
DenseConnection( Striatum_cog('V'), GPi_cog('I'),      -2.0 )
DenseConnection( Striatum_mot('V'), GPi_mot('I'),      -2.0 )
DenseConnection( Striatum_ass('V'), GPi_cog('I'),      -2.0*np.ones((1,2*n+1)))
DenseConnection( Striatum_ass('V'), GPi_mot('I'),      -2.0*np.ones((2*n+1,1)))
DenseConnection( STN_cog('V'),      GPi_cog('I'),       (4.0/n)*np.ones((2*n+1,1)))
DenseConnection( STN_mot('V'),      GPi_mot('I'),       (4.0/n)*np.ones((1,2*n+1)))
DenseConnection( GPi_cog('V'),      Thalamus_cog('I'), -0.5 )
DenseConnection( GPi_mot('V'),      Thalamus_mot('I'), -0.5 )
DenseConnection( Thalamus_cog('V'), Cortex_cog('I'),    1.0 )
DenseConnection( Thalamus_mot('V'), Cortex_mot('I'),    1.0 )
DenseConnection( Cortex_cog('V'),   Thalamus_cog('I'),  0.4 )
DenseConnection( Cortex_mot('V'),   Thalamus_mot('I'),  0.4 )

# To be fixed!
# DenseConnection( SNc_dop('fDA'),   Striatum_cog('DA'),       1.0 )
# DenseConnection( SNc_dop('fDA'),   Striatum_mot('DA'),       1.0 )
# DenseConnection( SNc_dop('fDA'),   Striatum_ass('DA'),       1.0 )

# Initial DA levels
# -----------------------------------------------------------------------------
# Unnecessary now, DA comes from SNc
# setDAlevels(DA)

# Trial setup
# -----------------------------------------------------------------------------
inputCurrents_noise = np.zeros(6)
c1,c2,m1,m2 = 0,0,0,0
SNc_dop['SNc_h'] = SNc_h_base

@clock.at(500*millisecond)
def set_trial(t):
    global cues_cog, cogDecision, cogDecisionTime, motDecision, motDecisionTime, c1, c2, m1, m2, inputCurrents_noise, flag
    global daD, corticalLegend

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

    inputCurrents_noise = np.random.normal(0,v*Cortex_N,6)
    
    cogDecision = False
    cogDecisionTime = 3500.0
    motDecision = False
    motDecisionTime = 3500.0

    if neuralPlot:
        texts = ('Cog1','Cog2','Cog3','Cog4','Mot1','Mot2','Mot3','Mot4')
        t_i = 0
        for t in corticalLegend.get_texts():
            if t_i < 4 and c1 == t_i:
                t.set_text(texts[t_i]+' *')
            elif t_i < 4 and c2 == t_i:
                t.set_text(texts[t_i]+' ^')
            elif t_i >= 4 and m1 == t_i-4:
                t.set_text(texts[t_i]+' *')
            elif t_i >= 4 and m2 == t_i-4:
                t.set_text(texts[t_i]+' ^')
            else:
                t.set_text(texts[t_i])
            t_i += 1


@before(clock.tick)
def computeSoftInput(t):
    if motDecisionTime < 3500:
        inputsOffAt =  motDecisionTime/1000+delayReward+0.2
    else:
        inputsOffAt = 3.2
    v = sigmoid(t,0,cues_amplitude,.725,.042) - sigmoid(t,0,cues_amplitude,inputsOffAt,.084)

    Cortex_mot['Iext'][0,m1]  = v + inputCurrents_noise[0]
    Cortex_mot['Iext'][0,m2]  = v + inputCurrents_noise[1]
    Cortex_cog['Iext'][c1,0]  = v + inputCurrents_noise[2]
    Cortex_cog['Iext'][c2,0]  = v + inputCurrents_noise[3]
    Cortex_ass['Iext'][c1,m1] = v + inputCurrents_noise[4]
    Cortex_ass['Iext'][c2,m2] = v + inputCurrents_noise[5]  

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
    relativeFiringRate = SNc_dop['fDA_del']-baseActivity
    SNc_dop['DAtoD2c'] = positiveClip(convolv_D2Kernel(t,SNc_dop['DAtoD2c'],relativeFiringRate))

@before(clock.tick)
def fillDopamineBuffer(t):
    global DA_buffIndex
    buffPos = DA_buffIndex % DA_buffSize
    DA_buff[buffPos] = SNc_dop['V']
    SNc_dop['fDA_del'] = DA_buff[(buffPos+1) % DA_buffSize]
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

@before(clock.tick)
def propagateDA(t): # This should be implemented as a connection! but for quick testing...
    setDAlevels(SNc_dop['V'])

@after(clock.tick)
def earlyStop(t):
    global currentTrial, continuousFailedTrials
    if stopAfterSelecion and motDecisionTime < 3500:
        if t > motDecisionTime / 1000 + afterSelectionTime:
            if neuralPlot:
                resetPlot()
                if storePlotsEvery > 0 and (currentTrial % storePlotsEvery) == 0:
                    ax.set_xlim((0,t))
                    ax2.set_xlim((0,t))
                    ax3.set_xlim((0,t))
                    plt.savefig(plotsFolder+str(DA)+'_'+str(alpha_LTP)+'_'+str(alpha_LTD)+'_neuralActivity_'+str(currentTrial)+".pdf",bbox_inches='tight')
                    plt.savefig(plotsFolder+str(DA)+'_'+str(alpha_LTP)+'_'+str(alpha_LTD)+'_neuralActivity_'+str(currentTrial)+".png",bbox_inches='tight')
            if STORE_DATA:
                printData()
            partialReset()
            clock.reset()
            currentTrial += 1
            continuousFailedTrials = 0

# Learning
# -----------------------------------------------------------------------------
P, R, A = [], [], []
smoothR = 0

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
            # As Shawn et al. (2008) shown that D2 sped LTD but not blocked it, SNc_dop['V'] is introduced as a weighted additive term.
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

@after(clock.tick)
def register(t):
    global currentTrial, continuousFailedTrials, selectionError, cogDecision
    global cogDecisionTime, motDecision, motDecisionTime
    global choice, nchoice, mchoice, pError, smoothR, pastW, cog_cues_value
    
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
    c1, c2 = cues_cog[:2]
    m1, m2 = cues_mot[:2]
    mot_choice = np.argmax(Cortex_mot['V'])
    cog_choice = np.argmax(Cortex_cog['V'])

    # The motor selection is the executed one, then it
    # defines the selected cue in a cognitive domain.
    # The actual cognitive selection might differ.
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
            print "! Failed trial: selected a non-presented cue (not ",m1,"nor",m2,"but",motor_choice,")"
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
    advantage = 1 + (cues_reward[choice] - np.max([cues_reward[choice],cues_reward[nchoice]]))
    A.append(advantage)

    W = W_cortex_cog_to_striatum_cog
    Wm = W_cortex_mot_to_striatum_mot
    if learn:
        # Compute reward
        reward = np.random.uniform(0,1) < cues_reward[choice]
        R.append(reward)
        smoothR = alpha_SucessEMA * smoothR + (1-alpha_SucessEMA) * reward
        if dynamicDA:
            SNc_dop['SNc_h'] = SNc_h_base - gamma_DAbySuccess * smoothR

        # Compute prediction error
        ## Error regulates PPNg activity, the prediction error encoding signal
        if relativeValue:
            pError = reward - (cog_cues_value[choice]+cog_cues_value[nchoice])/2.0
        else:
            pError = reward - cog_cues_value[choice]
        # Update cues values
        cog_cues_value[choice] += pError* alpha_c
        # Constant depression over values
        # cog_cues_value = clip(cog_cues_value - alpha_c /20.0,0,1)

        if oneStepConstantLTD:
            W_cortex_cog_to_striatum_cog._weights -= alpha_LTD

    if doPrint:
        # Just for displaying ordered cue
        oc1,oc2 = min(c1,c2), max(c1,c2)
        if cues_reward[choice]>cues_reward[nchoice]:
            print "Choice:          [%d] / %d  (good)" % (choice,nchoice)
        else:
            print "Choice:           %d / [%d] (bad)" % (nchoice,choice)
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
    global fig, ax, ax2, ax3, neuralData_y_2, neuralData_y2, neuralData_y3, neuralData_y4, neuralData_y5, corticalLegend, cogMarker, motMarker, movMarker
    cogMarker = motMarker = movMarker = []
    plotSteps = 10*millisecond
    nData = 4
    nData2 = 8
    nData3 = 3
    nData3_2 = 3
    nData4 = 4
    nData4_2 = nData4
    nData5 = 6 # values plus choiced and non-choiced cues
    fig, (ax,ax2,ax3,axw,axv) = plt.subplots(5,1,figsize=(10,10),num="DA: "+str(DA))#+" X_"+str(aux_X)+" Y_"+str(aux_Y))
    ax.set_xlim(0,duration)
    ax.set_ylim(-2,25)
    ax.set_title('Cognitive striatal activity', fontsize=10)
    ax_2 = ax.twiny()
    ax_2.set_xlim(0,nbTrials)
    ax_2.tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=False)
    ax_2.set_xticklabels([])
    ax2.set_xlim(0,duration)
    ax2.set_ylim(-2,100)
    ax2.set_title('Cortical activity', fontsize=10)
    ax3.set_xlim(0,duration)
    ax3.set_ylim(-10,20)
    ax3.set_title('SNc signals', fontsize=10)
    ax3_2 = ax3.twinx()
    ax3_2.set_xlim(0,duration)
    ax3_2.set_ylim(-20,20)
    neuralData_x = np.arange(0,duration+plotSteps,plotSteps)
    neuralData_y = np.full((len(neuralData_x),nData),None,dtype=float)
    neuralData_y2 = np.full((len(neuralData_x),nData2),None,dtype=float)
    neuralData_y3 = np.full((len(neuralData_x),nData3),None,dtype=float)
    neuralData_y3_2 = np.full((len(neuralData_x),nData3_2),None,dtype=float)
    neuralSignals = ax.plot(neuralData_x,neuralData_y, alpha=0.7)
    neuralSignals2 = ax2.plot(neuralData_x,neuralData_y2, alpha=0.7)
    neuralSignals3 = ax3.plot(neuralData_x,neuralData_y3, alpha=0.7)
    neuralSignals3_2 = ax3_2.plot(neuralData_x,neuralData_y3_2, alpha=0.3,linewidth=3)
    for l in range(4):
        plt.setp(neuralSignals2[l+4],'color',plt.getp(neuralSignals2[l],'color'),'ls','--')
    for l in range(2):
        plt.setp(neuralSignals3_2[l+1],'ls',':')

    axv.set_xlim(0,nbTrials)
    axv.set_ylim(0,1)
    axv.set_title('Learned cognitive values', fontsize=10)
    axw.set_xlim(0,nbTrials)
    axw.set_ylim(0.39,.61)
    axw.set_title('Cognitive corticostriatal weights', fontsize=10)
    axdw = axw.twinx()
    axdw.set_ylim(-.02,.06)
    axdw.set_xlim(0,nbTrials)
    axdw.axhline(0, color='black')
    neuralData_x2 = np.arange(nbTrials)
    neuralData_y_2 = np.full((len(neuralData_x2),1),None,dtype=float)
    neuralData_y4 = np.full((len(neuralData_x2),nData4),None,dtype=float)
    neuralData_y4_2 = np.full((len(neuralData_x2),nData4_2),None,dtype=float)
    neuralData_y5 = np.full((len(neuralData_x2),nData5),None,dtype=float)
    neuralSignals_2 = ax_2.plot(neuralData_x2,neuralData_y_2,alpha=0.3,color='magenta',linewidth=3)
    neuralSignals4 = axw.plot(neuralData_x2,neuralData_y4, alpha=0.7)
    neuralSignals4_2 = axdw.plot(neuralData_x2,neuralData_y4_2, alpha=0.3,linewidth=5)
    neuralSignals5 = axv.plot(neuralData_x2,neuralData_y5, alpha=0.7)

    plt.setp(neuralSignals5[4:], 'marker', 'D','linestyle','','alpha',0.2,'markersize',5)
    plt.setp(neuralSignals5[4], 'color','g')
    for axes in (ax,ax2,ax3,ax3_2,axv,axw,axdw):
        axes.tick_params(axis='both', which='major', labelsize=10)

    ax.legend(neuralSignals,['Cog1','Cog2','Cog3','Cog4'],loc='right', fontsize='x-small', bbox_to_anchor= (1.1, 0.5), 
            borderaxespad=0, frameon=False)
    ax_2.legend(neuralSignals_2,['SNc'],loc='right', fontsize='x-small', bbox_to_anchor= (1.18, 0.5), 
            borderaxespad=0, frameon=False)
    corticalLegend = ax2.legend(neuralSignals2,['Cog1','Cog2','Cog3','Cog4','Mot1','Mot2','Mot3','Mot4'],loc='right', fontsize='x-small',framealpha=0.6,ncol=2, bbox_to_anchor= (1.2, 0.5), 
            borderaxespad=0, frameon=False)
    ax3.legend(neuralSignals3,['DA','tonicDA','I_DA/5'],loc='right', fontsize='x-small',framealpha=0.6,ncol=1, bbox_to_anchor= (1.12, 0.5), 
            borderaxespad=0, frameon=False)
    ax3_2.legend(neuralSignals3_2,['I_r','I_rew','Ie_rew'],loc='right', fontsize='x-small',framealpha=0.6,ncol=1, bbox_to_anchor= (1.2, 0.5), 
            borderaxespad=0, frameon=False)
    axw.legend(neuralSignals4,['Wcog[1]','Wcog[2]','Wcog[3]','Wcog[4]'],loc='right', fontsize='x-small',framealpha=0.6,ncol=1, bbox_to_anchor= (1.14, 0.5), 
            borderaxespad=0, frameon=False)
    axv.legend(neuralSignals5,['Vcog[1]','Vcog[2]','Vcog[3]','Vcog[4]','Chosen cue','Other cue'],loc='right', fontsize='x-small',framealpha=0.6,ncol=1, bbox_to_anchor= (1.15, 0.5), 
            borderaxespad=0, frameon=False)

    plt.tight_layout()

    def drawPlots():
        fig.canvas.draw()
        fig.canvas.flush_events()

    @after(clock.tick)
    def plotDecisionMarkers(t):
        global cogMarker, motMarker, movMarker
        if len(cogMarker) == 0 and cogDecisionTime/1000 < t:
            cogMarker = ax2.plot(t,40,marker='^',alpha=0.4,markersize=10,color='r',linestyle=':')
        if len(motMarker) == 0 and motDecisionTime/1000 < t:
            motMarker = ax2.plot(t,40,marker='^',alpha=0.4,markersize=10,color='b',linestyle=':')
        if len(movMarker) == 0 and motDecisionTime/1000 + delayReward < t:
            movMarker = ax2.plot(t,40,marker='.',alpha=0.4,markersize=20,color='g',linestyle=':')

    @clock.at(dt)
    def plotTrial_data(t):
        neuralData_y_2[currentTrial] = SNc_dop['V']
        neuralData_y4[currentTrial] = np.diag(W_cortex_cog_to_striatum_cog._weights)
        neuralData_y5[currentTrial][:4] = cog_cues_value

        neuralSignals_2[0].set_ydata(neuralData_y_2)

        if currentTrial>0:
            neuralData_y4_2[currentTrial] = neuralData_y4[currentTrial] - neuralData_y4[currentTrial-1]
            neuralData_y5[currentTrial-1][4] = (3-choice)/3.0
            neuralData_y5[currentTrial-1][5] = (3-nchoice)/3.0

        for i in np.arange(nData4):
            neuralSignals4[i].set_ydata(neuralData_y4[:,i])
            neuralSignals4_2[i].set_ydata(neuralData_y4_2[:,i])
        for i in np.arange(nData5):
            neuralSignals5[i].set_ydata(neuralData_y5[:,i])
    
    @clock.every(plotSteps)
    def plotNeural_data(t):
        if flashPlots>0 and (currentTrial % flashPlots) != 0:
            return
        index = int(round(t/plotSteps))
        neuralData_y[index][0] = Striatum_cog['V'][0]
        neuralData_y[index][1] = Striatum_cog['V'][1]
        neuralData_y[index][2] = Striatum_cog['V'][2]
        neuralData_y[index][3] = Striatum_cog['V'][3]
        neuralData_y2[index][0] = Cortex_cog['V'][0]
        neuralData_y2[index][1] = Cortex_cog['V'][1]
        neuralData_y2[index][2] = Cortex_cog['V'][2]
        neuralData_y2[index][3] = Cortex_cog['V'][3]
        neuralData_y2[index][4] = Cortex_mot['V'][0][0]
        neuralData_y2[index][5] = Cortex_mot['V'][0][1]
        neuralData_y2[index][6] = Cortex_mot['V'][0][2]
        neuralData_y2[index][7] = Cortex_mot['V'][0][3]
        neuralData_y3[index][0] = SNc_dop['V']
        neuralData_y3[index][1] = -SNc_dop['SNc_h']
        neuralData_y3[index][2] = SNc_dop['D2_IPSC']/5.0
        neuralData_y3_2[index][0] = SNc_dop['Ir']
        neuralData_y3_2[index][1] = SNc_dop['Irew']
        neuralData_y3_2[index][2] = SNc_dop['Ie_rew']

        for i in np.arange(nData):
            neuralSignals[i].set_ydata(neuralData_y[:,i])
        for i in np.arange(nData2):
            neuralSignals2[i].set_ydata(neuralData_y2[:,i])
        for i in np.arange(nData3):
            neuralSignals3[i].set_ydata(neuralData_y3[:,i])
        for i in np.arange(nData3_2):
            neuralSignals3_2[i].set_ydata(neuralData_y3_2[:,i])
        # print SNc_dop['V']
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

    if invertAt and currentTrial == invertAt:
          cues_reward = np.flipud(cues_reward)
          invertAt = 0
    run(time=duration, dt=dt)

    if clock.time >= duration:
        # If the 'completeTrial' flag is 'True' or there was no selection
        # during the experiment, the current trial's lenght is 'duration' seconds

        # The model is set to its initial state
        # reset()
        if neuralPlot:
            if storePlotsEvery > 0 and (currentTrial % storePlotsEvery) == 0:
                plt.savefig(plotsFolder+str(DA)+'_'+str(alpha_LTP)+'_'+str(alpha_LTD)+'_neuralActivity_'+str(currentTrial)+".pdf",bbox_inches='tight')
                plt.savefig(plotsFolder+str(DA)+'_'+str(alpha_LTP)+'_'+str(alpha_LTD)+'_neuralActivity_'+str(currentTrial)+".png",bbox_inches='tight')
            if currentTrial < nbTrials-1:
                resetPlot()
        if not motDecision:
            # Detection and counting of failed trials
            if doPrint:
                print "Failed trial!! ",cues_cog[:2]
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
        if STORE_DATA:
            if applyInMotorLoop:
                st =  '\t'.join(['{:.0f}'.format(-1) for i in range(2*n)])
            else:
                st = '\t'.join(['{:.0f}'.format(-1) for i in range(n)])
            f.write("%d\t%d\t%d\t%d\t%d\t%d\t%s\n" % (currentTrial,failedTrials,selectionError,-1,-1,-1,
                st))
        else:
            f.write("%d\t%d\t%d\n" % (currentTrial,failedTrials,selectionError))

if STORE_DATA and forceSelection and failedTrials >= nbFailedTrials:
    filename += '_error'
    with open(filename,'a') as f:
        f.write("\n")

if neuralPlot:
    plt.ioff()
    plt.show(block=False)