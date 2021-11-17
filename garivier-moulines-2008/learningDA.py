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

from learningOptions import *       # Added argument options parser
from learningHardCodedOptions import *       # Added argument options parser
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
    DA_buff[:] = -SNc_dop['SNc_h'][0]

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

def getUnitData(neuralPopVar): # Example: neuralPopVar = SNc_dop['V']
    return neuralPopVar

def fillData():
    DATA['SNc']          = '\t'.join(['{:.5f}'.format(SNc_dop['V'][i][0]) for i in range(SNc_neurons)])
    DATA['SNc_h']        = '\t'.join(['{:.5f}'.format(SNc_dop['SNc_h'][i][0]) for i in range(SNc_neurons)])
    DATA['P-buff']       = np.array(P[-perfBuff:]).mean()
    DATA['choice']       = choice
    DATA['nchoice']      = nchoice if not garivierMoulines else -1
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
    DATA['STR_DA']       = '\t'.join(['{:.5f}'.format(Striatum_cog['DA'][i][0]) for i in cues_cog])

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

SNc_dop_sensitivity = np.zeros(SNc_neurons)+SNc_dop_base_sensitivity

###### Importing neural populations and connectivity
from learningNeuralPopulations import *
#################

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
    for i in range(SNc_neurons):
        DA_buff[buffPos] = SNc_dop['V'][i]
        SNc_dop['V_lag'][i] = DA_buff[(buffPos+1) % DA_buffSize]
    DA_buffIndex += 1

@before(clock.tick)
def deliverReward(t):
    if motDecisionTime/1000 + delayReward < t:
        if t < motDecisionTime/1000 + reward_ext and SNc_dop['Irew'][0] == 0:
            for i in range(SNc_neurons):
                SNc_dop[i]['Irew'] = SNc_pError[i] * alpha_Rew_DA
                SNc_dop[i]['Irew'] = SNc_dop[i]['Irew'] + noise(SNc_dop[i]['Irew'],SNc_PPTN_N)
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
smoothR = -1
smoothA = -1
pError = 0.0
SNc_pError = np.zeros(SNc_neurons)
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
        dDA = np.mean([SNc_dop['V'][i]-(-SNc_dop['SNc_h'][i]) for i in range(SNc_neurons)])  # current DA activity w.r.t. base level
        if (constantLTD or oneStepConstantLTD) and dDA < 0:
            return
        if SNc_dop['Ir'][0] > 0 and dDA > 0: # LTP
            alpha = alpha_LTP
        elif SNc_dop['Ir'][0] < 0: # LTD
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
            DA_activity = np.mean(SNc_dop['V'])
            dw = - 0.001 * alpha_LTD * Striatum_cog['It'][pop][0] * (1 + gamma_DA_LTD * DA_activity)
            w = clip(W.weights[pop][pop] + dw,Wmin,Wmax)
            trialLtdLtp[pop,0] += dw
            trialLtdLtp[pop,1] += w - W.weights[pop, pop]
            W.weights[pop][pop] = w

        if applyInMotorLoop:
            W = W_cortex_mot_to_striatum_mot
            DA_activity = np.mean(SNc_dop['V'])
            for pop in range(n):
                dw = - 0.001 * alpha_LTD * Striatum_mot['It'][0][pop] * (1 + gamma_DA_LTD * DA_activity)
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

        # Compute reward
        reward = np.random.uniform(0,1) < cues_reward[choice]
        R.append(reward)

        # How good was the selection compared to the best choice presented
        regret = np.max([cues_reward[choice],cues_reward[nchoice]]) - cues_reward[choice]
        perceived_regret = np.max([cog_cues_value[choice],cog_cues_value[nchoice]]) - reward # cog_cues_value[choice]
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

        # Compute reward
        reward = np.random.uniform(0,1) < cues_reward[choice]
        R.append(reward)

        # How good was the selection compared to the best choice presented
        regret = np.max(cues_reward) - cues_reward[choice]
        perceived_regret = np.max(cog_cues_value) - reward # cog_cues_value[choice]
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
        if smoothR == -1: # first trial and test
            smoothR = reward
            smoothA = perceived_advantage if usePerception else advantage
        else:
            smoothR = alpha_SuccessEMA * smoothR + (1-alpha_SuccessEMA) * reward
            smoothA = alpha_SuccessEMA * smoothA + (1-alpha_SuccessEMA) * (perceived_advantage if usePerception else advantage)
        if dynamicDA:
            if dynamicDAoverA:
                if minSmoothA >= 0:
                    ssmoothA = np.max((smoothA-minSmoothA,0))/(1-minSmoothA)
                    SNc_dop['SNc_h'] = SNc_h_base - gamma_DAbySuccess * ssmoothA
                else:
                    SNc_dop['SNc_h'] = sigmoid(smoothA,Vmin=SNc_h_base,Vmax=SNc_h_base-gamma_DAbySuccess,Vh=tonicDA_h,Vc=-minSmoothA)
            else:
                if minSmoothA >= 0:
                    ssmoothR = np.max((smoothR-minSmoothA,0))/(1-minSmoothA)
                    SNc_dop['SNc_h'] = SNc_h_base - gamma_DAbySuccess * ssmoothR
                else:
                    SNc_dop['SNc_h'] = sigmoid(ssmoothA,Vmin=SNc_h_base,Vmax=SNc_h_base-gamma_DAbySuccess,Vh=tonicDA_h,Vc=-minSmoothA)

        # Compute prediction error
        pError = reward - cog_cues_value[choice]

        # Update cues values
        cog_cues_value[choice] += pError* alpha_c

        ## Error regulates PPNg activity, the prediction error encoding signal
        prediction = cog_cues_value[choice]
        if relativeValue:
            if not garivierMoulines:
                prediction = (cog_cues_value[choice]+cog_cues_value[nchoice])/2.0
            else:
                prediction = np.mean(cog_cues_value)

        for i in range(SNc_neurons):
            if i < SNc_pos_neurons:
                SNc_pError[i] = reward - np.min((1,prediction+SNc_deltaDA))
            elif i < (SNc_pos_neurons+SNc_neg_neurons):
                SNc_pError[i] = reward - np.max((0,prediction-SNc_deltaDA))
            else:
                SNc_pError[i] = reward - prediction


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
    from learningPlottingCode import *


# Simulation
# -----------------------------------------------------------------------------

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