from dana import *
from learningOptions import *
from datetime import datetime       # Added to create a store file with time data
from socket import gethostname      # Added to specify different output folders
                                    # deppending on the computer where the code is
                                    # running
from collections import OrderedDict
import csv

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
# gamma_DAbySuccess = 6 # [sp/s]
alpha_SuccessEMA   = .8
gamma_DA_LTD      = 0.025 # (1+gamma_DA_LTD * DA) -> 1.8 (4.0) - 2.2 (6.0) - 2.6 (8.0)
gamma_mGluR_LTD   = 0.01 # 60 * (1+DA) -> 60 * 5 - 60*9 -> 300-700
gamma_eCB1_LTD    = 0.1 # (1-20)

# Noise level (%)
Cortex_N        =   0.01
Striatum_N      =   0.01
Striatum_corr_N =   N_factor * 0.05
STN_N           =   0.01
GPi_N           =   0.03
Thalamus_N      =   0.01
SNc_N           =   0.01
SNc_PPTN_N      =   0.00
# if Weights_N:
#     Weights_N *= aux_X
# Cortex_N        =   0.01
# Striatum_N      =   0.01
# Striatum_corr_N =   0.1
# STN_N           =   0.01
# GPi_N           =   0.03
# Thalamus_N      =   0.01
# SNc_N           =   0.1

# DA reward sensitivity
SNc_dop_base_sensitivity = 0.5
# Positive and negative neurons (%)
SNc_dop_pos_perc = 0.1
SNc_dop_neg_perc = 0.1

# Neural population size
SNc_neurons     = DA_neurons
SNc_pos_neurons = np.floor(SNc_dop_pos_perc * SNc_neurons)
SNc_neg_neurons = np.floor(SNc_dop_neg_perc * SNc_neurons)
SNc_deltaDA     = 0.2

# DA buffer for a time delay on arD2
DA_buffSize = int(round(arD2_lag/dt))
DA_buff = np.zeros((DA_buffSize,))
DA_buffIndex = 0

# Sigmoid parameter
Vmin       =  0.0
Vmax       = 20.0
Vh0         = 18.18 # 18.38
Vc         =  6.0

# Tonic DA sigmoid parameters
tonicDA_h = 0.9
tonicDA_c = 0.1

# Learning parameters
decision_threshold = 30
alpha_c     = 0.2  # 0.05
Wmin, Wmax = 0.45, 0.55
# Wmin, Wmax = 0.4, 0.6

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
if minSmoothA != 0:
    file_base += 'msDA'+str(minSmoothA)+'_'
if SNc_neurons != 1:
    file_base += 'nDA'+str(SNc_neurons)+'_'

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
    DATA = OrderedDict([('SNc',''),('SNc_h',''),('P-buff',0),('choice',0),('nchoice',0),('motTime',0),('weights',''),('mot_weights',''),('cogTime',0),
                        ('failedTrials',0),('persistence',0),('values',''),('P',0),('P-3',0),('R',0),('R-buff',0),('pA',0),
                        ('P-3',0),('LTD-LTP',''),('A',0),('A-buff',0),('pchoice',''),('Regret',0),('cumRegret',0),('c_rewards',''),('STR_DA',0)])
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

usePerception = True


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


cues_mot = np.arange(n)
cues_cog = np.arange(n)
labels = ['Cog'+str(i) for i in cues_cog]+['Mot'+str(i) for i in cues_mot]
labels_plusOne = ['Cog-'+str(i+1) for i in cues_cog]+['Mot-'+str(i+1) for i in cues_mot]
ctxLabels = labels_plusOne

cogDecision = False
motDecision = False
cogDecisionTime = 3500.0
motDecisionTime = 3500.0

if zeroValues:
    cog_cues_value = np.zeros(n)
else:
    cog_cues_value = np.ones(n) * 0.5
mot_cues_value = np.ones(n) * 0.5

currentTrial = 0
P, R, A, Regret = [], [], [], []
cumRegret = 0