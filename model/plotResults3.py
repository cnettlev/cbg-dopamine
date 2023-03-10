#!/usr/bin/python3
import csv
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import copy
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import wilcoxon, mannwhitneyu
import os
import sys
import pickle
import subprocess
from utils import checkFolder, askFlag

matplotlib.rcParams.update(
    {
        'text.usetex': False,
        'font.family': 'stixgeneral',
        'mathtext.fontset': 'stix',
        'patch.edgecolor' : 'none'
    }
)

# tableau colors
# colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan', 'k']
colors = []
norm = matplotlib.colors.Normalize(vmin=0,vmax=20)
for i in range(20):
    colors += [matplotlib.cm.tab20(norm(i))]
# colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan', 'k']
# plt.style.use('seaborn-dark-palette')

garivierMoulines = False
showFig = False
datafolder = 'data/'#/higherW/'
commonData = datafolder+'higherW/tonicDA/'
figfolder = 'figures/'
figExtensions = ['.png', '.pdf']
dataFilesFolder = 'dataFiles/'

checkFolder(dataFilesFolder)

baseBase = 'DA_c_cLTD_d_' + ('gM_' if garivierMoulines else '')
commonSTDP = '0.0002_3e-05_'
commonNoise = 'X1_Y1__'
commonDA = '4.0'
commonDeltaDA = 'dDA_2.0_'
trialSeeds = True
trialSeedsStart = 0
strippedLines = []
WEIGHTS = ["w0.6_0.566_0.533_0.5__", "w0.55_0.525_0.5_0.475__"] 

runby = sys.argv[1] if len(sys.argv)>1 else 'tonicDA'

if runby == 'tonicDA':
    # DA = ['0.0', '1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0', '10.0']
    DA = ['4.0']
    DA = ['0.0', '1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0', '10.0']
    commonDeltaDA = 'dDA_2.0_'
    alternatives = ['Tonic DA = '+ d for d in DA]
    shortAlt = DA
    strippedLines = []#'0.0', '1.0', '2.0', '3.0', '4.0', '15.0', '16.0']
    clusterName = 'Tonic dopamine'

    if len(alternatives) < 6:
        colors = [colors[(n+2)%len(colors)] for n in range(len(colors))]
        print(colors)

    nameBase = [baseBase + commonDeltaDA  + commonNoise + commonSTDP + tonic for tonic in DA]

elif 'regPatterns' in runby:
    DA = ['0.0', '1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0', '10.0']
    DA = ['2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0']
    alternatives = ['Tonic DA = '+ d for d in DA]
    shortAlt = DA
    strippedLines = ['0.0', '1.0', '7.0', '8.0', '9.0', '10.0']

    fixed = int(runby[-1])
    if len(alternatives) < 6:
        colors = [colors[(n+2)%len(colors)] for n in range(len(colors))]
        print(colors)

    weights = WEIGHTS[fixed]
    clusterName = 'Tonic dopamine\nconsidering '+('1st' if fixed == 0 else '2nd')+' set\nof ctx weights'

    nameBase = [baseBase + 'dDA_0.0_'  + commonNoise[:-1] + weights + commonSTDP + tonic for tonic in DA]

elif 'singlePattern' in runby:
    DA = ['2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0', '10.0', '11.0']
    alternatives = ['Tonic DA = '+ d for d in DA]
    shortAlt = DA
    strippedLines = ['0.0', '1.0', '2.0', '3.0', '4.0', '15.0', '16.0']

    if len(alternatives) < 6:
        colors = [colors[(n+2)%len(colors)] for n in range(len(colors))]
        print(colors)

    weights = WEIGHTS[0]
    clusterName = 'Tonic dopamine considering a fixed set of ctxstr weights'
    pair = runby.split("-")[1]

    nameBase = [baseBase + 'dDA_0.0_'  + commonNoise[:-1] + 'sP' + pair +'_' + weights + commonSTDP + tonic for tonic in DA]

elif runby == 'tonicDAfixed':
    DA = ['2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0', '10.0', '11.0', '12.0', '13.0', '14.0', '15.0', '16.0']
    alternatives = ['Tonic DA = '+ d for d in DA]
    shortAlt = DA
    strippedLines = ['0.0', '1.0', '2.0', '3.0', '4.0', '15.0', '16.0']
    clusterName = 'Fixed tonic dopamine'

    if len(alternatives) < 6:
        colors = [colors[(n+2)%len(colors)] for n in range(len(colors))]
        print(colors)

    nameBase = [baseBase + 'dDA_0.0_'  + commonNoise + commonSTDP + tonic for tonic in DA]

elif runby == 'deltaDA':
    DELTA = ['0.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0']
    DA = ['4.0']*len(DELTA)
    alternatives = ['DeltaDA = '+ delta for delta in DELTA]
    shortAlt = DELTA
    clusterName = 'Range of tonic dopamine'


    nameBase = [baseBase + 'dDA_' + delta + '_' + commonNoise + commonSTDP + commonDA for delta in DELTA]

elif runby == 'noise':
    NOISE = ['X1', 'X10.0', 'X15.0', 'X20.0', 'X25.0', 'X30.0']
    DA = ['4.0']*len(NOISE)
    alternatives = [n.replace('X','n X') for n in NOISE]
    shortAlt = alternatives
    clusterName = 'Overall noise (except striatal inputs)'

    nameBase = [baseBase + commonDeltaDA + noise + '_Y1__' + commonSTDP + commonDA for noise in NOISE]

elif runby == 'noise-str':
    NOISE = ['Y1__' ,'Y1.5__' ,'Y2.0__', 'Y2.5__' ,'Y3.0__' ,'Y3.5__']
    DA = ['4.0']*len(NOISE)
    alternatives = [n.replace('__','').replace('Y','STRn X ') for n in NOISE]
    shortAlt = [n.replace('__','').replace('Y','n x ') for n in NOISE]
    clusterName = 'Noise on striatal inputs'

    nameBase = [baseBase + commonDeltaDA + 'X1_' + noise + commonSTDP + commonDA for noise in NOISE]

elif 'stdp' in runby:
    LTD = ['1e-05', '2e-05', '3e-05', '4e-05', '5e-05']
    LTP = ['0.0001', '0.00015', '0.0002', '0.00025', '0.0003']

    clusterName = 'STDP '
    fixed = int(runby[-1])
    if 'fixltd' in runby:
        STDP = [ltp+'_'+LTD[fixed]+'_' for ltp in LTP]
        clusterName += 'for LTD: '+LTD[fixed]
    else:
        STDP = [LTP[fixed]+'_'+ltd+'_' for ltd in LTD]
        clusterName += 'for LTP: '+LTP[fixed]
    
    DA = ['4.0']*len(STDP)
    alternatives = []
    for stdp in STDP:
        splitted = stdp.split('_')
        alternatives.append('LTP: ' + "{:e}".format(float(splitted[0])) + ' - LTD: ' + "{:e}".format(float(splitted[1])))        
    shortAlt = [a.replace('LT','') for a in alternatives]
    nameBase = [baseBase + commonDeltaDA + commonNoise + stdp + commonDA for stdp in STDP]

else:
    print('Wrong! try again')
    exit(1)

if 'tonicDA' == runby:
    clustBase = ('hW_' if 'higherW' in datafolder else '') + runby
else:
    clustBase = ('hW_2_' if 'higherW' in datafolder else '') + runby
nAlt = len(alternatives)
datafolder += ('stdp' if 'stdp' in runby else ('regPatterns' if 'regPatt' in runby else (runby.split('-')[0] if 'singlePa' in runby else runby) ) )+'/'
nExp     = 80 if garivierMoulines else (20 if 'stdp' in runby else  (30 if 'Pattern' in runby else (15 if 'tonicDAfixed' == runby else 50)))
nTrials  = 1000 if garivierMoulines else  (50 if 'Pattern' in runby else 120)

if garivierMoulines:
    cues_reward  = np.array([0.5,0.4,0.3])
    cues_reward2 = np.array([0.5,0.4,0.9])
    cues_reward3 = np.array([0.5,0.4,0.3])
    cues_rewards = [cues_reward, cues_reward2, cues_reward3]
else:
    cues_reward  = np.array([3.0,2.0,1.0,0.0])/3.0
    cues_reward2 = cues_reward[::-1] # inverted
    cues_rewards = [cues_reward, cues_reward2]

files = []

for nB_i in range(len(nameBase)):
    nB = nameBase[nB_i]
    da = DA[nB_i]

    cluster = nB
    parameters = {'nameBase': nB}
    clusterData = {'fName': cluster, 'data': [], 'files': [], 'params': parameters}
    # clusterData['name'] = ('Dynamic $DA_T$'+( '(A)   ' if '_pAdv_' in cluster else '(R)   ')) if '_d_' in cluster else '$DA_T$ at ' + da + ' [sp/s]'
    # clusterData['name'] = 'Dynamic $DA_T$ LTP '+nB[nB.index('4.0_')+4:nB.index('_3e-05_')] if '_d_' in cluster else '$DA_T$ at ' + da + ' [sp/s]'
    clusterData['name'] = alternatives[nB_i]
    clusterData['cname'] = clusterData['name'].replace('$','').replace('_T','')
    for n in range(nExp):
        filename = cluster+'_'+'{:03d}'.format(n)
        if trialSeeds and trialSeedsStart <= n:
            filename = filename.replace('_d_','_d_s'+str(n)+'_')
        elif trialSeeds:
            filename = filename.replace('_d_','_d_sX_')
        clusterData['files'].append(filename)
    files.append(clusterData)
    # break

N = len(files)
cues = range(3) if garivierMoulines else range(4)
weights = ['weights_'+str(c) for c in cues]
values  = ['values_'+str(c) for c in cues]
ltd     = ['ltd_'+str(c) for c in cues]
ltp     = ['ltp_'+str(c) for c in cues]
stdp    = ['stdp_'+str(c) for c in cues]
cues_rewardsL  = ['cues_r_'+str(c) for c in cues]
pairs = cues if garivierMoulines else [[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]]
if 'singlePattern' in runby:
    pairs = [pairs[int(pair)]]
pairsLabels = ['Cue '+ str(c+1) for c in pairs] if garivierMoulines else ['Cues '+str(p[0]+1)+'-'+str(p[1]+1) for p in pairs]
patTrials = int(nTrials / len(pairsLabels))

dataDict = {
            'selection'    : [ ], 
            'nselected'    : [ ], 
            'wSelection'   : [ ],
            'performance'  : [ ],
            'wP'           : [ ],
            'rewards'      : [ ],
            'cumRewards'   : [ ],
            'cumPerformance': [ ],
            'wR'           : [ ],
            'filtRewards'  : [ ],
            'reversal'     : [300,500] if garivierMoulines else [],
            'regret'       : [ ],
            'cumRegret'    : [ ],
            'wRegret'      : [ ],
            'pRegret'      : [ ],
            'pRegret_rew'  : [ ],
            'cumCoherent'  : [ ], 
            'coherent'     : [ ], 
            'wCoherent'    : [ ], 
            'SNc'          : [ ],
            'SNc_t'        : [ ],
            'advantage'    : [ ],
            'wAdvantage'   : [ ],
            'pAdvantage'   : [ ],
            'wpAdvantage'  : [ ],
            'cumAdvantage' : [ ],
            'cumPAdvantage': [ ],
            'filtAdvantage': [ ],
            'fails'        : [ ],
            'cogTime'      : [ ],
            'motTime'      : [ ],
            'accSTDP'      : [ ],
            'lenght'       : [ ],
            'entropy'      : [ ],
            't'            : range(nTrials),
            }

dataPlotInfo = {
            'advantage'    : { 'name': 'Advantage', 'color': '', 'ylim': [0,1], 'legend': ['Advantage'], 'ylabel': 'Amplitude'},
            'wAdvantage'   : { 'name': 'Advantage', 'color': '', 'ylim': [0,1], 'legend': ['Advantage'], 'ylabel': 'Amplitude'},
            'pAdvantage'   : { 'name': 'Perceived Advantage', 'color': '', 'ylim': [0,1], 'legend': ['Advantage'], 'ylabel': 'Amplitude'},
            'wpAdvantage'  : { 'name': 'Perceived Advantage', 'color': '', 'ylim': [0,1], 'legend': ['Advantage'], 'ylabel': 'Amplitude'},
            'cumAdvantage' : { 'name': 'Accumulated advantage',  'color': 'r', 'ylim': [0,200], 'legend': ['Accum. advantage' ], 'ylabel': 'Amount'},
            'cumPAdvantage': { 'name': 'Accumulated perceived advantage',  'color': 'r', 'ylim': [0,nTrials*.4], 'legend': ['Accum. advantage' ], 'ylabel': 'Amount'},
            'entropy'      : { 'name': 'Entropy of selections',  'color': 'r', 'ylim': [0,6], 'legend': ['Entropy' ], 'ylabel': 'Entropy [bits]'},
            'filtAdvantage': { 'name': 'Advantage (filtered)', 'color': '', 'legend': None, 'ylim': [0,1], 'legend': ['Advantage'], 'ylabel': 'Amplitude'},
            'lenght'       : { 'name': 'Completed trials',  'color': '', 'ylabel': 'Completed\ntrials', 'ylim': [0,nTrials+10]},
            'LTD'          : { 'name': 'Long-term depression', 'color': '', 'legend': None, 'ylim': [-0.001,0], 'ylabel': 'Amplitude'},
            'LTP'          : { 'name': 'Long-term potentiation', 'color': '', 'ylim': [0,0.03], 'ylabel': 'Amplitude'},
            'SNc'          : { 'name': 'SNc', 'color': '', 'ylim': [0,12], 'legend': ['$V_{SNc}$'], 'ylabel': 'Tonic dopamine\nactivation [sp/s]'},
            'SNc_t'        : { 'name': 'SNc Tonic Activity', 'color': '', 'ylim': [0,12], 'legend': ['$SNc_T$'], 'ylabel': 'Tonic dopamine\nactivation [sp/s]'},
            'STDP'         : { 'name': 'Synaptic-term dynamic plasticity', 'color': '', 'ylim': [-0.05,0.05], 'legend': ['STDP'], 'ylabel': 'Variation at\ncorticostriatal connections'},
            'accSTDP'      : { 'name': 'Accumulated STDP', 'color': '', 'ylim': [-0.2,0.1], 'legend': ['STDP'], 'ylabel': 'Accumulated variations at\ncorticostriatal connections'},
            'wRegret'      : { 'name': 'Regret', 'color': '', 'ylim': [0,.6], 'legend': ['Regret'], 'ylabel': 'Regret'},
            'regret'       : { 'name': 'Regret', 'color': 'r', 'ylim': [0,1], 'legend': ['Regret'], 'ylabel': 'Amplitude'},
            'pRegret'      : { 'name': 'Regret', 'color': 'r', 'ylim': [0,1], 'legend': ['Regret'], 'ylabel': 'Amplitude'},
            'pRegret_rew'  : { 'name': 'Regret', 'color': 'r', 'ylim': [0,1], 'legend': ['Regret'], 'ylabel': 'Amplitude'},
            'cumRegret'    : { 'name': 'Accumulated regret',  'color': 'r', 'ylim': [0,nTrials*.3], 'legend': ['Accum. regret' ], 'ylabel': 'Accumulated\nregret'},
            'wSelection'   : { 'name': 'Selections', 'color': 'b', 'legend': [ 'Mean selections'], 'ylabel': 'Amplitude'},
            'selection'    : { 'name': 'Selected cues', 'color': 'magenta', 'legend': ['Selected'], 'ylabel': 'Amplitude'},
            'nselected'    : { 'name': 'Unselected cues',  'color': 'r', 'legend': ['Not selected' ], 'ylabel': 'Amplitude'},
            'selectionTime': { 'name': '', 'color': ['tab:orange', 'tab:olive'], 'ylim': [500,4000], 'legend': ['Motor','Cognitive'], 'ylabel': 'Selection\ntime [ms]'},
            'performance'  : { 'name': 'Performance', 'color': 'b', 'legend': ['Performance' ], 'ylabel': 'Performance'},
            'wP'           : { 'name': 'Performance', 'color': 'b', 'ylim': [0,1], 'legend': ['Mean performance'], 'ylabel': 'Performance'},
            'cumPerformance': { 'name': 'Accumulated performance', 'color': '', 'legend': ['Performance' ], 'ylim': [0,nTrials], 'ylabel': 'Accumulated\nperformance'},
            'rewards'      : { 'name': 'Rewards', 'color': 'g', 'legend': ['Reward' ], 'ylabel': 'Amplitude'},
            'cumRewards'   : { 'name': 'Accumulated rewards', 'color': '', 'legend': ['Reward' ], 'ylim': [0,nTrials], 'ylabel': 'Accumulated\nrewards'},
            'filtRewards'  : { 'name': 'Rewards (filtered)', 'color': '', 'legend': ['Reward' ], 'ylabel': 'Amplitude'},
            'wR'           : { 'name': 'Rewards', 'color': 'g', 'ylim': [0,1], 'legend': ['Rewards'], 'ylabel': 'Rewards'},
            'weights'      : { 'name': 'Weights', 'color': '', 'ylim': [0.4,0.6] if 'higherW' in datafolder else [0.45,0.65], 'legend': ['$w_'+str(c)+'$' for c in cues], 'ylabel': 'Amplitude'},
            'values'       : { 'name': 'Values', 'color': '', 'ylim': [0,1], 'legend': ['$v_'+str(c)+'$' for c in cues], 'ylabel': 'Cues\' values'},
            'probChoice'   : { 'name': 'Estimated probability of choice preferences', 'color': '', 'ylim': [0.2,1], 'legend': pairsLabels, 'ylabel': 'Mean of 10 last\nchoices between pairs'},
            'coherent'     : { 'name': 'Coherent selections', 'color':'', 'ylim': [0,2], 'legend': ['Coh. selections'], 'ylabel': 'Amplitude'},
            'wCoherent'    : { 'name': 'Coherent selections', 'color':'', 'ylim': [0,1], 'legend': ['Coh. selections'], 'ylabel': 'Choise\'\ncoherence'},
            'cumCoherent'  : { 'name': 'Coherent selections', 'color':'tab:orange', 'ylim': [0,nTrials], 'legend': ['Coherent selections','Accum. coherent'], 'ylabel': 'Accumulated\ncoherence'},
            'cues_rewards' : { 'name': 'Cues rewards', 'color': '', 'ylim': [0,1], 'legend': ['$cue_'+str(c+1)+'$' for c in cues], 'ylabel': 'Amplitude'},
            'fails'        : { 'name': 'Failed trials', 'color':'', 'ylabel': 'Counts'},
            }

other_colors = []

data = []
commonNames = {'cumCoherent':'cumCoherent', 'coherent':'coherent', 'wCoherent':'wCoherent', 'motTime':'selectionTime','cogTime':'selectionTime'}
compriseWeights = True
compriseValues  = True
compriseLTDLTP  = True
compriseSTDP    = True

for c in cues:
    dataDict[weights[c]] = []
    dataDict[values[c]] = []
    dataDict[ltd[c]] = []
    dataDict[ltp[c]] = []
    dataDict[stdp[c]] = []

    if compriseWeights:
        commonNames[weights[c]] = weights[c]
        dataPlotInfo[weights[c]] = { 'name': '', 'legend': ['$w_'+str(c_+1)+'$' for c_ in cues], 'ylabel': 'Corticostriatal\nweights', 'color': colors[0:len(cues)], 'ylim': [0.49,0.61] if ('higherW' in datafolder or 'regPatterns0' in runby) else [0.45,0.55]}
    else:
        dataPlotInfo[weights[c]] = { 'name': 'Corticostriatal\nweight '+str(c+1), 'ylabel': 'Corticostriatal\nweight '+str(c+1), 'legend': ['$w_'+str(c_+1)+'$' for c_ in cues], 'color': colors[c], 'ylim': [0.49,0.61] if ('higherW' in datafolder or 'regPatterns0' in runby) else [0.45,0.55]}
    if compriseValues:
        commonNames[values[c]] = values[c]#'values'
        dataPlotInfo[values[c]] = { 'name': '', 'legend': ['$v_'+str(c_+1)+'$' for c_ in cues], 'color': colors[0:len(cues)], 'ylim': [0,1], 'ylabel': 'Cues\nvalues'}
    else:
        dataPlotInfo[values[c]] = { 'name': 'Value '+str(c+1), 'legend': ['$v_'+str(c_+1)+'$' for c_ in cues], 'color': colors[c], 'ylim': [0,1], 'ylabel': 'Amplitude'}

    if compriseLTDLTP:
        commonNames[ltd[c]] = ltd[c]
        commonNames[ltp[c]] = ltp[c]
        dataPlotInfo[ltd[c]] = { 'name': 'Long-term depression', 'legend': None, 'color': colors[0:len(cues)], 'ylim': [-0.005,0]}
        dataPlotInfo[ltp[c]] = { 'name': 'Long-term potentiation', 'legend': None, 'color': colors[0:len(cues)], 'ylim': [0,0.03]}
    else:
        dataPlotInfo[ltd[c]] = { 'name': 'Long-term depression on corticostriatal connection '+str(c+1), 'legend': None, 'color': colors[0:len(cues)], 'ylim': [-0.005,0]}
        dataPlotInfo[ltp[c]] = { 'name': 'Long-term potentiation on corticostriatal connection '+str(c+1), 'legend': None, 'color': colors[0:len(cues)], 'ylim': [0,0.03]}

    if compriseSTDP:
        commonNames[stdp[c]] = stdp[c]
        dataPlotInfo[stdp[c]] = { 'name': 'Synaptic-term\ndynamic plasticity', 'legend': None, 'color': colors[0:len(cues)], 'ylim': [-0.05,0.05]}
    else:
        dataPlotInfo[stdp[c]] = { 'name': 'Synaptic-term\ndynamic plasticity at\ncorticostriatal connection '+str(c+1), 'legend': None, 'color': colors[0:len(cues)], 'ylim': [-0.05,0.05]}

    dataDict[cues_rewardsL[c]] = []
    commonNames[cues_rewardsL[c]] = 'cues_rewards'



for pL_i in range(len(pairsLabels)):
    dataDict[pairsLabels[pL_i]] = []
    # commonNames[pairsLabels[pL_i]] = 'probChoice'
    dataPlotInfo[pairsLabels[pL_i]] = { 'name': pairsLabels[pL_i], 'ylabel': 'Choice preference\nfor '+pairsLabels[pL_i] , 'legend': None, 'color': colors[pL_i], 'ylim': [0,1]}

eps = np.finfo(float).eps
def weighted(prev,new,alpha=0.8):
    return alpha*prev+(1-alpha)*new

def doPlot(data,plotVars, block=False):
    def plotMeanVar(ax,x,mean=None,var=None,min=None,max=None,all=None,lbl=None,color='b',ylim='',xlim='', ylabel= ''):
        if color:
            if mean is not None:
                if var is not None:
                    ax.fill_between(x,mean+var,mean-var,alpha=0.5,color=color)
                lines = ax.plot(x,mean,color=color,alpha=0.6,linewidth=4.0)
            if min is not None and max is not None:
                ax.fill_between(x,max,min,alpha=0.1,color=color)
            if all is not None:
                if isinstance(color, list):
                    allTransposed = np.transpose(all)
                    for index in range(allTransposed.shape[0]):
                        ax.plot(x,allTransposed[index],color=color[index],alpha=0.1,linewidth=1.0)
                else:
                    ax.plot(x,all,color=color,alpha=0.1,linewidth=2.0)
        else:
            if mean is not None:
                if var is not None:
                    ax.fill_between(x,mean+var,mean-var,alpha=0.5)
                lines = ax.plot(x,mean,alpha=0.6,linewidth=4.0)
            if min is not None and max is not None:
                ax.fill_between(x,max,min,alpha=0.1)
            if all is not None:
                ax.plot(x,all,alpha=0.4,linewidth=2.0)

        # if lbl:
        #     ax.legend(labels=lbl,framealpha=0.6,ncol=len(cues),loc='upper right')
        if ylim:
            ax.set_ylim(ylim)
        if xlim:
            ax.set_xlim(xlim)
        if ylabel:
            ax.set_ylabel(ylabel)

        return lines

    for figVars in plotVars:        
        n_axis = -1
        nPlots = 0
        usedNames = []
        for var in figVars['vars']:
            if var in commonNames:
                if not (dataPlotInfo[commonNames[var]]['name'] in usedNames):
                    usedNames.append(dataPlotInfo[commonNames[var]]['name'])
                    nPlots +=1
            else:
                nPlots +=1

        if figVars.get('size',''):
            fig, axs = plt.subplots(nPlots,1,figsize=figVars['size'])
        else:
            fig, axs = plt.subplots(nPlots,1,figsize=(6,2))

        if nPlots == 1:
            axs = [axs]

        currentColor = 0

        for var in figVars['vars']:
            print('\t -> '+var)
            sameAxis = False

            if var in commonNames:
                plotInfo = dataPlotInfo[commonNames[var]]
                if (n_axis >= 0 and axs[n_axis].get_title() != plotInfo['name']) or n_axis < 0:
                    n_axis += 1
                    axs[n_axis].set_title(plotInfo['name'])
                else:
                    sameAxis = True
            else:
                plotInfo = dataPlotInfo[var]
                n_axis += 1
                axs[n_axis].set_title(plotInfo['name'])

            startAt = len(data['mean'][var]) - len(data['mean']['t'])

            if sameAxis:
                currentColor += 1

            lines = plotMeanVar(
                ax = axs[n_axis],
                x = data['mean']['t'],
                mean = data['mean'][var][startAt:] if var in addMean else None,
                var = data['var'][var][startAt:] if var in addMean and not var in avoidVar else None,
                min = data['min'][var][startAt:] if var in addMinMax else None,
                max = data['max'][var][startAt:] if var in addMinMax else None,
                all = np.transpose(data['all'][var])[startAt:] if var in addAll else None,
                lbl = plotInfo.get('legend',None),
                color = plotInfo['color'][currentColor] if isinstance(plotInfo['color'], list) else plotInfo['color'],
                ylim = plotInfo['ylim'] if 'ylim' in plotInfo else '',
                xlim = (data['mean']['t'][0],data['mean']['t'][-1]),
                ylabel = plotInfo['ylabel'] if 'ylabel' in plotInfo else data['name'],
                )

        for ax in axs:
            ylim = ax.get_ylim()
            if len(data['mean']['reversal']):
                for rv in data['mean']['reversal']:
                    ax.plot([rv,rv],ylim,'--',color=(0.3,0.3,0.3,0.5),linewidth=4,alpha=0.3)
            ax.grid(alpha=0.4)
            if ax == axs[-1]:
                ax.set_xlabel('Trial')

            if plotInfo.get('legend',None):
                ax.set_title('')

        # plt.suptitle(data['name'],x=0.8)

        plt.tight_layout()

        for ext in figExtensions:
            figsubfolder = figfolder+figVars['title']+'/'
            checkFolder(figsubfolder)
            figName = figsubfolder+figVars['title']+data['fName'].replace('.csv',ext)
            if figName[-4:] != ext:
                figName += ext
            
            plt.savefig(figName)

        if not figVars.get('no_legend',False) and plotInfo.get('legend',None):
            figLegend, ax = plt.subplots(2,1,figsize=(6,2))
            lines = []
            for leg, col in zip(plotInfo['legend'],plotInfo['color']):
                lines += ax[0].plot([[n]*N for n in range(N)],[[n]*N for n in range(N)],linewidth=4.0,label=leg,color=col)

            #ax.legend(np.transpose(lines),labels=np.transpose(plotInfo.get('legend',None)),
            ax[0].legend(framealpha=0.6,ncol=min((4,len(lines))),loc='upper center',)#,bbox_to_anchor=(0.5, 0.5))
            ax[0].set_xlim((-2,-1))
            if plotInfo['name']:
                ax[1].set_title(plotInfo['name'],y=0.2)
            for axis in ax:
                axis.set_axis_off()
                axis.margins(0)

            figLegend.tight_layout()
            params = dict(bottom=0, left=0, right=1)
            figLegend.subplots_adjust(**params)
            for ext in figExtensions:
                figName = figsubfolder+figVars['title']+'legend'
                if figName[-4:] != ext:
                    figName += ext
                plt.savefig(figName)
            plt.close(figLegend)

        if showFig:
            plt.show(block=block)
        else:
            plt.close(fig)


def getStatistics(meanData,varData,minData,maxData,semData,allData,dataArray,keys,mV=[]):
    allKeys = []
    for k in keys:
        allKeys += k['vars']
    allKeys += mV

    noDuplicates = []
    [noDuplicates.append(x) for x in allKeys if x not in noDuplicates]

    for k in noDuplicates:
        for data in dataArray:
            allData[k].append(data[k])        
            expectedLen = nTrials +1# +(1 if (k[0]=='w' and not ('weight' in k)) or 'cum' in k else 0)
            if len(allData[k][-1]) < expectedLen:
                allData[k][-1] += [np.nan]*(expectedLen-len(allData[k][-1]))


        allData[k] = np.array(allData[k])
        # if k in addMinMax:
        minData[k] = np.nanmin(allData[k],axis=0)
        maxData[k] = np.nanmax(allData[k],axis=0)
        varData[k] = np.nanvar(allData[k],axis=0,ddof=1)
        meanData[k] = np.nanmean(allData[k],axis=0)
        semData[k] = np.nanstd(allData[k],ddof=1, axis=0) / np.sqrt(nExp)

addMean = ['wP','wR','regret','cumRegret','cumRewards','cumCoherent','wCoherent','wAdvantage','wpAdvantage','wRegret','pRegret','pRegret_rew','pAdvantage','satAdvantage_','filtAdvantage','SNc','SNc_t','fails','cogTime','motTime','accSTDP','entropy']+values+weights+pairsLabels+ltd+ltp+stdp#+cues_rewardsL
avoidVar = ['cumRegret','fails','cogTime','motTime']# +pairsLabels
addMinMax = ['wCoherent','wP','wR','wRegret','pRegret','pRegret_rew','pAdvantage','wpAdvantage','filtAdvantage','fails','cogTime','motTime','SNc']#+cues_rewardsL
addAll = ['cumRegret','pAdvantage','wpAdvantage','accSTDP']+weights+values+ltd+ltp+stdp# +pairsLabels

plotVars = [# {'vars': ['wP','cumRegret','wCoherent','cumCoherent'], 'title': 'perf_regret_cselections_'},
            {'vars': weights, 'title': 'weights_', 'size': (6,2) if compriseWeights else (6,10)},
            {'vars': values, 'title': 'values_', 'size': (6,2) if compriseValues else (6,10)},
            {'vars': ['wP','wR','SNc','wRegret','wCoherent'], 'title': 'perf_', 'size': (6,10)},
            # {'vars': ['fails'], 'title': 'failedTrials_'},
            {'vars': ['accSTDP'], 'title': 'accSTDP_'},
            {'vars': ['SNc'], 'title': 'dopamine_'},
            {'vars': ['entropy'], 'title': 'entr_'},
            # {'vars': ['wP'], 'title': 'perf_'},
            {'vars': ['wR'], 'title': 'rew_'},
            {'vars': ['wCoherent'], 'title': 'coherence_'},
            # {'vars': ltd, 'title': 'ltd_', 'size': (6,10)},
            # {'vars': ltp, 'title': 'ltp_', 'size': (6,10)},
            {'vars': stdp, 'title': 'stdp_', 'size': (6,10)},
            ## {'vars': ['SNc_t'], 'title': 'tonicDA_'},
            # {'vars': ['cumRewards'], 'title': 'cumReward_'},
            {'vars': ['cumRegret'], 'title': 'cumRegret_'},
            {'vars': ['wAdvantage'], 'title': 'wAdvantage_'},
            # {'vars': ['pAdvantage'], 'title': 'percAdvantage_'},
            # {'vars': ['wpAdvantage'], 'title': 'percAdvantage_weighted_'},
            # {'vars': ['filtAdvantage'], 'title': 'filtAdvantage_'},
            {'vars': ['wRegret'], 'title': 'regret_'},
            # {'vars': ['pRegret'], 'title': 'pRegret_'},
            # {'vars': ['pRegret_rew'], 'title': 'pRegretRew_'},
            #{'vars': cues_rewardsL, 'title': 'cues_R_'},
            {'vars': pairsLabels, 'title': 'pchoice_'},
            {'vars': ['motTime','cogTime'], 'title': 'selectionTimes_'},
            # {'vars': values+weights+['motTime','cogTime'], 'title': 'beh_', 'no_legend': True },
            #{'vars': weights+values, 'title': 'weights_values_'}
            ]

checkFolder(figfolder)

moreVarsForStatistics = ['wR','regret','rewards','performance','cumAdvantage','cumPAdvantage','advantage','filtRewards','cumCoherent','wCoherent','filtAdvantage','cumRewards','cumPerformance','fails']
shouldLoadFiles = askFlag("Existing dataFile, should it (them) be loaded? ", default='a') # 'N' #False
shouldPlotEitherway = askFlag("Actualize plots of loaded files? ", default='n') # 'N' #False
shouldUseTrunkedExps = askFlag("Should we continue?", default='always')
shouldSaveFiles = askFlag("Save datafiles?", default='a')

for dataFiles in files:
    print("\nReading "+datafolder+dataFiles['cname'])

    if dataFilesFolder and os.path.isfile(dataFilesFolder+dataFiles['fName']):

        # always = checkAlwaysFlag(always,"Existing dataFile, should it (them) be loaded? ")
        #if always and always != 'N':
        if shouldLoadFiles.askAndCheck() :
            with open(dataFilesFolder+dataFiles['fName'],'rb') as dFile:
                dataFiles['meanVar'] = pickle.load(dFile)
                #mV = np.load(dFile, allow_pickle=True)
                #dataFiles['meanVar'] = mV['meanVar']
                #print(dataFiles['meanVar'])
                print('data loaded, continuing...')
                if shouldPlotEitherway.askAndCheck():
                    doPlot(dataFiles['meanVar'],plotVars,False)
            continue

    lenghts = []

    for dataFile in dataFiles['files']:
        # print '\t:',dataFile
        data_i = copy.deepcopy(dataDict)
        data_i['fName'] = dataFile

        data_i['wSelection'].append(0)
        data_i['wP'].append(0.5)
        data_i['wR'].append(0.5)
        data_i['cumPerformance'].append(0)
        data_i['cumRewards'].append(0)
        data_i['cumRegret'].append(0)
        data_i['cumCoherent'].append(0)
        data_i['wCoherent'].append(0)
        data_i['wAdvantage'].append(0)
        data_i['pAdvantage'].append(0)
        data_i['wpAdvantage'].append(0)
        data_i['cumAdvantage'].append(0)
        data_i['cumPAdvantage'].append(0)
        data_i['wRegret'].append(0)
        for c in cues:
            data_i[stdp[c]].append(0)

        try:
            file = open(datafolder+dataFile)
        except:
            file = open(commonData+dataFile.replace("dDA_2.0","dDA_6.0"))

        reader = csv.DictReader(file,delimiter=',')
        lenOfFile = len(list(reader))
        lenghts.append(lenOfFile)
        if ( lenOfFile < nTrials):
            print(datafolder+dataFile+' lacks trials ('+str(lenOfFile)+'/'+str(nTrials)+')')
            if not shouldUseTrunkedExps.askAndCheck():
                continue
        file.seek(0)
        reader.__init__(file, delimiter=',')

        trial = 0
        currentCuesRewards = cues_reward
        nreversal = np.array([r-1 for r in data_i['reversal']])

        for row in reader:
            if trial >= nTrials:
                break
            data_i['selection'].append(float(row['choice']))
            data_i['wSelection'].append(weighted(data_i['wSelection'][-1],float(row['choice'])))
            data_i['nselected'].append(float(row['nchoice']))

            data_i['performance'].append(float(row['P']))
            data_i['wP'].append(weighted(data_i['wP'][-1],float(row['P'])))
            data_i['cumPerformance'].append(data_i['cumPerformance'][-1]+data_i['performance'][-1])

            data_i['rewards'].append(float(row['R'] == 'True'))
            data_i['cumRewards'].append(data_i['cumRewards'][-1]+data_i['rewards'][-1])
            data_i['wR'].append(weighted(data_i['wR'][-1],data_i['rewards'][-1]))
            data_i['filtRewards'].append(np.max((data_i['wR'][-1]-0.6,0))/(1-0.6))

            if row['SNc'][0] == '[':
                data_i['SNc'].append(float(row['SNc'][2:-2]))
            else:
                data_i['SNc'].append(float(row['SNc']))
            if row.get('SNc_h'):
                if row['SNc_h'][0] == '[':
                    data_i['SNc_t'].append(float(row['SNc_h'][2:-2]))
                else:
                    data_i['SNc_t'].append(float(row['SNc_h']))

            if trial in nreversal:
                # print "change at",trial,"from "
                # print currentCuesRewards
                currentCuesRewards = cues_rewards[np.sum(nreversal<=trial)]
                # print "to:"
                # print currentCuesRewards

            choice = int(row['choice'])
            nchoice = int(row['nchoice'])
            # bestCuesReward = np.max(currentCuesRewards) if garivierMoulines else \
            #                  np.max((currentCuesRewards[choice],currentCuesRewards[nchoice]))
            # regret = bestCuesReward - currentCuesRewards[choice]
            data_i['regret'].append(float(row['Regret']))
            data_i['wRegret'].append(weighted(data_i['wRegret'][-1],data_i['regret'][-1]))
            data_i['advantage'].append(1-data_i['regret'][-1])
            data_i['wAdvantage'].append(weighted(data_i['wAdvantage'][-1],data_i['advantage'][-1]))
            data_i['cumRegret'].append(data_i['cumRegret'][-1]+data_i['regret'][-1])

            weights_r = [ float(v) for v in row['weights'].split('\t') ]
            values_r =  [ float(v) for v in row['values'].split('\t') ]
            c_rewards = []
            if row.get('c_rewards',''):
                c_rewards = [float(r) for r in row['c_rewards'].split('\t')]
            for c in cues:
                data_i[weights[c]].append(weights_r[c])
                data_i[values[c]].append(values_r[c])
                if c_rewards:
                    data_i[cues_rewardsL[c]].append(c_rewards[c])
                if trial:
                    data_i[stdp[c]].append(data_i[weights[c]][-1]-data_i[weights[c]][-2])

                if c:
                    accSTDP += data_i[stdp[c]][-1]
                else:
                    accSTDP = data_i[stdp[c]][-1]

            if trial:
                data_i['accSTDP'].append(accSTDP+data_i['accSTDP'][-1])
            else:
                data_i['accSTDP'].append(accSTDP)

            if garivierMoulines:
                data_i['pRegret'].append(np.max(values_r) - values_r[choice])
                data_i['pRegret_rew'].append(np.max(values_r) - data_i['rewards'][-1])# values_r[choice])
                pAdv = 1-data_i['pRegret_rew'][-1]
            else:
                data_i['pRegret'].append(np.max([values_r[choice],values_r[nchoice]]) - values_r[choice])
                data_i['pRegret_rew'].append(np.max([values_r[choice],values_r[nchoice]]) - data_i['rewards'][-1])# values_r[choice])
                pAdv = 1-data_i['pRegret_rew'][-1]

            #data_i['pAdvantage'].append(weighted(data_i['pAdvantage'][-1],float(row['pA']),0.8))#
            data_i['pAdvantage'].append(weighted(data_i['pAdvantage'][-1],pAdv,0.8))#
            data_i['wpAdvantage'].append(weighted(data_i['wpAdvantage'][-1],pAdv,0.85))#data_i['pAdvantage'][-1],0.85))
            data_i['cumAdvantage'].append(data_i['cumAdvantage'][-1]+data_i['advantage'][-1])
            data_i['cumPAdvantage'].append(data_i['cumPAdvantage'][-1]+data_i['wpAdvantage'][-1])
            data_i['filtAdvantage'].append(np.max((data_i['wpAdvantage'][-1]-0.9,0))/(1-0.9))

            probChoice = row['pchoice'].split('\t')
            entr = 0
            for i, pL in enumerate(pairsLabels):
                print(i)
                print(pL)
                print(probChoice)
                print(probChoice[i])
                data_i[pL].append(float(probChoice[i]))
                entr -= data_i[pL][-1]*np.log2(data_i[pL][-1])
            data_i['entropy'].append(entr)

            ltdltp_i = row['LTD-LTP'].split('\t')
            for i, ltd_i in enumerate(ltd):
                data_i[ltd_i].append(float(ltdltp_i[len(cues)*i+0]))

            for i, ltp_i in enumerate(ltp):
                data_i[ltp_i].append(float(ltdltp_i[len(cues)*i+2]))

            if garivierMoulines:
                weights_f = [float(w) for w in weights_r]
                data_i['coherent'].append(0 + ( weights_f[choice] >= (np.max(weights_f) - eps) ) )
            else:
                data_i['coherent'].append(0+(float(weights_r[choice]) > float(weights_r[nchoice])))
            data_i['wCoherent'].append(weighted(data_i['wCoherent'][-1],data_i['coherent'][-1]))
            data_i['cumCoherent'].append(data_i['cumCoherent'][-1]+data_i['coherent'][-1])
            data_i['fails'].append(float(row['failedTrials']))
            data_i['motTime'].append(float(row['motTime']))
            data_i['cogTime'].append(float(row['cogTime']))

            trial += 1

        data_i['coherent'] = [ c * np.mean(data_i['cumCoherent']) for c in data_i['coherent'] ]
        data_i['wCoherent'] = [ c for c in data_i['wCoherent'] ]
        dataFiles['data'].append(data_i)

        file.close()

    meanData = copy.deepcopy(dataDict)
    varData  = copy.deepcopy(dataDict)
    minData  = copy.deepcopy(dataDict)
    maxData  = copy.deepcopy(dataDict)
    semData  = copy.deepcopy(dataDict)
    allData  = copy.deepcopy(dataDict)

    getStatistics(meanData,varData,minData,maxData,semData,allData,dataFiles['data'],plotVars,moreVarsForStatistics)

    data = {
            'mean': meanData, 
            'var' : varData, 
            'min' : minData,
            'max' : maxData,
            'sem' : semData,
            'all' : allData,
            'nExp': len(dataFiles['data']),
            'name': dataFiles['name']+' N: '+str(nExp),
            'fName': dataFiles['fName'],
            'reversal': dataDict['reversal'],
            'lenght' : lenghts
            }

    dataFiles['meanVar'] = data

    if dataFilesFolder and not shouldLoadFiles.check() and shouldSaveFiles.askAndCheck() :
        with open(dataFilesFolder+dataFiles['fName'],'wb') as dFile:
            pickle.dump(dataFiles['meanVar'],dFile)
        with open(dataFilesFolder+dataFiles['fName']+'.npz','wb') as dFile:
            np.savez(dFile,meanVar=dataFiles['meanVar'])

    doPlot(data,plotVars,False)

# Get statistics
###############

def doWilcoxon(x,y,strdiff,xname,yname,alternatives=['less','greater','two-sided']):
    for alt in alternatives:
        w = wilcoxon(x,y,alternative=alt)
        if w[1] < 0.05:
            strdiff = np.concatenate((strdiff,["[W-"+alt+"]"+yname+" - (p="+str(w[1])+" - "+str(w[0])+")"]))
            # strdiff[0] = np.concatenate((strdiff[0],["[W] "+yname+" - (p="+str(w[1])+")"]))
            # strdiff[1] = np.concatenate((strdiff[1],["[W] "+xname+" - (p="+str(w[1])+")"]))

    return strdiff

def doMannwhitneyu(x,y,strdiff,xname,yname,alternatives=['less','greater','two-sided']):
    for alt in alternatives:
        w = mannwhitneyu(x,y,alternative=alt)
        if w[1] < 0.05:
            strdiff = np.concatenate((strdiff,["[U-"+alt+"] "+yname+" - (p="+str(w[1])+")"]))
            # strdiff[0] = np.concatenate((strdiff[0],["[U-"+alt.upper()[0]+"] "+yname+" - (p="+str(w[1])+")"]))
            # strdiff[1] = np.concatenate((strdiff[1],["[U-"+alt.upper()[0]+"] "+xname+" - (p="+str(w[1])+")"]))

    return strdiff


printStatistics = [ 'wP','wR','cumRegret','wRegret', 'wCoherent','cumCoherent','cumPerformance',
                    'cumRewards','cumAdvantage','cogTime','motTime','accSTDP']+\
                     weights+values+stdp
expLevelStatistics = ['lenght']
statisticsAgainstAll = ['weights_3', 'motTime']
statistics = [
              {'name':'wilcoxon','func':doWilcoxon},
              # {'name':'mannwhitneyu','func':doMannwhitneyu}
              ]
meansDict =  {'cluster': '', 'nExp':nExp,'nTrial':0,'mean':0.0,'var':0.0}
statisticsTrials = [300,500,nTrials-1] if garivierMoulines else [1,9,19,29,59,89,119]
statisticsFolder = 'statistics/'
checkFolder(statisticsFolder)
statisticsFileName = statisticsFolder+clustBase
statisticsFiles = {}

shouldComputeStatistics = askFlag("Compute statistics?", default='n')
if shouldComputeStatistics.askAndCheck() and statisticsFileName:

    statisticsFiles['trials'] = open(statisticsFileName+'trials','w')
    statisticsFiles['allValues'] = open(statisticsFileName+'allV','w')
    meansFolder = statisticsFolder+'means/'
    checkFolder(meansFolder)
    meansFile = [meansFolder+clustBase+'-'+st+'.csv' for st in (printStatistics+expLevelStatistics)]
    for mF in meansFile:
        if not os.path.isfile(mF):
            with open(mF, 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=meansDict.keys())
                writer.writeheader()
    

def printAtMeanFile(file,clusterName,nExp,nTrial,mean,var,mDict=meansDict):
    with open(file,'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=mDict.keys())
        mDict['cluster'] = clusterName
        mDict['nTrial'] = nTrial
        mDict['mean']   = mean
        mDict['var']    = var
        writer.writerow(mDict)

def printAndWrite(txt, allFiles = False,fileObj=statisticsFiles):
    if allFiles:
        fileObj = statisticsFiles['allValues']
    else:
        fileObj = statisticsFiles['trials']

    if fileObj:
        fileObj.write(txt+'\n')
    print(txt)

def printStats(diff,clustersNames=[n['cname'] for n in files], allFiles = False):
    for df_i in range(len(diff)):
        printAndWrite('\t\t'+clustersNames[df_i], allFiles=allFiles)
        if len(diff[df_i]) == 0:
            printAndWrite('\t\t\tNONE', allFiles=allFiles)
        for line in diff[df_i]:
            printAndWrite('\t\t\t'+line, allFiles=allFiles)

if shouldComputeStatistics.check():
    storedTrialValues = [{} for t in statisticsTrials]
    for st_i, st in enumerate(printStatistics):
        print("\n\n** Getting statistics for "+st+" **")
        trialValues = [[] for t in statisticsTrials]
        allValues = [np.array([]) for f in range(N)]

        for f_i, file in enumerate(files):
            for t_i,t in enumerate(statisticsTrials):
                trialValues[t_i].append([file['meanVar']['all'][st][rep][t] for rep in range(nExp)])
                if (st[0:3] == 'wei') or ('Time' in st):
                    if st in storedTrialValues[t_i].keys():
                        storedTrialValues[t_i][st].append([file['meanVar']['all'][st][rep][t] for rep in range(nExp)])
                    else:
                        storedTrialValues[t_i][st] = [[file['meanVar']['all'][st][rep][t] for rep in range(nExp)]]
            for rep in range(nExp):
                allValues[f_i] = np.concatenate((allValues[f_i],file['meanVar']['all'][st][rep]))

            for t in statisticsTrials:
                countedExp = np.sum(np.array(file['meanVar']['lenght'])>=t)
                printAtMeanFile(meansFile[st_i],file['cname'],countedExp,t,file['meanVar']['mean'][st][t],file['meanVar']['var'][st][t])
            #printAndWrite('Trial '+str(t),meansFile[st_i])
            #for file in files:
            #    printAndWrite('\t'+file['cname'] +': '+
            #        str(file['meanVar']['mean'][st][t])+'+-'+str(file['meanVar']['var'][st][t]),meansFile[st_i])
            #printAndWrite('',meansFile[st_i])

        trialValues = np.array(trialValues)

        for st_type in statistics:
            differsALL = [np.array([]) for f in range(N)]
            differsAt = [[np.array([]) for f in range(N)] for t in statisticsTrials]
            for c1 in range(N):
                for c2 in range(N):
                    if c1 == c2:
                        if st in statisticsAgainstAll:
                            if st[0:3] == 'wei':
                                for cci in weights:
                                    if cci == st:
                                        continue
                                    differsAt[t_i][c1] = st_type['func'](trialValues[t_i][c1],storedTrialValues[t_i][cci][c2],differsAt[t_i][c1],files[c1]['cname'],' / '+cci+' / '+files[c2]['cname'])
                            elif st == 'motTime' :
                                print('yeah')
                                differsAt[t_i][c1] = st_type['func'](trialValues[t_i][c1],storedTrialValues[t_i]['cogTime'][c2],differsAt[t_i][c1],files[c1]['cname'],' / cogTime / '+files[c2]['cname'])
                        continue

                    differsALL[c1] = st_type['func'](allValues[c1],allValues[c2],differsALL[c1],files[c1]['cname'],files[c2]['cname'])
                    for t_i in range(len(statisticsTrials)):
                        if (trialValues[t_i][c1]==trialValues[t_i][c2]).all():
                            differsAt[t_i][c1] = np.concatenate((differsAt[t_i][c1],['Equal to '+files[c2]['cname']]))
                            continue
                        elif np.isnan(trialValues[t_i][c1]).all():
                            differsAt[t_i][c1] = np.concatenate((differsAt[t_i][c1],['All '+files[c1]['cname']+' is nan']))
                            continue
                        elif np.isnan(trialValues[t_i][c2]).all():
                            differsAt[t_i][c1] = np.concatenate((differsAt[t_i][c1],['All '+files[c2]['cname']+' is nan']))
                            continue
                        differsAt[t_i][c1] = st_type['func'](trialValues[t_i][c1],trialValues[t_i][c2],differsAt[t_i][c1],files[c1]['cname'],files[c2]['cname'])
                        if st in statisticsAgainstAll:
                            if st[0:3] == 'wei':
                                for cci in weights:
                                    if cci == st:
                                        continue
                                    differsAt[t_i][c1] = st_type['func'](trialValues[t_i][c1],storedTrialValues[t_i][cci][c2],differsAt[t_i][c1],files[c1]['cname'],' / '+cci+' / '+files[c2]['cname'])
                            elif st == 'motTime' :
                                print('yeah')
                                differsAt[t_i][c1] = st_type['func'](trialValues[t_i][c1],storedTrialValues[t_i]['cogTime'][c2],differsAt[t_i][c1],files[c1]['cname'],' / cogTime / '+files[c2]['cname'])
                    # [differsLV[c1],differsLV[c2]] = st_type['func'](lastValues[c1],lastValues[c2],[differsLV[c1],differsLV[c2]],files[c1]['cname'],files[c2]['cname'])
                    # [differsALL[c1],differsALL[c2]] = st_type['func'](lastValues[c1],lastValues[c2],[differsALL[c1],differsALL[c2]],files[c1]['cname'],files[c2]['cname'])

            printAndWrite(st+" -- "+st_type['name'],allFiles = True)
            printAndWrite("\tDiffers in all values: ",allFiles = True)
            printStats(differsALL, allFiles = True)
            for t_i,t in enumerate(statisticsTrials):
                printAndWrite('')
                printAndWrite(st+" -- "+st_type['name'])
                printAndWrite("\tDiffers in trial ["+str(t)+"]: ")
                printStats(differsAt[t_i])

    for eS_i, eS in enumerate(expLevelStatistics):
        for st_type in statistics:
            differsAt = [np.array([]) for f in range(N)]
            for f_i, file in enumerate(files):
                for f_o, file2 in enumerate(files):
                    if f_i == f_o :
                        continue
                    if file['meanVar'][eS]==file2['meanVar'][eS]:
                        differsAt[f_i] = np.concatenate((differsAt[f_i],['Equal to '+file2['cname']]))
                        continue
                    differsAt[f_i] = st_type['func'](file['meanVar'][eS],file2['meanVar'][eS],differsAt[f_i],file['cname'],file2['cname'])

                printAtMeanFile(meansFile[len(printStatistics)+eS_i],file['cname'],nExp,-1,np.nanmean(file['meanVar'][eS],axis=0),np.nanvar(file['meanVar'][eS],axis=0,ddof=1))

            printAndWrite(eS+" -- "+st_type['name'],allFiles = True)
            printAndWrite("\tDiffers in all values: ",allFiles = True)
            printStats(differsAt, allFiles = True)



    trialSectionStatistics = []
    for i, pair in enumerate(pairsLabels):
        trialSectionStatistics.append({'var': pair, 'trials':[patTrials*i,patTrials*(i+1)]})
    
    for tls in trialSectionStatistics:
        trialValues = []
        for f_i, file in enumerate(files):
            trialValues.append([])
            for rep in range(nExp):
                trialValues[-1] = np.concatenate((trialValues[-1],
                    file['meanVar']['all'][tls['var']][rep][tls['trials'][0]:tls['trials'][1]]))

        for st_type in statistics:
            differsAt = [np.array([]) for f in range(len(files))]
            for f_i, file in enumerate(files):
                if np.isnan(trialValues[f_i]).all():
                    differsAt[f_i] = np.concatenate((differsAt[f_i],['All '+file['cname']+' is nan']))
                    continue
                for f_o, file2 in enumerate(files):
                    if np.isnan(trialValues[f_o]).all():
                        differsAt[f_i] = np.concatenate((differsAt[f_i],[file2['cname']+' is nan']))
                        continue
                    if f_i == f_o :
                        continue
                    if (trialValues[f_i]==trialValues[f_o]).all():
                        differsAt[f_i] = np.concatenate((differsAt[f_i],['Equal to '+file2['cname']]))
                        continue
                    differsAt[f_i] = st_type['func'](trialValues[f_i],trialValues[f_o],differsAt[f_i],file['cname'],file2['cname'])

            printAndWrite(tls['var']+" -- "+st_type['name'])
            printAndWrite("\tDiffers in trials",tls['trials'].__str__,": ")
            printStats(differsAt)



# Plots comparing clusters
##########################
shouldPlotClusters = askFlag("Do cluster plots?", default='a')
if not shouldPlotClusters.askAndCheck():
    print("That's all, thanks!")
    exit(0)
useColors = len(colors) >= len(files)

# 'others' options: 
#          fillIn    (variance in colored shadow)
#          addMinMax (shows min and max values)
#          addVar    (includes variance lines)
#          legend    (includes legend)

## def newGenPlot(name, var, otherOpt, addedDist):
##     return { 'name': name, 'vars': [var], 'options': {'others': otherOpt, 'addDist': [{'lim': d[lims], }] } }
## 
## generalPlots.append( newGenPlot( name = 'wP', var = 'wP', otherOpt = ['fillIn', 'addMinMax'],
##                                  addedDist = [ {'lims': [50,200] if garivierMoulines else [0,1] } ]))

generalPlots = [
##        {   'name': 'wP',
##            'vars': ['wP'],
##            'options': { 'others':['fillIn'],
##                         # 'addDist': [{'lim': [50,200] if garivierMoulines else [0,1], 'bbox':(.08, .12, .25, .25), 'distTrialPos': nTrials-1, 'type': 'boxplot'},
##                         #             {'lim': [50,200] if garivierMoulines else [0,1], 'bbox':(.08, .60, .25, .25), 'distTrialPos': 59, 'type': 'boxplot'}
##                         #         ]
##                            }
##            },
##        {   'name': 'cumCoherence',
##            'vars': ['cumCoherent'],
##            'options': { 'others':['addMinMax'],
##                         'addDist': [{'lim': [0,100] if garivierMoulines else [30,100] , 'bbox':(.08, .12, .25, .25), 'distTrialPos': nTrials-1, 'type': 'boxplot'},
##                                     {'lim': [30,160] if garivierMoulines else [0,60] , 'bbox':(.08, .60, .25, .25), 'distTrialPos': 59, 'type': 'boxplot'}
##                                 ]
##                            }
##            },
##        {   'name': 'entropy',
##            'vars': ['entropy'],
##            'options': { 'others':['fillIn'],
##                         'addDist': [{'lim': [0,100] if garivierMoulines else [0,6] , 'bbox':(.08, .12, .25, .25), 'distTrialPos': nTrials-1, 'type': 'boxplot'},
##                                     {'lim': [30,160] if garivierMoulines else [0,6] , 'bbox':(.08, .60, .25, .25), 'distTrialPos': 59, 'type': 'boxplot'}
##                                 ]
##                            }
##            },
##        {   'name': 'tonicDA',
##            'vars': ['SNc'],
##            'options': { 'others':['fillIn', 'addMinMax'],}
##                         # 'addDist': [{'lim': [0,100] if garivierMoulines else [60,130] , 'bbox':(.08, .12, .25, .25), 'distTrialPos': nTrials-1, 'type': 'boxplot'},
##                         #             {'lim': [30,160] if garivierMoulines else [30,60] , 'bbox':(.08, .60, .25, .25), 'distTrialPos': 59, 'type': 'boxplot'}
##                         #         ]
##                         #    }
##            },
##        {   'name': 'STDP',
##            'vars': ['accSTDP'],
##            'options': { 'others':['fillIn'],}
##                         # 'addDist': [{'lim': [0,100] if garivierMoulines else [60,130] , 'bbox':(.08, .12, .25, .25), 'distTrialPos': nTrials-1, 'type': 'boxplot'},
##                         #             {'lim': [30,160] if garivierMoulines else [30,60] , 'bbox':(.08, .60, .25, .25), 'distTrialPos': 59, 'type': 'boxplot'}
##                         #         ]
##                         #    }
##            },
##        {   'name': 'wR',
##            'vars': ['wR'],
##            'options': {'others':['fillIn' ]}
##            },
##        {   'name': 'cumRegret_hist',
##            'vars': ['cumRegret'],
##            'options': {'others':['addMinMax'],
##                        'addDist': [{'lim': [50,200] if garivierMoulines else [10,90], 'bbox':(.6, .05, .4, .4), 'distTrialPos': nTrials-1, 'type': 'hist'},
##                                    {'lim': [50,200] if garivierMoulines else [0,30], 'bbox':(.08, .55, .4, .4), 'distTrialPos': 59, 'type': 'hist'}]
##                        }
##            },
##        {   'name': 'cumRegret',
##            'vars': ['cumRegret'],
##            'options': {'others':['addMinMax'],
##                        'addDist': [{'lim': [50,200] if garivierMoulines else [10,90], 'bbox':(.6, .05, .4, .4), 'distTrialPos': nTrials-1, 'type': 'boxplot'},
##                                    {'lim': [50,200] if garivierMoulines else [0,30], 'bbox':(.08, .55, .4, .4), 'distTrialPos': 59, 'type': 'boxplot'}]
##                        }
##            },
##        {   'name': 'wCoherent',
##            'vars': ['wCoherent'],
##            'options': {'others':['fillIn'],
##                         #'addDist': {'lim': [0,1], 'bbox':(.55, .15, .4, .4), 'distTrialPos': nTrials/2-1, 'type': 'boxplot'},
##                         }
##            },
##        {   'name': 'rewards',
##            'vars': ['rewards'],
##            'options': {'others':['fillIn' ],
##                         # 'addDist': [{'lim': [50,200] if garivierMoulines else [0,1], 'bbox':(.6, .05, .4, .4), 'distTrialPos': nTrials-1, 'type': 'boxplot'},
##                         #             {'lim': [50,200] if garivierMoulines else [0,1], 'bbox':(.6, .55, .4, .4), 'distTrialPos': 59, 'type': 'boxplot'}
##                         #         ]
##                        }
##            },
##        {   'name': 'advantage',
##            'vars': ['advantage'],
##            'options': {'others':['fillIn' ],
##                         'addDist': [{'lim': [50,200] if garivierMoulines else [0,1], 'bbox':(.6, .05, .4, .4), 'distTrialPos': nTrials-1, 'type': 'boxplot'},
##                                     {'lim': [50,200] if garivierMoulines else [0,1], 'bbox':(.6, .55, .4, .4), 'distTrialPos': 59, 'type': 'boxplot'}
##                                 ]
##                        }
##            },
##        {   'name': 'filtRewards',
##            'vars': ['filtRewards'],
##            'options': {'others':['fillIn' ],
##                         'addDist': [{'lim': [50,200] if garivierMoulines else [0,1], 'bbox':(.6, .05, .4, .4), 'distTrialPos': nTrials-1, 'type': 'boxplot'},
##                                     {'lim': [50,200] if garivierMoulines else [0,1], 'bbox':(.6, .55, .4, .4), 'distTrialPos': 59, 'type': 'boxplot'}
##                                 ]
##                        }
##            },
##        {   'name': 'filtAdvantage',
##            'vars': ['filtAdvantage'],
##            'options': {'others':['fillIn' ],
##                         'addDist': [{'lim': [50,200] if garivierMoulines else [0,1], 'bbox':(.6, .05, .4, .4), 'distTrialPos': nTrials-1, 'type': 'boxplot'},
##                                     {'lim': [50,200] if garivierMoulines else [0,1], 'bbox':(.6, .45, .4, .4), 'distTrialPos': 59, 'type': 'boxplot'}
##                                 ]
##                        }
##            },
##        {   'name': 'regret',
##            'vars': ['wRegret'],
##            'options': {'others':['fillIn' ],
##                         # 'addDist': [{'lim': [50,200] if garivierMoulines else [0,1], 'bbox':(.6, .05, .4, .4), 'distTrialPos': nTrials-1, 'type': 'boxplot'},
##                         #             {'lim': [50,200] if garivierMoulines else [0,1], 'bbox':(.6, .45, .4, .4), 'distTrialPos': 59, 'type': 'boxplot'}
##                         #         ]
##                        }
##            },
##        {   'name': 'cumRewards',
##            'vars': ['cumRewards'],
##            'options': {'others':['addMinMax', 'legend'],
##                        'addDist': [{'lim': [50,200] if garivierMoulines else [10,60], 'bbox':(.08, .45, .4, .4), 'distTrialPos': 59, 'type': 'boxplot'},
##                                    {'lim': [50,200] if garivierMoulines else [50,120], 'bbox':(.6, .05, .4, .4), 'distTrialPos': nTrials-1, 'type': 'boxplot'},
##                                    ]
##                        }
##            },
##        {   'name': 'cumRewards_hist',
##            'vars': ['cumRewards'],
##            'options': {'others':['addMinMax', 'legend'],
##                        'addDist': [{'lim': [50,200] if garivierMoulines else [10,60], 'bbox':(.08, .55, .4, .4), 'distTrialPos': 59, 'type': 'hist'},
##                                    {'lim': [50,200] if garivierMoulines else [50,80], 'bbox':(.6, .05, .4, .4), 'distTrialPos': nTrials-1, 'type': 'hist'},
##                                    ]
##                        }
##            },
##        {   'name': 'performance',
##            'vars': ['performance'],
##            'options': {'others':[], 
##                        }
##            },
#         {   'name': 'wpAdvantage',
#             'vars': ['wpAdvantage'],
#             'options': {'others':['fillIn', 'addVar', 'addMinMax'],
#                         'addDist': {'lim': [0,1], 'bbox':(.55, .1, .4, .4), 'distTrialPos': nTrials-1, 'type': 'boxplot'}
#                         }
#             },
##        {   'name': 'cumAdvantage',
##            'vars': ['cumAdvantage'],
##            'options': {'others':['addMinMax', 'legend'],
##                        'addDist': [{'lim': [50,200] if garivierMoulines else [20,60], 'bbox':(.08, .55, .4, .4), 'distTrialPos': 59, 'type': 'boxplot'},
##                                    {'lim': [50,200] if garivierMoulines else [50,150], 'bbox':(.6, .05, .4, .4), 'distTrialPos': nTrials-1, 'type': 'boxplot'},
##                                    ]
##                        }
##            },
##        {   'name': 'cumPAdvantage',
##            'vars': ['cumPAdvantage'],
##            'options': {'others':['addMinMax', 'legend'],
##                        'addDist': [{'lim': [50,200] if garivierMoulines else [20,60], 'bbox':(.08, .55, .4, .4), 'distTrialPos': 59, 'type': 'boxplot'},
##                                    {'lim': [50,200] if garivierMoulines else [50,150], 'bbox':(.6, .05, .4, .4), 'distTrialPos': nTrials-1, 'type': 'boxplot'},
##                                    ]
##                        }
##            },
#         {   'name': 'pRegret',
#             'vars': ['pRegret'],
#             'options': {'others':['fillIn', 'addVar', 'addMinMax'],
#                          'addDist': [{'lim': [50,200] if garivierMoulines else [0,1], 'bbox':(.6, .05, .4, .4), 'distTrialPos': nTrials-1, 'type': 'boxplot'},
#                                      {'lim': [50,200] if garivierMoulines else [0,1], 'bbox':(.6, .55, .4, .4), 'distTrialPos': 59, 'type': 'boxplot'}
#                                  ]
#                         }
#             },
##        {   'name': 'fails',
##            'vars': ['fails'],
##            'options': {'others':['fillIn' ]}
##            },
##        {   'name': 'weights_1',
##            'vars': ['weights_0'],
##            'options': {'others':['addSem', 'legend' ],
##                        'addDist': {'lim': [0.49,0.61], 'bbox':(.55, .1, .4, .4), 'distTrialPos': 9, 'type': 'boxplot'}
##                        }
##            },
##        {   'name': 'weights_2',
##            'vars': ['weights_1'],
##            'options': {'others':['addSem', 'legend' ],
##                        'addDist': {'lim': [0.49,0.61], 'bbox':(.55, .1, .4, .4), 'distTrialPos': 9, 'type': 'boxplot'}
##                        }
##            },
##        {   'name': 'weights_3',
##            'vars': ['weights_2'],
##            'options': {'others':['addSem', 'legend' ],
##                        'addDist': {'lim': [0.49,0.61], 'bbox':(.55, .1, .4, .4), 'distTrialPos': 9, 'type': 'boxplot'}
##                        }
##            },
##        {   'name': 'weights_4',
##            'vars': ['weights_3'],
##            'options': {'others':['addSem', 'legend' ],
##                        'addDist': {'lim': [0.49,0.61], 'bbox':(.55, .1, .4, .4), 'distTrialPos': 9, 'type': 'boxplot'}
##                        }
##            },
        {   'name': 'weights',
            'vars': weights,
            'options': {'others':['fillIn'],
                        #'addDist': {'lim': [0,1], 'bbox':(.55, .1, .4, .4), 'distTrialPos': 9, 'type': 'boxplot'}
                        }
            },
##        {   'name': 'values_1',
##            'vars': ['values_0'],
##            'options': {'others':['fillIn', 'legend' ],
##                        'addDist': {'lim': [0,1], 'bbox':(.55, .1, .4, .4), 'distTrialPos': 100, 'type': 'boxplot'}
##                        }
##            },
##        {   'name': 'values_2',
##            'vars': ['values_1'],
##            'options': {'others':['fillIn', 'legend' ],
##                        'addDist': {'lim': [0,1], 'bbox':(.55, .1, .4, .4), 'distTrialPos': 100, 'type': 'boxplot'}
##                        }
##            },
##        {   'name': 'values_3',
##            'vars': ['values_2'],
##            'options': {'others':['fillIn', 'legend' ],
##                        'addDist': {'lim': [0,1], 'bbox':(.55, .1, .4, .4), 'distTrialPos': 100, 'type': 'boxplot'}
##                        }
##            },
##        {   'name': 'values_4',
##            'vars': ['values_3'],
##            'options': {'others':['fillIn', 'legend' ],
##                        'addDist': {'lim': [0,1], 'bbox':(.55, .1, .4, .4), 'distTrialPos': 100, 'type': 'boxplot'}
##                        }
##            },
##        {   'name': 'stdp_1',
##            'vars': ['stdp_0'],
##            'options': {'others':['fillIn', 'legend' ],
##                        'addDist': {'lim': [-0.01,0.01], 'bbox':(.55, .1, .4, .4), 'distTrialPos': 1, 'type': 'boxplot'}
##                        }
##            },
##        {   'name': 'stdp_2',
##            'vars': ['stdp_1'],
##            'options': {'others':['fillIn', 'legend' ],
##                        'addDist': {'lim': [-0.01,0.01], 'bbox':(.55, .1, .4, .4), 'distTrialPos': 1, 'type': 'boxplot'}
##                        }
##            },
##        {   'name': 'stdp_3',
##            'vars': ['stdp_2'],
##            'options': {'others':['fillIn', 'legend' ],
##                        'addDist': {'lim': [-0.01,0.01], 'bbox':(.55, .1, .4, .4), 'distTrialPos': 1, 'type': 'boxplot'}
##                        }
##            },
##        {   'name': 'stdp_4',
##            'vars': ['stdp_3'],
##            'options': {'others':['fillIn', 'legend' ],
##                        'addDist': {'lim': [-0.01,0.01], 'bbox':(.55, .1, .4, .4), 'distTrialPos': 1, 'type': 'boxplot'}
##                        }
##            },
##        {   'name': 'selectionTime',
##            'vars': ['motTime', 'cogTime'],
##            'options': {'others':['fillIn'],
##                        # 'addDist': {'lim': [-0.01,0.01], 'bbox':(.55, .1, .4, .4), 'distTrialPos': 1, 'type': 'boxplot'}
##                        }
##            },
        {   'name': 'clusterPerformance',
            'vars': ['wP', 'wR','wRegret','wCoherent'],
            'options': {'others':['fillIn'],
                        # 'addDist': {'lim': [-0.01,0.01], 'bbox':(.55, .1, .4, .4), 'distTrialPos': 1, 'type': 'boxplot'}
                        }
            },
]

if figExtensions:
    clustFigFolder = 'figures/clustered/'
    checkFolder(clustFigFolder)

print("\nClustered plots")

ls = ['-' for i in range(nAlt)]
for iAlt, alt in enumerate(shortAlt):
    if alt in strippedLines:
        ls[iAlt] = ':'

for gPlot in generalPlots:
    cluster_index = 0
    nVars = len(gPlot['vars'])            
    distOptions = gPlot['options'].get('addDist',None)
    if distOptions and (not isinstance(distOptions, list)):
        distOptions = [distOptions]
    fig, axs = plt.subplots(nVars,1,figsize=(6,2*nVars))

    print('\t -> '+gPlot['name'])

    for cI, cluster in enumerate(files):
        for dN_i in range(nVars):
            dN = gPlot['vars'][dN_i]
            startAt = len(cluster['meanVar']['mean'][dN]) - len(cluster['meanVar']['mean']['t'])
            dP_info = dataPlotInfo[commonNames[dN] if dN in commonNames else dN]
            if nVars == 1:
                axs_i = axs
            else:
                axs_i = axs[dN_i]

            mVarP = cluster['meanVar']['mean'][dN][startAt:] + cluster['meanVar']['var'][dN][startAt:]
            mVarM = cluster['meanVar']['mean'][dN][startAt:] - cluster['meanVar']['var'][dN][startAt:]

            mSemP = cluster['meanVar']['mean'][dN][startAt:] + cluster['meanVar']['sem'][dN][startAt:]
            mSemM = cluster['meanVar']['mean'][dN][startAt:] - cluster['meanVar']['sem'][dN][startAt:]

            if distOptions and cluster_index == 0:
                axs_insets = []
                for dOpt in distOptions:
                    axs_id = inset_axes(axs_i,width="100%",height="100%",bbox_to_anchor=dOpt['bbox'],bbox_transform=axs_i.transAxes)
                    axs_id.patch.set_alpha(0.6)
                    bbox = dict(boxstyle="round", fc="white", alpha=0.5)
                    # plt.setp(ax2.get_xticklabels(), bbox=bbox)
                    plt.setp(axs_id.get_xticklabels(), bbox=bbox)
                    plt.setp(axs_id.get_yticklabels(), bbox=bbox)
                    axs_insets.append(axs_id)

            axs_i.set_ylabel(dP_info['name'])
            axs_i.grid(alpha=0.4)
            if (dN_i == nVars-1) and (cluster == files[-1]):
                # axs_i.set_ylabel(dataPlotInfo[dN]['name'])
                axs_i.set_xlabel('Trial')
                # axs_i.set_ylabel('Amplitude')
                if len(cluster['meanVar']['reversal']):
                    for rv in cluster['meanVar']['reversal']:
                        axs_i.plot([rv,rv],axs_i.get_ylim(),'--',color=(0.3,0.3,0.3,0.5),linewidth=4,alpha=0.3)

            if useColors:
                axs_i.plot(cluster['meanVar']['mean']['t'],cluster['meanVar']['mean'][dN][startAt:],color=colors[cluster_index],alpha=0.6,linewidth=3.0,label=cluster['name'], ls=ls[cI])
                if 'fillIn' in gPlot['options']['others']:
                    axs_i.fill_between(cluster['meanVar']['mean']['t'],mVarM,mVarP,color=colors[cluster_index],alpha=0.6)
                if 'addVar' in gPlot['options']['others']:
                    axs_i.plot(cluster['meanVar']['mean']['t'],mVarM,color=colors[cluster_index],alpha=0.3,linewidth=3.0)
                    axs_i.plot(cluster['meanVar']['mean']['t'],mVarP,color=colors[cluster_index],alpha=0.3,linewidth=3.0)
                if 'addSem' in gPlot['options']['others']:
                    axs_i.fill_between(cluster['meanVar']['mean']['t'],mSemM,mSemP,color=colors[cluster_index],alpha=0.3)
                    # axs_i.plot(cluster['meanVar']['mean']['t'],mSemM,color=colors[cluster_index],alpha=0.3,linewidth=3.0)
                    # axs_i.plot(cluster['meanVar']['mean']['t'],mSemP,color=colors[cluster_index],alpha=0.3,linewidth=3.0)
                if 'addMinMax' in gPlot['options']['others']:
                    axs_i.plot(cluster['meanVar']['mean']['t'],cluster['meanVar']['min'][dN][startAt:],':',color=colors[cluster_index],alpha=0.6,linewidth=3.0)
                    axs_i.plot(cluster['meanVar']['mean']['t'],cluster['meanVar']['max'][dN][startAt:],':',color=colors[cluster_index],alpha=0.6,linewidth=3.0)

            else:
                axs_i.plot(cluster['meanVar']['mean']['t'],cluster['meanVar']['mean'][dN][startAt:],alpha=0.6,linewidth=3.0,label=cluster['name'], ls=ls[cI])
                if 'fillIn' in gPlot['options']['others']:
                    axs_i.fill_between(cluster['meanVar']['mean']['t'],mVarM,mVarP,alpha=0.6)
                if 'addVar' in gPlot['options']['others']:
                    axs_i.plot(cluster['meanVar']['mean']['t'],mVarM,alpha=0.3,linewidth=3.0)
                    axs_i.plot(cluster['meanVar']['mean']['t'],mVarP,alpha=0.3,linewidth=3.0)
                if 'addSem' in gPlot['options']['others']:
                    axs_i.plot(cluster['meanVar']['mean']['t'],mSemM,alpha=0.3,linewidth=3.0)
                    axs_i.plot(cluster['meanVar']['mean']['t'],mSemP,alpha=0.3,linewidth=3.0)
                if 'addMinMax' in gPlot['options']['others']:
                    axs_i.plot(cluster['meanVar']['mean']['t'],cluster['meanVar']['min'][dN][startAt:],':',alpha=0.6,linewidth=3.0)
                    axs_i.plot(cluster['meanVar']['mean']['t'],cluster['meanVar']['max'][dN][startAt:],':',alpha=0.6,linewidth=3.0)

            axs_i.set_xlim((0,nTrials-1))
            if dP_info.get('ylim',None):
                axs_i.set_ylim(dP_info['ylim'])

            if distOptions:
                for idx, dOpt in enumerate(distOptions):
                    axs_id = axs_insets[idx]
                    dist = np.array([cluster['meanVar']['all'][dN][c][int(dOpt['distTrialPos'])] for c in range(nExp)])
                    if np.isnan(dist).all():
                        continue
                    dist = dist[~np.isnan(dist)]
                    # b = np.linspace(dP_info['ylim'][0],dP_info['ylim'][1]/2,10)
                    if dOpt['type'] == 'hist':
                        b = np.linspace(dOpt['lim'][0],dOpt['lim'][1],20)
                        [hist,edges] = np.histogram(dist,bins=b)
                        # axs_id.bar(edges[:-1]+np.diff(edges)/2,hist,color=colors[cluster_index],alpha=0.3,width=1)
                        axs_id.fill_between(edges[:-1]+np.diff(edges)/2,0,hist,color=colors[cluster_index],alpha=0.2)
                        axs_id.plot(edges[:-1]+np.diff(edges)/2,hist,color=colors[cluster_index],alpha=0.5,linewidth=4)
                        axs_id.set_title('Histogram at $t='+str(dOpt['distTrialPos']+1)+"$",fontsize=10, bbox=dict(boxstyle="round", fc="white", alpha=0.5))
                        axs_id.set_ylabel('Counts',fontsize=10)
                        axs_id.set_xlabel('Amplitude',fontsize=10)
                    elif dOpt['type'] == 'boxplot':
                        bP = axs_id.boxplot(dist,positions=[cluster_index+1],notch=True,patch_artist=True,widths=0.75,medianprops={'color':'k'}) # ,boxprops={'color':colors[cluster_index]}
                        xlimits = [d+1 for d in range(len(alternatives))]
                        axs_id.set_xticks([0]+xlimits+[len(alternatives)+1])
                        axs_id.set_xticklabels(['']+['' for x in xlimits]+[''])
                        #axs_id.set_ylabel('Amplitude',fontsize=10)
                        #axs_id.set_xlabel('Amplitude',fontsize=10)
                        axs_id.yaxis.grid(True,alpha=0.4)
                        axs_id.set_title('Amplitude at $t='+str(dOpt['distTrialPos']+1)+"$",fontsize=10, bbox=dict(boxstyle="round", fc="white", alpha=0.5))
                        axs_id.set_ylim(dOpt['lim'])
                        for patch, color in zip(bP['boxes'], colors):
                            patch.set_facecolor(colors[cluster_index])
                            if ls[cI] != '-':
                                if cI == nAlt-1:
                                    patch.set_edgecolor('gray')
                                patch.set_hatch('///')
            # if ('legend' in gPlot['options']['others']) and (dN_i == nVars - 1):
            #     axs_i.legend(framealpha=0.6,loc='upper right')

        cluster_index += 1

    plt.tight_layout()
    for ext in figExtensions:
        figDataName = gPlot['name'] + '/'
        subFolder = clustFigFolder+figDataName
        checkFolder(subFolder)
        plt.savefig(subFolder+clustBase+ext)
    if showFig:
        if gPlot == generalPlots[-1]:
            plt.show(block=True)
        else:
            plt.show(block=False)
    else:
        plt.close(fig)


figLegend, ax = plt.subplots(2,1,figsize=(6,2))
params = dict(bottom=0, left=0, right=1)
figLegend.subplots_adjust(**params)
lines = []
for n in range(nAlt):
    if useColors:
        lines.append(ax[0].plot([n]*nAlt,[n]*nAlt,linewidth=4.0, color=colors[n], ls=ls[n]))
    else:
        lines.append(ax[0].plot([n]*nAlt,[n]*nAlt,linewidth=4.0, ls=ls[n]))
ax[0].legend(lines,labels=alternatives,framealpha=0.6,
            ncol=min((3,len(lines))),loc='upper center',)
ax[0].set_xlim((-2,-1))
#ax[1].set_title(plotInfo['name'],y=0.2)
for axis in ax:
    axis.set_axis_off()
    axis.margins(0)

figLegend.tight_layout()
for ext in figExtensions:
    figName = clustFigFolder+'legend-'+clustBase
    if figName[-4:] != ext:
        figName += ext
    plt.savefig(figName)
plt.close(figLegend)


########### Exp level boxplots
doGeneralBoxplots = askFlag('Should boxplots be done?', default='a')

if doGeneralBoxplots.askAndCheck():

    def newBoxEntry(name,var,options=[],folder='',violinPlot=False):
        if folder == '':
            folder = 'box-'+name
        if type(var) == str:
            var = [var]
        return {'name': name, 'vars': var, 'options': {'others': options, 'violin' : violinPlot}, 'folder': folder}

    generalPlots = []
    generalPlots.append(newBoxEntry('Completed trials','lenght'))
    generalPlots.append(newBoxEntry('Performance','wP',['lastTrial'],violinPlot=True))
    generalPlots.append(newBoxEntry('accSTDP','accSTDP',['lastTrial'],violinPlot=True))
    generalPlots.append(newBoxEntry('weights',weights,['lastTrial'],violinPlot=True))
    generalPlots.append(newBoxEntry('sClusterPerformance',['wP','wR','wRegret','wCoherent'],['lastTrial'],violinPlot=True))
    generalPlots.append(newBoxEntry('clusterPerformance',['cumPerformance','cumRewards','cumRegret','cumCoherent'],['lastTrial'],violinPlot=True))
    generalPlots.append(newBoxEntry('clusterPerformanceNorm',['cumPerformance','cumRewards','cumRegret','cumCoherent'],['lastTrial','normalize'],violinPlot=True))
    generalPlots.append(newBoxEntry('Accumulated regret','cumRegret',['lastTrial'],violinPlot=True))
    generalPlots.append(newBoxEntry('Accumulated rewards','cumRewards',['lastTrial'],violinPlot=True))
    generalPlots.append(newBoxEntry('Accumulated coherence','cumCoherent',['lastTrial','normalize'],violinPlot=True))
    generalPlots.append(newBoxEntry('performanceTrial20',['wP','wR','wRegret','wCoherent'],['singleTrial',20],violinPlot=True))
    generalPlots.append(newBoxEntry('performanceTrial60',['wP','wR','wRegret','wCoherent'],['singleTrial',60],violinPlot=True))
    generalPlots.append(newBoxEntry('performanceTrial90',['wP','wR','wRegret','wCoherent'],['singleTrial',90],violinPlot=True))
    generalPlots.append(newBoxEntry('wCoherent',['wCoherent'],['all'],violinPlot=True))
    generalPlots.append(newBoxEntry('wCoherentBox',['wCoherent'],['all']))
    generalPlots.append(newBoxEntry('Accumulated coherence - Box','cumCoherent',['lastTrial','normalize']))
    for i, pair in enumerate(pairsLabels):
        generalPlots.append(newBoxEntry('Choices_'+pair,pair,['all'],folder='pChoice',violinPlot=True))
        generalPlots.append(newBoxEntry('Choices_'+pair,pair,['all'],folder='pChoice_box'))

    if figExtensions:
        clustFigFolder = 'figures/clustered/'
        checkFolder(clustFigFolder)

    print("\nClustered plots")

    for gPlot in generalPlots:
        nVars = len(gPlot['vars'])     
        fig, axs = plt.subplots(nVars,1,figsize=(6,nVars*2))

        print('\t -> '+gPlot['name'])

        for cluster_index, cluster in enumerate(files):
            for dN_i in range(nVars):
                dN = gPlot['vars'][dN_i]
                dP_info = dataPlotInfo[commonNames[dN] if dN in commonNames else dN]

                if nVars == 1:
                    axs_i = axs
                else:
                    axs_i = axs[dN_i]

                if 'lastTrial' in gPlot['options']['others']:
                    lastTrials = [np.argmax(np.isnan(cluster['meanVar']['all'][dN][c]))-1 for c in range(nExp)]
                    dist = np.array([cluster['meanVar']['all'][dN][c][lastTrials[c]] for c in range(nExp)])
                    if 'normalize' in gPlot['options']['others']:
                        for c in range(nExp):
                            if lastTrials[c] < 0:
                                dist[c] /= nTrials
                            else:
                                dist[c] /= lastTrials[c]
                elif 'singleTrial' in gPlot['options']['others']:
                    trial = gPlot['options']['others'][gPlot['options']['others'].index('singleTrial')+1]
                    dist = np.array([])
                    lastTrials = [np.argmax(np.isnan(cluster['meanVar']['all'][dN][c]))-1 for c in range(nExp)]
                    for c in range(nExp):
                        if lastTrials[c] >= trial-1:
                            dist = np.append(dist,cluster['meanVar']['all'][dN][c][trial-1])
                elif 'between' in gPlot['options']['others']:
                    trials = gPlot['options']['others'][gPlot['options']['others'].index('between')+1]
                    print(trials)
                    dist = np.array([])
                    for c in range(nExp):
                        dist = np.concatenate((dist,cluster['meanVar']['all'][dN][c][trials[0]:trials[1]]))
                elif 'all' in gPlot['options']['others']:
                    dist = np.array([])
                    for c in range(nExp):
                        dist = np.concatenate((dist,cluster['meanVar']['all'][dN][c]))
                else:
                    dist = np.array(cluster['meanVar'][dN])

                expLen = len(dist)
                dist = dist[~np.isnan(dist)]
                expLenNoNaN = len(dist)

                if gPlot['options']['violin'] and expLenNoNaN:
                    vP = axs_i.violinplot(dist,positions=[cluster_index+1],widths=0.75) # ,boxprops={'color':colors[cluster_index]}
                    for vP_key in vP.keys():
                        if vP_key == 'bodies':
                            for vPobject in vP[vP_key]:
                                vPobject.set_color(colors[cluster_index])
                        else:
                            vP[vP_key].set_edgecolor(colors[cluster_index])
                            vP[vP_key].set_linestyle(ls[cluster_index])
                else:    
                    bP = axs_i.boxplot(dist,positions=[cluster_index+1],notch=True,patch_artist=True,widths=0.75,medianprops={'color':'k'}) # ,boxprops={'color':colors[cluster_index]}

                    for patch, color in zip(bP['boxes'], colors):
                        patch.set_facecolor('gray')

                    # if useColors:
                    #     for patch, color in zip(bP['boxes'], colors):
                    #         patch.set_facecolor(colors[cluster_index])
                    # else:
                    #     for patch, color in zip(bP['boxes'], colors):
                    #         patch.set_facecolor('gray')
                
                if cluster_index == N-1:
                    xlimits = [d+1 for d in range(N)]
                    axs_i.set_xticks([0]+xlimits+[N+1])
                    axs_i.set_xticklabels(['']+shortAlt+[''])
                    axs_i.yaxis.grid(True,alpha=0.4)
                    axs_i.set_ylabel("Normalized\n"+dP_info.get('ylabel','').lower() if 'normalize' in gPlot['options']['others'] else dP_info.get('ylabel',''))

                # axs_i.set_title(dP_info['name'],loc='left')
                # if (dN_i == nVars-1) and (cluster == files[-1]):
                # 
                #     axs_i.set_xlabel(clusterName)

                ylim = dP_info.get('ylim',None)
                if ylim:
                    if 'normalize' in gPlot['options']['others']:
                        axs_i.set_ylim([0,1])
                    else:
                        axs_i.set_ylim(ylim)

                # if expLen != len(dist): # there were nan values
                #     ylim = axs_i.get_ylim()
                #     axs_i.text(cluster_index+1,1.03*ylim[1],'#'+str(len(dist)),fontsize=8,ha='center')#+'/'+str(expLen),fontsize=8,ha='center')

                if ('legend' in gPlot['options']['others']) and (dN_i == nVars - 1):
                    axs_i.legend(framealpha=0.6,loc='upper right')

        plt.tight_layout()
        for ext in figExtensions:
            figDataName = gPlot['folder'] + '/'
            subFolder = clustFigFolder+figDataName
            checkFolder(subFolder)
            plt.savefig(subFolder+clustBase+ext)
        if showFig:
            if gPlot == generalPlots[-1]:
                plt.show(block=True)
            else:
                plt.show(block=False)
        else:
            plt.close(fig)

# Open figures folder
doOpen = askFlag('Open figures folder?', default='n')
if doOpen.askAndCheck():
    bashCmd = ['xdg-open',os.getcwd()+'/figures/']
    process = subprocess.Popen(bashCmd, stdout=subprocess.PIPE)