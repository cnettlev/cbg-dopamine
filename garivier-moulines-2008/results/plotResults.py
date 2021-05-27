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
import scipy

matplotlib.rcParams.update(
    {
        'text.usetex': False,
        'font.family': 'stixgeneral',
        'mathtext.fontset': 'stix',
        'patch.edgecolor' : 'none'
    }
)

# tableau colors
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

# plt.style.use('seaborn-dark-palette')

garivierMoulines = False
showFig = False
datafolder = 'data/'
figfolder = 'figures/'
figExtensions = ['.png', '.pdf']

baseBase = 'CBG-DA_' + ('gM_' if garivierMoulines else '')

nameBase = [ baseBase + 'c_cLTD_d_pAdv_rv_r60_N0.7_4.0',
             baseBase + 'c_cLTD_rv_r60_N0.7_4.0',
             baseBase + 'c_cLTD_rv_r60_N0.7_6.0',
             baseBase + 'c_cLTD_rv_r60_N0.7_8.0',]
#nameBase = ['DA_c_cLTD_r100']
ltdLtp   = '_0.0002_3e-05_'
# wNoise   = ['0.025','0.05','0.075','']
DA       = ['4.0', '4.0', '6.0', '8.0']
#DA = ['8.0']
nExp     = 3 if garivierMoulines else 100
nTrials  = 1000 if garivierMoulines else 120

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

    cluster = nB+ltdLtp
    parameters = {'nameBase': nB}
    clusterData = {'fName': cluster, 'data': [], 'files': [], 'params': parameters}
    clusterData['name'] = 'Dynamic $DA_T$' if '_d_' in cluster else '$DA_T$ at ' + da + ' [sp/s]'
    clusterData['cname'] = clusterData['name'].replace('$','').replace('_T','')
    for n in range(nExp):
        clusterData['files'].append(cluster+'{:03d}'.format(n))
    files.append(clusterData)
    # break

N = len(files)
cues = range(3) if garivierMoulines else range(4)
weights = ['weights_'+str(c) for c in cues]
values  = ['values_'+str(c) for c in cues]
cues_rewardsL  = ['cues_r_'+str(c) for c in cues]
pairs = cues if garivierMoulines else [[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]]
pairsLabels = ['Cue '+ str(c) for c in pairs] if garivierMoulines else ['Cues '+str(p[0])+'-'+str(p[1]) for p in pairs]

dataDict = {
            'selection'    : [ ], 
            'nselected'    : [ ], 
            'wSelection'   : [ ],
            'performance'  : [ ],
            'wP'           : [ ],
            'rewards'      : [ ],
            'wR'           : [ ],
            'reversal'     : [60],
            'regret'       : [ ],
            'cumRegret'    : [ ],
            'wRegret'      : [ ],
            'cumCoherent'  : [ ], 
            'coherent'     : [ ], 
            'wCoherent'    : [ ], 
            'SNc'          : [ ],
            'advantage'    : [ ],
            'wAdvantage'   : [ ],
            'pAdvantage'   : [ ],
            'filtAdvantage': [ ],
            't'            : range(nTrials),
            }

dataPlotInfo = {
            'advantage'    : { 'name': 'Advantage', 'color': '', 'ylim': [0,1], 'legend': ['Advantage']},
            'wAdvantage'   : { 'name': 'Advantage', 'color': '', 'ylim': [0,1], 'legend': ['Advantage']},
            'pAdvantage'   : { 'name': 'Perceived Advantage', 'color': '', 'ylim': [0,1], 'legend': ['Advantage']},
            'filtAdvantage': { 'name': 'Advantage (filtered)', 'color': '', 'ylim': [0,1], 'legend': ['Advantage']},
            'SNc'          : { 'name': 'SNc', 'color': 'tab:blue', 'ylim': [0,12], 'legend': ['SNc [sp/s]']},
            'wRegret'      : { 'name': 'Regret', 'color': '', 'ylim': [0,1], 'legend': ['Regret']},
            'regret'       : { 'name': 'Regret', 'color': 'r', 'ylim': [0,1], 'legend': ['Regret']},
            'cumRegret'    : { 'name': 'Accumulated regret',  'color': 'r', 'ylim': [0,200], 'legend': ['Accum. regret' ]},
            'wSelection'   : { 'name': 'Selections', 'color': 'b', 'legend': [ 'Mean selections']},
            'selection'    : { 'name': 'Selected cues', 'color': 'magenta', 'legend': ['Selected']},
            'nselected'    : { 'name': 'Unselected cues',  'color': 'r', 'legend': ['Not selected' ]},
            'performance'  : { 'name': 'Performance', 'color': 'b', 'legend': ['Performance' ]},
            'wP'           : { 'name': 'Performance', 'color': 'b', 'ylim': [0,1], 'legend': ['Mean performance']},
            'rewards'      : { 'name': 'Rewards', 'color': 'g', 'legend': ['Reward' ]},
            'wR'           : { 'name': 'Rewards', 'color': 'g', 'ylim': [0,1], 'legend': ['Mean rewards']},
            'weights'      : { 'name': 'Weights', 'color': '', 'ylim': [0.4,0.6], 'legend': ['$w_'+str(c)+'$' for c in cues]},
            'values'       : { 'name': 'Values', 'color': '', 'ylim': [0,1], 'legend': ['$v_'+str(c)+'$' for c in cues]},
            'probChoice'   : { 'name': 'Choice preferences', 'color': '', 'ylim': [0,1], 'legend': pairsLabels},
            'coherent'     : { 'name': 'Coherent selections', 'color':'', 'ylim': [0,2], 'legend': ['Coh. selections']},
            'wCoherent'    : { 'name': 'Coherent selections', 'color':'', 'ylim': [0,2], 'legend': ['Coh. selections']},
            'cumCoherent'  : { 'name': 'Coherent selections', 'color':'tab:orange', 'ylim': [0,200], 'legend': ['Coherent selections','Accum. coherent']},
            'cues_rewards'  : { 'name': 'Cues rewards', 'color': '', 'ylim': [0,1], 'legend': ['$cue_'+str(c+1)+'$' for c in cues]},
            }

other_colors = []

data = []
commonNames = {'cumCoherent':'cumCoherent', 'coherent':'coherent', 'wCoherent':'wCoherent'}
for c in cues:
    dataDict[weights[c]] = []
    dataDict[values[c]] = []
    commonNames[weights[c]] = weights[c]
    commonNames[values[c]] = values[c]#'values'
    dataPlotInfo[weights[c]] = { 'name': 'Weights', 'legend': None, 'color': colors[0:len(cues)], 'ylim': [0.45,0.55]}
    dataPlotInfo[values[c]] = { 'name': 'Values', 'legend': None, 'color': colors[0:len(cues)], 'ylim': [0,1]}

    dataDict[cues_rewardsL[c]] = []
    commonNames[cues_rewardsL[c]] = 'cues_rewards'

for pL in pairsLabels:
    dataDict[pL] = []
    commonNames[pL] = 'probChoice'
    # dataPlotInfo[pL] = { 'name': 'Pairs', 'legend': None, 'color': colors[0:len(cues)], 'ylim': [0,1]}

eps = np.finfo(float).eps

def checkFolder(dir):
    if not os.path.isdir(dir):
        os.makedirs(dir)

def weighted(prev,new,alpha=0.8):
    return alpha*prev+(1-alpha)*new

def doPlot(data,plotVars, block=False):
    def plotMeanVar(ax,x,mean=None,var=None,min=None,max=None,all=None,lbl=None,color='b',reversal='',ylim='',xlim=''):
        if color:
            if mean is not None:
                if var is not None:
                    ax.fill_between(x,mean+var,mean-var,alpha=0.5,color=color)
                ax.plot(x,mean,color=color,alpha=0.6,linewidth=4.0)
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
                ax.plot(x,mean,alpha=0.6,linewidth=4.0)
            if min is not None and max is not None:
                ax.fill_between(x,max,min,alpha=0.1)
            if all is not None:
                ax.plot(x,all,alpha=0.4,linewidth=2.0)

        if lbl:
            ax.legend(labels=lbl,framealpha=0.6,ncol=len(cues),loc='upper right')
        if ylim:
            ax.set_ylim(ylim)
        if xlim:
            ax.set_xlim(xlim)

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

        fig, axs = plt.subplots(nPlots,1,figsize=(8,6))

        if nPlots == 1:
            axs = [axs]

        currentColor = 0

        for var in figVars['vars']:
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

            plotMeanVar(
                ax = axs[n_axis],
                x = data['mean']['t'],
                mean = data['mean'][var][startAt:] if var in addMean else None,
                var = data['var'][var][startAt:] if var in addMean and not var in avoidVar else None,
                min = data['min'][var][startAt:] if var in addMinMax else None,
                max = data['max'][var][startAt:] if var in addMinMax else None,
                all = np.transpose(data['all'][var])[startAt:] if var in addAll else None,
                lbl = plotInfo['legend'],
                color = plotInfo['color'][currentColor] if isinstance(plotInfo['color'], list) else plotInfo['color'],
                reversal = data['mean']['reversal'],
                ylim = plotInfo['ylim'] if 'ylim' in plotInfo else '',
                xlim = (data['mean']['t'][0],data['mean']['t'][-1])
                )

        for ax in axs:
            ylim = ax.get_ylim()
            if len(data['mean']['reversal']):
                for rv in data['mean']['reversal']:
                    ax.plot([rv,rv],ylim,'--',color=(0.3,0.3,0.3,0.5),linewidth=4,alpha=0.3)
            ax.grid()

        plt.suptitle(data['name'],x=0.8)

        plt.tight_layout()

        for ext in figExtensions:
            figsubfolder = figfolder+figVars['title']+'/'
            checkFolder(figsubfolder)
            figName = figsubfolder+figVars['title']+data['fName'].replace('.csv',ext)
            if figName[-4:] != ext:
                figName += ext
            
            plt.savefig(figName)
        if showFig:
            plt.show(block=block)
        else:
            plt.close(fig)

def getStatistics(meanData,varData,minData,maxData,allData,dataArray,keys,mV=[]):
    allKeys = []
    for k in keys:
        allKeys += k['vars']
    allKeys += mV

    noDuplicates = []
    [noDuplicates.append(x) for x in allKeys if x not in noDuplicates]

    for k in noDuplicates:
        for data in dataArray:
            allData[k].append(data[k])        

        if k in addMinMax:
            minData[k] = np.min(allData[k],axis=0)
            maxData[k] = np.max(allData[k],axis=0)

        varData[k] = np.var(allData[k],axis=0)
        meanData[k] = np.mean(allData[k],axis=0)

        allData[k] = np.array(allData[k])

addMean = ['SNc','wP','wR','regret','cumRegret','cumCoherent','wCoherent','wAdvantage','wRegret','pAdvantage','satAdvantage_','filtAdvantage']+cues_rewardsL+values+weights+pairsLabels
avoidVar = ['wCoherent','cumRegret','wRegret']
addMinMax = ['wP','wR','wRegret']+cues_rewardsL
addAll = ['wP']+weights+values+['wCoherent','SNc','cumRegret','wAdvantage']

plotVars = [{'vars': ['wP','cumRegret','wCoherent','cumCoherent'], 'title': 'perf_regret_cselections_'},
            {'vars': weights, 'title': 'weights_'},
            {'vars': values, 'title': 'values_'},
            {'vars': ['SNc'], 'title': 'dopamine_'},
            {'vars': ['cumRegret'], 'title': 'cumRegret_'},
            {'vars': ['wAdvantage'], 'title': 'wAdvantage_'},
            {'vars': ['pAdvantage'], 'title': 'percAdvantage_'},
            {'vars': ['filtAdvantage'], 'title': 'filtAdvantage_'},
            {'vars': ['wRegret'], 'title': 'regret_'},
            {'vars': cues_rewardsL, 'title': 'cues_R_'},
            {'vars': pairsLabels, 'title': 'pchoice_'},
            #{'vars': weights+values, 'title': 'weights_values_'}
            ]

checkFolder(figfolder)

moreVarsForStatistics = ['wR','regret']

collectedData = []
for dataFiles in files:
    print("Reading "+datafolder+dataFiles['cname'])

    for dataFile in dataFiles['files']:
        # print '\t:',dataFile
        data_i = copy.deepcopy(dataDict)
        data_i['fName'] = dataFile

        data_i['wSelection'].append(0)
        data_i['wP'].append(0)
        data_i['wR'].append(0)
        data_i['cumRegret'].append(0)
        data_i['cumCoherent'].append(0)
        data_i['wCoherent'].append(0)
        data_i['wAdvantage'].append(0)
        data_i['pAdvantage'].append(0)
        data_i['wRegret'].append(0)

        with open(datafolder+dataFile) as file:
            reader = csv.DictReader(file,delimiter=',')

            if ( len(list(reader)) < nTrials):
                continue
            file.seek(0)
            reader.__init__(file, delimiter=',')

            trial = 0
            currentCuesRewards = cues_reward
            nreversal = [r-1 for r in data_i['reversal']]

            for row in reader:
                data_i['selection'].append(float(row['choice']))
                data_i['wSelection'].append(weighted(data_i['wSelection'][-1],float(row['choice'])))
                data_i['nselected'].append(float(row['nchoice']))

                data_i['performance'].append(float(row['P']))
                data_i['wP'].append(weighted(data_i['wP'][-1],float(row['P'])))

                data_i['rewards'].append(float(row['R'] == 'True'))
                data_i['wR'].append(weighted(data_i['wR'][-1],data_i['rewards'][-1]))

                # data_i['SNc'].append(float(row['SNc'][2:-2]))
                data_i['SNc'].append(float(row['SNc']))

                if trial in nreversal:
                    # print "change at",trial,"from "
                    # print currentCuesRewards
                    currentCuesRewards = cues_rewards[np.sum(nreversal<=trial)]
                    # print "to:"
                    # print currentCuesRewards

                trial += 1

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
                c_rewards = [float(r) for r in row['c_rewards'].split('\t')]
                for c in cues:
                    data_i[weights[c]].append(weights_r[c])
                    data_i[values[c]].append(values_r[c])
                    data_i[cues_rewardsL[c]].append(c_rewards[c])

                if garivierMoulines:
                    pReg = np.max(values_r) - values_r[choice]
                    pAdv = 1-pReg
                else:
                    pReg = np.max([values_r[choice],values_r[nchoice]]) - values_r[choice]
                    pAdv = 1-pReg

                data_i['pAdvantage'].append(pAdv)
                data_i['filtAdvantage'].append(np.max((data_i['pAdvantage'][-1]-0.6,0))/(1-0.6))

                probChoice = row['pchoice'].split('\t')
                for i, pL in enumerate(pairsLabels):
                    data_i[pL].append(float(probChoice[i]))

                if garivierMoulines:
                    weights_f = [float(w) for w in weights_r]
                    data_i['coherent'].append(0 + ( weights_f[choice] >= (np.max(weights_f) - eps) ) )
                else:
                    data_i['coherent'].append(0+(float(weights_r[choice]) > float(weights_r[nchoice])))
                data_i['wCoherent'].append(weighted(data_i['wCoherent'][-1],data_i['coherent'][-1]))
                data_i['cumCoherent'].append(data_i['cumCoherent'][-1]+data_i['coherent'][-1])

        data_i['coherent'] = [ c * np.mean(data_i['cumCoherent']) for c in data_i['coherent'] ]
        data_i['wCoherent'] = [ c for c in data_i['wCoherent'] ]
        dataFiles['data'].append(data_i)

    meanData = copy.deepcopy(dataDict)
    varData  = copy.deepcopy(dataDict)
    minData  = copy.deepcopy(dataDict)
    maxData  = copy.deepcopy(dataDict)
    allData  = copy.deepcopy(dataDict)

    getStatistics(meanData,varData,minData,maxData,allData,dataFiles['data'],plotVars,moreVarsForStatistics)

    data = {
            'mean': meanData, 
            'var' : varData, 
            'min' : minData,
            'max' : maxData,
            'all' : allData,
            'nExp': len(dataFiles['data']),
            'name': dataFiles['name']+' N:'+str(nExp),
            'fName': dataFiles['fName'],
            'reversal': dataDict['reversal'],
            }

    dataFiles['meanVar'] = data

    doPlot(data,plotVars,False)

# Get statistics
###############

def printFunc(f):
    print f.__code__.co_varnames
    print f.__defaults__

def doWilcoxon(x,y,strdiff,xname,yname):
    w = wilcoxon(x,y)
    if w[1] < 0.05:
        strdiff[0] = np.concatenate((strdiff[0],["[W] "+yname+" - (p="+str(w[1])+")"]))
        strdiff[1] = np.concatenate((strdiff[1],["[W] "+xname+" - (p="+str(w[1])+")"]))

    return strdiff

def doMannwhitneyu(x,y,strdiff,xname,yname,alternatives=['less','greater','two-sided']):
    for alt in alternatives:
        w = mannwhitneyu(x,y,alternative=alt)
        if w[1] < 0.05:
            strdiff[0] = np.concatenate((strdiff[0],["[U-"+alt.upper()[0]+"] "+yname+" - (p="+str(w[1])+")"]))
            strdiff[1] = np.concatenate((strdiff[1],["[U-"+alt.upper()[0]+"] "+xname+" - (p="+str(w[1])+")"]))

    return strdiff


printStatistics = []#['wP','cumRegret', 'wCoherent','cumCoherent']
statistics = [
              {'name':'wilcoxon','func':doWilcoxon},
              {'name':'mannwhitneyu','func':doMannwhitneyu}
              ]

for st in printStatistics:
    print "\n\n** Getting statistics for",st,"**"
    lastValues = []
    allValues = [np.array([]) for f in range(len(files))]

    for cluster_i in range(len(files)):
        lastValues_i = [files[cluster_i]['meanVar']['all'][st][rep][-1] for rep in range(nExp)]
        allValues[cluster_i] = np.concatenate((allValues[cluster_i],files[cluster_i]['meanVar']['all'][st][rep]))
        lastValues.append(lastValues_i)

    for st_type in statistics:
        differsLV = [np.array([]) for f in range(len(files))]
        differsALL = [np.array([]) for f in range(len(files))]
        names = [n['cname'] for n in files]
        for c1 in range(len(files)):
            for c2 in range(len(files)):
                if c1 >= c2:
                    continue

                [differsLV[c1],differsLV[c2]] = st_type['func'](lastValues[c1],lastValues[c2],[differsLV[c1],differsLV[c2]],files[c1]['cname'],files[c2]['cname'])

                [differsALL[c1],differsALL[c2]] = st_type['func'](lastValues[c1],lastValues[c2],[differsALL[c1],differsALL[c2]],files[c1]['cname'],files[c2]['cname'])
               
        def printStats(diff,clustersNames=names):
            for df_i in range(len(diff)):
                print '\t\t'+names[df_i]
                for line in diff[df_i]:
                    print '\t\t\t'+line


        print st+" -- "+st_type['name']
        print "\tDiffers in last values: "
        printStats(differsLV)
        print
        print st+" -- "+st_type['name']
        print "\tDiffers in all values: "
        printStats(differsALL)

# Plots comparing clusters
##########################

useColors = len(colors) > len(files)

generalPlots = [['wP'], # Fig 1
                ['wR'], # Fig 2
                ['cumRegret'],
                ['wCoherent'],
                ]

fillIn = ['wP','wR','wCoherent']
avoidVar = ['wCoherent','cumRegret']
addLastDist = []#'cumRegret','wCoherent']
lastDistLimits = {'wCoherent': {'lim': [0,1], 'bbox':(.55, .15, .4, .4)},
                  'cumRegret': {'lim': [20,40], 'bbox':(.15, .55, .4, .4)}
                  }


for plotD in range(len(generalPlots)):
    dataName = generalPlots[plotD]
    cluster_index = 0
    fig, axs = plt.subplots(len(dataName),1,figsize=(6,4))

    for cluster in files:
        for dN_i in range(len(dataName)):
            dN = dataName[dN_i]
            startAt = len(cluster['meanVar']['mean'][dN]) - len(cluster['meanVar']['mean']['t'])
            if len(dataName) == 1:
                axs_i = axs
            else:
                axs_i = axs[dN_i]
            
            mVarP = cluster['meanVar']['mean'][dN][startAt:] + cluster['meanVar']['var'][dN][startAt:]
            mVarM = cluster['meanVar']['mean'][dN][startAt:] - cluster['meanVar']['var'][dN][startAt:]

            if dN in addLastDist and cluster_index == 0:
                axs_id = inset_axes(axs_i,width="100%",height="100%",bbox_to_anchor=lastDistLimits[dN]['bbox'],bbox_transform=axs_i.transAxes)
                    # width="30%",
                    # height="30%",
                    # loc=6) # upper left

            if dN in fillIn:
                if useColors:
                    axs_i.fill_between(cluster['meanVar']['mean']['t'],mVarM,mVarP,color=colors[cluster_index],alpha=0.6,label=cluster['name'])
                else:
                    axs_i.fill_between(cluster['meanVar']['mean']['t'],mVarM,mVarP,alpha=0.6,label=cluster['name'])
            else:
                if useColors:
                    axs_i.plot(cluster['meanVar']['mean']['t'],cluster['meanVar']['mean'][dN][startAt:],color=colors[cluster_index],alpha=0.6,linewidth=5.0,label=cluster['name'])
                    if not dN in avoidVar:
                        axs_i.plot(cluster['meanVar']['mean']['t'],mVarM,color=colors[cluster_index],alpha=0.3,linewidth=3.0)
                        axs_i.plot(cluster['meanVar']['mean']['t'],mVarP,color=colors[cluster_index],alpha=0.3,linewidth=3.0)

                else:
                    axs_i.plot(cluster['meanVar']['mean']['t'],cluster['meanVar']['mean'][dN][startAt:],alpha=0.6,linewidth=5.0,label=cluster['name'])
                    if not dN in avoidVar:
                        axs_i.plot(cluster['meanVar']['mean']['t'],mVarM,alpha=0.3,linewidth=3.0)
                        axs_i.plot(cluster['meanVar']['mean']['t'],mVarP,alpha=0.3,linewidth=3.0)

            if dN in addLastDist:
                dist = [cluster['meanVar']['all'][dN][c][-1] for c in range(nExp)]
                # b = np.linspace(dataPlotInfo[dN]['ylim'][0],dataPlotInfo[dN]['ylim'][1]/2,10)
                b = np.linspace(lastDistLimits[dN]['lim'][0],lastDistLimits[dN]['lim'][1],20)
                [hist,edges] = np.histogram(dist,bins=b)
                # axs_id.bar(edges[:-1]+np.diff(edges)/2,hist,color=colors[cluster_index],alpha=0.3,width=1)
                axs_id.fill_between(edges[:-1]+np.diff(edges)/2,0,hist,color=colors[cluster_index],alpha=0.2)
                axs_id.plot(edges[:-1]+np.diff(edges)/2,hist,color=colors[cluster_index],alpha=0.5,linewidth=4)
                axs_id.set_title('Histogram at $t='+str(nTrials)+"$",fontsize=10)
                axs_id.set_ylabel('Counts',fontsize=10)
                axs_id.set_xlabel('Amplitude',fontsize=10)

            # axs_i.set_ylim(dataPlotInfo[dN]['ylim'])

            if dN_i == 0:
                axs_i.set_title(dataPlotInfo[dN]['name'])
                axs_i.set_xlabel('Trial')
                axs_i.set_ylabel('Amplitude')

            if dN_i == len(dataName) - 1:
                axs_i.legend(framealpha=0.6,loc='upper right')

        cluster_index += 1

    plt.tight_layout()
    for ext in figExtensions:
        plt.savefig('figures/clustered_'+('gM_' if garivierMoulines else '')+'_'.join([dN for dN in dataName])+ext)
    if showFig:
        if plotD == len(generalPlots)-1:
            plt.show(block=True)
        else:
            plt.show(block=False)
