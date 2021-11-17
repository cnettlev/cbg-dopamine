from dana import *
import itertools
from learningOptions import *       # Added argument options parser
from learningHardCodedOptions import *       # Added argument options parser
from learningNeuralPopulations import *

def flip(items, ncol):
    return itertools.chain(*[items[i::ncol] for i in range(ncol)])

cogMarker = motMarker = movMarker = []
plotSteps = 1*millisecond
nData = 2*n
nData2 = 2*n
nDataSnc = SNc_neurons
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
            borderaxespad=0, framealpha=0.4) #frameon=False)
    else:
        axs.legend(signals,loc=loc, ncol=n, fontsize='x-small',borderaxespad=0, framealpha=0.4) #frameon=False)

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
    neuralData_y_2[currentTrial] = np.mean(SNc_dop['V'])
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
    
    neuralData_ysncpptn[index][0:SNc_neurons] = [SNc_dop['V'][i][0] for i in range(SNc_neurons)]
    neuralData_ysncpptn[index][SNc_neurons] = SNc_dop['Ir'][0]
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