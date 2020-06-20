#!/usr/bin/env python
from joblib import Parallel, delayed
import multiprocessing
import subprocess
import sys
import numpy as np
from time import sleep

additionalOptions = ' -g -t 200 -b 12 --ltd-constant --correlatedNoise -S --relativeValue' # Increase the trials to 600, disable learning, use regular patterns -r100 --dynamicDA 

parallelDivisions = False 

plotNeuralData = False

firstDynamic = True
DA_DIV = np.array([5.0,4.0,5.0,6.0]) # np.array([6.0,4.0,6.0,8.0]) #np.arange(0.0,2.6,0.5)
DIVISIONS = DA_DIV.shape[0]
TIMES = 50
baseNumber = 0 
NJOBS = 50
WEIGHTS = None # np.array([["0.61414980 0.52987340 0.46941890 0.43399170","0.61561350 0.53347860 0.47005700 0.43940460","0.61691820 0.53227470 0.46791240 0.43998220","0.61740550 0.53296140 0.46909390 0.44151160","0.61779960 0.53731250 0.46666480 0.44043540","0.61750240 0.53647780 0.46687310 0.43947310"]])
#STDP = np.array([[0.0000025,1.5*0.000075],[0.0000025,0.5*0.000075],[2*0.0000025,0.000075]])


cLTD = None # np.array([0.0001,0.00005,0.00001])
#STDP = np.array([[0.0000015,0.0000375],[0.000002,0.0000375],[0.0000025,0.0000375],[0.000003,0.0000375],[0.0000035,0.0000375],[0.000004,0.0000375],[0.0000045,0.0000375],[0.000005,0.0000375],
#                 [0.0000015,0.00005],[0.000002,0.00005],[0.0000025,0.00005],[0.000003,0.00005],[0.0000035,0.00005],[0.000004,0.00005],[0.0000045,0.00005],[0.000005,0.00005],
#                 [0.0000015,0.000075],[0.000002,0.000075],[0.0000025,0.000075],[0.000003,0.000075],[0.0000035,0.000075],[0.000004,0.000075],[0.0000045,0.000075],[0.000005,0.000075]])

ltdStep = 0.0000005
ltpStep = 0.00002
(ltdMin,ltdMax) = (0.0000015,0.000005)
LTD = np.array([0.000004])# np.arange(0.000005-2*ltdStep,0.000005+4*ltdStep,ltdStep) # np.array([0.000005]) #
LTP = np.array([0.00009])#np.arange(0.00007,0.00007+4*ltpStep,ltpStep) # np.array([0.00003])#[0.0000375,0.00005,0.0000625,0.000075])
#LTP = np.append(LTP,0.0005)
STDP_factor = 1

STDP = np.zeros((len(LTD)*len(LTP),2))
index = 0
for ltd in LTD:
    for ltp in LTP:
        STDP[index] = [ltd*STDP_factor,ltp*STDP_factor]
        index += 1

NOISE = np.array([0.01])#,0.2,0.4])

if len(sys.argv) > 1:
    folder = sys.argv[1]
    additionalOptions += ' -F'+folder
else:
    folder = ''

if len(sys.argv) > 2:
    factor = float(sys.argv[2])
    additionalOptions += ' -X '+str(10*factor)+' -Y '+str(factor)

if len(sys.argv) > 3: #remaining options introduced as argument
    additionalOptions += ' '+sys.argv[3]

def createList(process):
    if type(process) is str:
        process = [process]
    return process

def addSTDP(processes):
    processes = createList(processes)
    output = []
    for p in processes:
        for stdp in range(STDP.shape[0]):
            output.append(p + ' -l' + str(STDP[stdp,0]) + ' -L' + str(STDP[stdp,1]))
    return output

def addNoise(processes):
    processes = createList(processes)
    output = []
    for p in processes:
        for n in range(NOISE.shape[0]):
            output.append(p + ' -N ' + str(NOISE[n]))
    return output

def addPlotting(processes):
    processes = createList(processes)
    output = []
    for p in processes:
        output.append(p + ' --flashPlots 50 --storePlots 50')
    return output

# def runParalelProcess(processes):
#     Parallel(n_jobs=NJOBS)(delayed(runProcess)(p) for p in processes)

def runProcess(process):
    if type(process) is not str: # is a list
        #for p in process:
        #    runProcess(p)
        Parallel(n_jobs=len(process))(delayed(runProcess)(p) for p in process)
        return
    p = subprocess.Popen(process+additionalOptions,stderr=subprocess.STDOUT, stdout=subprocess.PIPE, shell=True)
    print "Running ",process+additionalOptions
    p.communicate()
    print "Finished ",process+additionalOptions
    #return p

def runModelLearning(i):
    #print "*****    j:",DA
    forRange = TIMES if parallelDivisions else DIVISIONS
    for j in range(forRange):

        time = j if parallelDivisions else i
        if DA_DIV is not None:
            DA = DA_DIV[i] if parallelDivisions else DA_DIV[j]
        else:
            DA = 0.5*float(i) if parallelDivisions else 0.5*float(j) #/float(DIVISIONS)

        bProcess = 'python learning.py -d'+str(DA)+' -f'+str(time + baseNumber).zfill(3)
        if firstDynamic and j == 0:
            bProcess += ' --dynamicDA'

        p = addNoise(bProcess)
        p = addSTDP(p)

        if time == 0 and plotNeuralData:
            p = addPlotting(p)

        runProcess(p)

if parallelDivisions:
    Parallel(n_jobs=NJOBS)(delayed(runModelLearning)(i) for i in range(DIVISIONS))
else:
    Parallel(n_jobs=NJOBS)(delayed(runModelLearning)(i) for i in range(TIMES))
