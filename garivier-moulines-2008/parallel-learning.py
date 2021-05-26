#!/usr/bin/env python
from joblib import Parallel, delayed
import multiprocessing
import subprocess
import sys
import numpy as np
from time import sleep
from parallelOptions import options

additionalOptions = ' --ltd-constant --relativeValue -S ' + options.aP

parallelDivisions = options.pD

logFile = options.lF

plotNeuralData = options.pN

firstDynamic = options.fD
DA_DIV = np.array([4.0])#, 4.0, 6.0, 8.0])
DIVISIONS = DA_DIV.shape[0]
TIMES = options.t
baseNumber = options.bN
NJOBS = options.nJ
WEIGHTS = None # np.array([["0.61414980 0.52987340 0.46941890 0.43399170","0.61561350 0.53347860 0.47005700 0.43940460","0.61691820 0.53227470 0.46791240 0.43998220","0.61740550 0.53296140 0.46909390 0.44151160","0.61779960 0.53731250 0.46666480 0.44043540","0.61750240 0.53647780 0.46687310 0.43947310"]])

cLTD = None 

ltdStep = 0.0000005
ltpStep = 0.00002
(ltdMin,ltdMax) = (0.0000015,0.000005)
LTD = np.array([0.00003])# np.arange(0.000005-2*ltdStep,0.000005+4*ltdStep,ltdStep)
LTP = np.array([0.0002])#np.arange(0.00007,0.00007+4*ltpStep,ltpStep)

STDP_factor = 1

STDP = np.zeros((len(LTD)*len(LTP),2))
index = 0
for ltd in LTD:
    for ltp in LTP:
        STDP[index] = [ltd*STDP_factor,ltp*STDP_factor]
        index += 1

NOISE = np.array([1])

if logFile:
    additionalOptions += ' --debug'

folder = options.folder
if folder:
    additionalOptions += ' -F'+folder


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
        # output.append(p+' --flashPlots '+str(options.fP)+' --storePlots '+str(options.sP))
        output.append(p+' --storePlots '+str(options.sP))
    return output

def runProcess(process):
    if type(process) is not str: # is a list
        Parallel(n_jobs=len(process))(delayed(runProcess)(p) for p in process)
        return
    p = subprocess.Popen(process+additionalOptions,stderr=subprocess.STDOUT, stdout=subprocess.PIPE, shell=True)
    
    print "Running ",process+additionalOptions

    if logFile:
        fname = (options.logFolder+(process+additionalOptions).replace('/', '.')+".txt").replace(' ', '')
        with open(fname,'a') as logfile:
            for line in p.stdout:
                # sys.stdout.write(line)
                logfile.write(line)
    p.wait()
    print "Finished ",process+additionalOptions

def runModelLearning(i):
    forRange = TIMES if parallelDivisions else DIVISIONS
    for j in range(forRange):

        time = j if parallelDivisions else i
        if DA_DIV is not None:
            DA = DA_DIV[i] if parallelDivisions else DA_DIV[j]
        else:
            DA = 0.5*float(i) if parallelDivisions else 0.5*float(j)

        bProcess = 'python learning.py -d'+str(DA)+' -f'+str(time + baseNumber).zfill(3)
        fisrtDiv = (parallelDivisions and (i==0)) or (not parallelDivisions and (j==0))
        if firstDynamic and fisrtDiv:
            bProcess += ' --dynamicDA'

        p = addNoise(bProcess)
        p = addSTDP(p)

        if time == 0 and plotNeuralData:
        # if plotNeuralData:
            p = addPlotting(p)

        runProcess(p)

if parallelDivisions:
    Parallel(n_jobs=NJOBS)(delayed(runModelLearning)(i) for i in range(DIVISIONS))
else:
    Parallel(n_jobs=NJOBS)(delayed(runModelLearning)(i) for i in range(TIMES))
