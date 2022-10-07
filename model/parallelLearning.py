#!/usr/bin/env python2
from joblib import Parallel, delayed
import multiprocessing
import subprocess
import sys
import numpy as np
from time import sleep
from parallelOptions import options

additionalOptions = ' --ltd-constant --relativeValue -S --correlatedNoise --tonicDA-dynamic ' + options.aP

parallelDivisions = options.pD

logFile = options.lF

plotNeuralData = options.pN

firstDynamic = options.fD
DA_DIV = np.array(options.daDiv.split(' '))
DIVISIONS = DA_DIV.shape[0]
TIMES = options.t
baseNumber = options.bN
NJOBS = options.nJ
WEIGHTS = np.array([["0.60 0.566 0.533 0.50"]]) # ["0.54 0.525 0.49 0.475"],["0.60 0.566 0.533 0.50"], 

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

NOISE = np.array([7])

if logFile:
    additionalOptions += ' --debug'
    if options.folder:
        logNameRemove = ' -F'+options.folder


if options.R: # reproducibility
    additionalOptions += ' --seed-from-nfile'

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
        output.append(p+' --flashPlots '+str(options.fP)+' --storePlots '+str(options.sP))
        # output.append(p+' --storePlots '+str(options.sP))
    return output

def addWeights(processes):
    processes = createList(processes)
    output = []
    for w in WEIGHTS:
        weights = ' '.join(w)
        for p in processes:
            output.append(p+' -i '+weights)
    return output

def runProcess(process, time=''):
    if type(process) is not str: # is a list
        nProc = len(process)
        Parallel(n_jobs=nProc)(delayed(runProcess)(p,time+'.'+str(ip)) for ip, p in enumerate(process))
        return

    p = subprocess.Popen(process+additionalOptions,stderr=subprocess.STDOUT, stdout=subprocess.PIPE, shell=True)
    
    if time>=0:
        print "Running",time,process+additionalOptions
    else:
        print "Running",process+additionalOptions

    if logFile:
        fname = options.logFolder+process.replace('python2 learning.py -d','')+additionalOptions
        fname = fname.replace(logNameRemove, '').replace(' ', '')+".txt"
        with open(fname,'a') as logfile:
            for line in p.stdout:
                logfile.write(line)
    p.wait()
    if time>=0:
        print "Finished ("+str(time)+")"
    else:
        print "Finished ",process+additionalOptions

def runModelLearning(i):
    forRange = TIMES if parallelDivisions else DIVISIONS
    for j in range(forRange):

        time = j if parallelDivisions else i
        divIndex = i if parallelDivisions else j
        if DA_DIV is not None:
            DA = DA_DIV[divIndex]
        else:
            DA = 0.5*float(divIndex)

        bProcess = 'python2 learning.py -d'+str(DA)+' -f'+str(time + baseNumber).zfill(3)
        # fisrtDiv = (parallelDivisions and (i==0)) or (not parallelDivisions and (j==0))
        if firstDynamic and divIndex == 0:
            bProcess += ' --dynamicDA'

        # p = addNoise(bProcess)
        p = addSTDP(bProcess)

        if time == 0 and plotNeuralData:
        # if plotNeuralData:
            p = addPlotting(p)

        if options.sW:
            p = addWeights(p)

        runProcess(p, str(time)+'.'+str(divIndex))

if parallelDivisions:
    Parallel(n_jobs=NJOBS)(delayed(runModelLearning)(i) for i in range(DIVISIONS))
else:
    Parallel(n_jobs=NJOBS)(delayed(runModelLearning)(i) for i in range(TIMES))
