from optparse import OptionParser
import numpy as np
import os
from socket import gethostname      # Added to specify different output folders
                                    # deppending on the computer where the code is
                                    # running

usage = "usage: %prog [option_1 [arg_option_1] option_2 [arg_option_2] ...]"
parser = OptionParser(usage=usage)
parser.add_option("-b", "--performanceBuffSize", dest="perfBuffSize", default=0, type="int",
                  help="Number of trials used for computing the resulting performance. Use PERF_BUFF equal to 0 for using all \
                  the past trials. Negative values are automatically set as positive. (Int)", metavar="PERF_BUFF")
parser.add_option("--correlatedNoise", dest="cNoise", action="store_true", default=False,
                  help="Enables the use of correlated noise in the dopamine affected striatal populations.", metavar="corrNoise")
parser.add_option("-d", "--da", dest="DA", default=4.0, type="float",
                  help="Tonic DA level. (Float)", metavar="DA")
parser.add_option("--dynamicDA", dest="dynamicDA", action="store_true", default=False, help="Enable dynamic tonic DA.")
parser.add_option("--debug", dest="debug", action="store_true", default=False,
                  help="Enables \"DEBUG\" flag (meaning that it prints what is going on trial by trial.", metavar="DEBUG")
parser.add_option("-D", "--disLearning", dest="disableLearning", action="store_true", default=False, help="Disable learning.")
parser.add_option("-n", "--ncues", dest="n", default=4, type="int",
                  help="Number of cues to be presented. (Int)", metavar="nCues")
parser.add_option("-f", "--file", dest="nFile", default="000",
                  help="Index number to be included in a storing data file. (String)")
parser.add_option("-F", "--folder", dest="folder", default='Exp', metavar='Path',
                  help="Folder destination for the storing data file. (String)")
parser.add_option("--flashPlots", dest="fPlots", default=-1, type="int", metavar="FPLOTS",
                  help="Activates plotting if not, plotting every FPLOTS trials. A zero value means online plotting every trial. Deactivated by negative values.")
parser.add_option("-g", "--garivier-molines-2008", dest="GM2008", action="store_true", default=False,
                  help="Set-up the task following Aurelien Garivier and Eric Moulines's article arXiv:0805.3415 [math.ST].", metavar="ML")
parser.add_option("-w", "--initWmean", dest="initialWeightsMean", type="float", default=0.5, metavar='wMean',
                  help="Initial mean value for the corticostriatal weights. (Float)")
parser.add_option("-i", "--cogInitW", dest="cogInitialWeights", type="float", nargs=4, default=None, metavar='cw0 cw1 cw2 ... cwn',
                  help="Initial cognitive corticostriatal weights. (Float array separated by spaces)")
parser.add_option("-I", "--motInitW", dest="motInitialWeights", type="float", nargs=4, default=None, metavar='mw0 mw1 mw2 ... mwn',
                  help="Initial motor corticostriatal weights.(Float array separated by spaces)")
parser.add_option("-L", "--ltp", dest="LTP", default=0.002, type="float",
                  help="Value of the long term potentiation applied in the learning of corticostriatal connections. (Float)", metavar="LTP")
parser.add_option("-l", "--ltd", dest="LTD", default=0.001, type="float",
                  help="Value of the long term depression applied in the learning of corticostriatal connections. (Float)", metavar="LTD")
parser.add_option("--ltd-constant", dest="cLTD", action="store_true", default=False,
                  help="Enables a continuous long term depression applied in the corticostriatal connections, based on the activation of endocannabinoid receptors CB1 (generated by mGluR and D2R activation).", metavar="cLTD")
parser.add_option("--ltd-constant-onestep", dest="oScLTD", action="store_true", default=False,
                  help="Enables an artificially introduced long term depression applied in the corticostriatal connections, once by trial.", metavar="oScLTD")
parser.add_option("-M", "--mLearning", dest="applyInMotorLoop", action="store_true", default=False,
                  help="Enable learning in the motor loop.", metavar="ML")
parser.add_option("-N", "--noiseLevel", dest="noise", default=.5, type="float",
                  help="Noise level in the substantia nigra. (float)", metavar="sncNoise")
parser.add_option("-p", "--regPattern", dest="useRegularPattern", action="store_true", default=False,
                  help="Sets the presentation of specific pair of cues in order to produce a controlled number of presentations for each combination.")
parser.add_option("-P", "--activityPlots", dest="plotting", action="store_true", default=False,
                  help="Activates the presentation of plots with neural activity at every trial.")
parser.add_option("--pausePlots", dest="pPlots", action="store_true", default=False,
                  help="If plotting, introduce a pause (wait for a button).")
parser.add_option("--storePlots", dest="savePlots", default=-1, type="int", metavar="NPLOTS",
                  help="If plots are activated, stores the plots as pdf figures every NPLOTS trials.")
parser.add_option("-r", "--reverseAt", dest="reverseTrial", default=0, type="int", metavar="RTRIAL", help="Enables a flip of the vector \
                  that contains the reward probabilities at trial number RTRIAL of the experiment. 0 means no flip.")
parser.add_option("--relativeValue", dest="rValue", action="store_true", default=False,
                  help="Computes prediction error with respect to a relative expected value defines as the mean between expected values of perceived cues.")
parser.add_option("-R", "--expected-reward", dest="eReward", action="store_true", default=False,
                  help="Enables the generation of a DA burst based on the expected reward at the moment of the cognitive decisions.")
parser.add_option("-s", "--seed", dest="seed", default=0, type="int", metavar="S", help="Number use for initiating the random generator (0 for no initiation).")
parser.add_option("-S", "--storeData", dest="storeData", action="store_true", default=False,
                  help="Enables the storing of generated data.")
parser.add_option("-t", "--trials", dest="nTrials", default=120, type="int",
                  help="Number of trials. (Int)", metavar="nTrials")
parser.add_option("--tonicDA-timeConstant", dest="tau_tonicDA", type="float", default=0.001, metavar="tDA_tau", help="Set the time constant of the tonic DA filter when dynamicDA is used.")
parser.add_option("-z", "--zeroValues", dest="zValues", action="store_true", default=False,
                  help="Set initial cues values as zero.")
parser.add_option("-X","--parameterX", dest="parX", default=1, type="float",
                  help="An auxiliar parameter for fast and short manipulations of the model. (float)", metavar="X")
parser.add_option("-Y","--parameterY", dest="parY", default=1, type="float",
                  help="Another auxiliar parameter. (float)", metavar="Y")

(options, args) = parser.parse_args()

def createUnexistentFolder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

perfBuff = np.abs(options.perfBuffSize)
DA = options.DA
dynamicDA = options.dynamicDA
tauTonicDA = options.tau_tonicDA
n = options.n
nbTrials = options.nTrials
nFile = options.nFile
folder = options.folder
if folder:
    createUnexistentFolder(folder)
    if folder[-1] is not '/':
        folder = folder+'/'
flashPlots = options.fPlots
Wmean = options.initialWeightsMean
if options.cogInitialWeights != None:
	cogInitialWeights = np.diag(np.array(options.cogInitialWeights));
else:
	cogInitialWeights = None
if options.motInitialWeights != None:
	motInitialWeights = np.diag(np.array(options.motInitialWeights));
else:
	motInitialWeights = None
alpha_LTP = options.LTP
alpha_LTD = options.LTD
constantLTD = options.cLTD
oneStepConstantLTD = options.oScLTD
applyInMotorLoop = options.applyInMotorLoop
regPattern = options.useRegularPattern
neuralPlot = options.plotting or (flashPlots>=0)
if neuralPlot and flashPlots == -1:
      flashPlots = 0
pausePlots = options.pPlots
storePlotsEvery = options.savePlots
if storePlotsEvery > 0:
      if flashPlots > 0 and (storePlotsEvery%flashPlots)!=0:
            print "Error: Trying to store plots withouth being draw",storePlotsEvery%flashPlots
            print "Draw every",flashPlots
            print "Store every",storePlotsEvery
            exit()
      plotsFolder = folder+'neuralPlots/'
      createUnexistentFolder(plotsFolder)
learn = not options.disableLearning
SNc_N = options.noise
invertAt = [options.reverseTrial]
relativeValue = options.rValue
cogReward = options.eReward
randomInit = options.seed
STORE_DATA = options.storeData
useCorrelatedNoise = options.cNoise
zeroValues = options.zValues
doPrint = options.debug
aux_X = options.parX
aux_Y = options.parY

if options.GM2008:
  n = 3
  nbTrials = 10000 #9000
  invertAt = [3000,5000] #[3000,5000]


file_base =  'gDA_'+str(SNc_N)
if folder:
    file_base = folder+file_base

STORE_FAILED = False
FAILED_FILE = folder+'failed_'+str(DA)

if STORE_DATA or STORE_FAILED:
    
    if 'corcovado' == gethostname():
        if not folder:
            filename = '/mnt/Data/Cristobal/tonicDA_effects/connections/Exp/'+file_base+'__'+datetime.now().strftime("%Y%m%d")+'_'+nFile
        else:
            filename = file_base+str(DA)+'_'+str(alpha_LTP)+'_'+str(alpha_LTD)+'_'+nFile
    else:
        if not folder:
            filename = './Exp/'+file_base+str(DA)+'__'+datetime.now().strftime("%Y%m%d")+'_'+nFile
        else:
            filename = file_base+'_'+str(alpha_LTP)+'_'+str(alpha_LTD)+'_'+str(DA)+'_'+nFile