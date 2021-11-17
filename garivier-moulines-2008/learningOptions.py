from optparse import OptionParser
import numpy as np
import os
import errno

usage = "usage: %prog [option_1 [arg_option_1] option_2 [arg_option_2] ...]"
parser = OptionParser(usage=usage)
parser.add_option("-b", "--performanceBuffSize", dest="perfBuffSize", default=12, type="int",
                  help="Number of trials used for computing the resulting performance. Use PERF_BUFF equal to 0 for using all \
                  the past trials. Negative values are automatically set as positive. (Int)", metavar="PERF_BUFF")
parser.add_option("-d", "--da", dest="DA", default=4.0, type="float",
                  help="Tonic DA level. (Float)", metavar="DA")
parser.add_option("-D", "--disLearning", dest="disableLearning", action="store_true", default=False, help="Disable learning.")
parser.add_option("--DA-neurons", dest="DA_neurons", default=1, type="int", help="Defines the number of midbrain dopaminergic neurons (from SNc).")
parser.add_option("--dynamicDA", dest="dynamicDA", action="store_true", default=False, help="Enable dynamic tonic DA.")
parser.add_option("--debug", dest="debug", action="store_true", default=False,
                  help="Enables \"DEBUG\" flag (meaning that it prints what is going on trial by trial.", metavar="DEBUG")
parser.add_option("-f", "--file", dest="nFile", default="",
                  help="Index number to be included in a storing data file. (String)")
parser.add_option("-F", "--folder", dest="folder", default='', metavar='Path',
                  help="Folder destination for the storing data file. (String)")
parser.add_option("--flashPlots", dest="fPlots", default=-1, type="int", metavar="FPLOTS",
                  help="Activates plotting if not, plotting every FPLOTS trials. A zero value means online plotting every trial. Deactivated by negative values.")
parser.add_option("-G","--garivierMoulines", dest="garivierMoulines", action="store_true", default=False,
                  help="Sets task according to Gavirier and Molines 2008.", metavar="GM")
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
parser.add_option("-N", "--noiseLevel", dest="noise", default=1.0, type="float",
                  help="Noise factor. (float)", metavar="NOISE")
parser.add_option("-n", "--ncues", dest="n", default=4, type="int",
                  help="Number of cues to be presented. (Int)", metavar="nCues")
parser.add_option("--noisyWeights", dest="wnoise", default=0.05, type="float",
                  help="If positive, enables gaussian noise injection at the start of a trial at the corticostriatal cognitive weights, with zero mean and a standard deviation of (weightValue-wMin)*wNoise. (float)", metavar="wNoise")
parser.add_option("-p", "--regPattern", dest="useRegularPattern", action="store_true", default=False,
                  help="Sets the presentation of specific pair of cues in order to produce a controlled number of presentations for each combination.")
parser.add_option("-P", "--activityPlots", dest="plotting", action="store_true", default=False,
                  help="Activates the presentation of plots with neural activity at every trial.")
parser.add_option("--pAdvantage", dest="dynamicDAoverA", action="store_true", default=False,
                  help="If dynamic DA is active, SNc activity variates following a perceived advantage.", metavar="PADV")
parser.add_option("--pasquereau", dest="pasquereau", action="store_true", default=False,
                  help="Sets task according to Pasquereau 2007.", metavar="PASQ")
parser.add_option("--pausePlots", dest="pPlots", action="store_true", default=False,
                  help="If plotting, introduce a pause (wait for a button).")
parser.add_option("-r", "--reverseAt", dest="reverseTrial", default=60, type="int", metavar="RTRIAL", help="Enables a flip of the vector \
                  that contains the reward probabilities at trial number RTRIAL of the experiment. 0 means no flip.")
parser.add_option("--relativeValue", dest="rValue", action="store_true", default=False,
                  help="Computes prediction error with respect to a relative expected value defines as the mean between expected values of perceived cues.")
parser.add_option("-R", "--expected-reward", dest="eReward", action="store_true", default=False,
                  help="Enables the generation of a DA burst based on the expected reward at the moment of the cognitive decisions.")
parser.add_option("-s", "--seed", dest="seed", default=0, type="int", metavar="S", help="Number use for initiating the random generator (0 for no initiation).")
parser.add_option("-S", "--storeData", dest="storeData", action="store_true", default=False,
                  help="Enables the storing of generated data.")
parser.add_option("--smoothAdv", dest="minSmoothA", default=0, type="float", metavar="mADV", help="If different than zero (it's default value) \
                  and dynamicDA flag is activated, creates a minimal hard threshold (for positive values) or enables the use of a sigmoid transfer function shaping advantage (using --pAdvantage) or reward perception.")
parser.add_option("--staticThreshold", dest="sThreshold", action="store_true", default=False,
                  help="Disables DA dependence at striatal threshold.")
parser.add_option("--staticCtxStr", dest="sCtxStr", action="store_true", default=False,
                  help="Disables DA dependence at corticostriatal weights.")
parser.add_option("--storePlots", dest="savePlots", default=-1, type="int", metavar="NPLOTS",
                  help="If plots are activated, stores the plots as pdf figures every NPLOTS trials.")
parser.add_option("-t", "--trials", dest="nTrials", default=120, type="int",
                  help="Number of trials. (Int)", metavar="nTrials")
parser.add_option("--tonicDA-timeConstant", dest="tau_tonicDA", type="float", default=0.001, metavar="tDA_tau", help="Set the time constant of the tonic DA filter when dynamicDA is used.")
parser.add_option("--tonicDA-increase", dest="gamma_DA", type="float", default=6, metavar="gamma_DA", help="Set the top tonic activity, reached on a repeteadly succesfull experence.")
parser.add_option("--uncorrelatedNoise", dest="ucNoise", action="store_true", default=False,
                  help="Disables the use of correlated noise in the dopamine affected striatal populations.", metavar="ucorrNoise")
parser.add_option("-w", "--initWmean", dest="initialWeightsMean", type="float", default=0.5, metavar='wMean',
                  help="Initial mean value for the corticostriatal weights. (Float)")
parser.add_option("-X","--parameterX", dest="parX", default=1, type="float",
                  help="An auxiliar parameter for fast and short manipulations of the model. (float)", metavar="X")
parser.add_option("-Y","--parameterY", dest="parY", default=1, type="float",
                  help="Another auxiliar parameter. (float)", metavar="Y")
parser.add_option("-z", "--zeroValues", dest="zValues", action="store_true", default=False,
                  help="Set initial cues values as zero.")

(options, args) = parser.parse_args()

def createUnexistentFolder(folder):
    if not os.path.exists(folder):
        try:
            os.makedirs(folder)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise  # raises the error again        

perfBuff = np.abs(options.perfBuffSize)
DA = options.DA
DA_neurons = options.DA_neurons
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
storePlotsEvery = options.savePlots
neuralPlot = options.plotting or (flashPlots>=0) or (storePlotsEvery>=0)
# if neuralPlot and flashPlots == -1:
#       flashPlots = 0
pausePlots = options.pPlots
if storePlotsEvery > 0:
      # if flashPlots > 0 and (storePlotsEvery%flashPlots)!=0:
      #       print "Error: Trying to store plots withouth being draw",storePlotsEvery%flashPlots
      #       print "Draw every",flashPlots
      #       print "Store every",storePlotsEvery
      #       exit()
      plotsFolder = folder+'neuralPlots/'
      createUnexistentFolder(plotsFolder)
learn = not options.disableLearning
N_factor = options.noise
invertAt = [options.reverseTrial]
relativeValue = options.rValue
cogReward = options.eReward
randomInit = options.seed
STORE_DATA = options.storeData
if STORE_DATA:
    dataFolder = folder+'data/'
    createUnexistentFolder(dataFolder)
staticThreshold = options.sThreshold
staticCtxStr = options.sCtxStr
useCorrelatedNoise = not options.ucNoise
zeroValues = options.zValues
doPrint = options.debug
aux_X = options.parX
aux_Y = options.parY

Weights_N = options.wnoise
garivierMoulines = options.garivierMoulines
pasquereau = options.pasquereau

dynamicDAoverA = options.dynamicDAoverA
gamma_DAbySuccess = options.gamma_DA
minSmoothA = options.minSmoothA