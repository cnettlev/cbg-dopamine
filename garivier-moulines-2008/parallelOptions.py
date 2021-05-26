from optparse import OptionParser
import numpy as np
import os
import errno

usage = "usage: %prog [option_1 [arg_option_1] option_2 [arg_option_2] ...]"
parser = OptionParser(usage=usage)


parser.add_option("-a", "--additionalOptions", dest="aP", default='',
                  help="Additional options as e.g.: -a \" -X 3.0 -d 4.6 \".")
parser.add_option("-b", "--baseNumber", dest="bN", default=0, type="int",
                  help="Counting starts at this number.")
parser.add_option("-f", "--folder", dest="folder", default='',
                  help="Self-explained.")
parser.add_option("-F", "--firstDynamic", dest="fD", action="store_true", default=False,
                  help="First set includes dynamic dopamine.")
parser.add_option("--flashPlots", dest="fP", default=50, type="int",
                  help="Plot every fP.")
parser.add_option("-l","--logFile", dest="lF", action="store_true", default=False,
                  help="Enables logfiles")
parser.add_option("-n", "--nJobs", dest="nJ", type="int", default=12,
                  help="Number of parallel processes.")
parser.add_option("-p", "--parallelDivisions", dest="pD", action="store_true", default=False,
                  help="Parallelize by dopamine activity.")
parser.add_option("-P", "--plotNeuralData", dest="pN", action="store_true", default=False,
                  help="Self-explained.")
parser.add_option("--storePlots", dest="sP", default=50, type="int",
                  help="Store plots every fP.")
parser.add_option("-t", "--times", type="int", dest="t", default=1,
                  help="Self-explained.")

(options, args) = parser.parse_args()

def createUnexistentFolder(folder):
    if not os.path.exists(folder):
        try:
            os.makedirs(folder)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise  # raises the error again    


if options.folder:
    if options.folder[-1] is not '/':
        options.folder = options.folder+'/'

if options.lF:
  options.logFolder = options.folder + 'logs/'
  createUnexistentFolder(options.logFolder)


