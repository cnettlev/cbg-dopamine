

basicOptions='-l 0.0000065 -L 0.00009 -t 200 -r 100 --ltd-constant --correlatedNoise --noisyWeights 0.01 --relativeValue '

flashplots=' --flashPlots 1 ' # --storePlots 10'

realTimePlot=' -P '

DA4=' -d 4.0 '
DA6=' -d 6.0 '
DA8=' -d 8.0 '

#python learning.py $basicOptions $flashplots $@
python learning.py $basicOptions $realTimePlot $@

