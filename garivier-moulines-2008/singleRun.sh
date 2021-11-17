

basicOptions='--debug -l 0.00003 -L 0.0002 -t 200 -r 100 --ltd-constant --relativeValue -S'

flashplots=' --flashPlots 1 ' # --storePlots 10'

realTimePlot=' -P '

DA4=' -d 4.0 '
DA6=' -d 6.0 '
DA8=' -d 8.0 '

python learningDA.py $basicOptions $flashplots $@
#python learning.py $basicOptions $@ # -l 0.0000065

