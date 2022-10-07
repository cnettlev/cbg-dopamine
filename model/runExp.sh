#!/bin/bash

expName=directLoopWorkingZone/unforcedSel

HN=`hostname`

if [ "$HN" = "corcovado" ]; then
	preFolder=/mnt/Data/Cristobal/tonicDA_effects/dynamicDAStrCtx/
else
	preFolder=""
fi

folder=$preFolder`date +%Y%m%d`$expName
DIRECTORY=$folder
DIR_NUM=0

while [ -d "$DIRECTORY" ]; do
	DIRECTORY="$folder"_"$DIR_NUM"
	((DIR_NUM++))
done
folder=$DIRECTORY

echo
echo "Using $folder folder for storing experimental data"
echo

mkdir -p "$folder"

python2 parallelLearning.py -f $folder -t 25 -R -a " --tonicDA-deltaOverReward 0 --noisyWeights 0 " -p -d "0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0" 
# python parallelLearning.py -f $folder -t 10 -b 40 -R -a   " --tonicDA-deltaOverReward 0 -d 0.0" 
# python parallelLearning.py -f $folder -t 50 -R -a         " --tonicDA-deltaOverReward 0" -d "1.0 9.0 10.0"