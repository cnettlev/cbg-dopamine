#!/bin/bash

expName=directLoopWorkingZone/tonicDA

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

#mkdir -p $folder
mkdir -p "$folder"

# tonicDA=( "2.0" "6.0" "8.0" "10.0" "12.0" )
# for da in "${tonicDA[@]}"; do
# 
# 	python parallel-model-learning.py "$folder" -t 20 --flashPlots 10 --storePlots 10
# 	sleep 1
# done

python parallelLearning.py -f $folder -t 20 -P --flashPlots 10 --storePlots 10

# LTDLTP=( "2.0" "6.0" "8.0" "10.0" "12.0" )
# for da in "${tonicDA[@]}"; do
# 
# 	python parallel-model-learning.py "$folder" $da
# 	sleep 1
# done
# 
# 
# 
# tonicDA=( "2.0" "6.0" "8.0" "10.0" "12.0" )
# for da in "${tonicDA[@]}"; do
# 
# 	python parallel-model-learning.py "$folder" $da
# 	sleep 1
# done
# 
