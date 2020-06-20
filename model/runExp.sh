#!/bin/bash

expName=_ldtNoise

HN=`hostname`

if [ "$HN" = "corcovado" ]; then
	preFolder=/mnt/Data/Cristobal/tonicDA_effects/dynamicDA/
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
# mkdir -p "$folder"_zeroValue
# mkdir -p "$folder"_zeroRelativeValue
#tonicDA_tau=( "0.0005" "0.001" "0.005" "0.01" )

# for tau in "${tonicDA_tau[@]}"; do
# 
# 	python parallel-model-learning.py $model $folder $tau
# 	sleep 1
# done
# 
# python parallel-model-learning.py $model $folder
# python parallel-model-learning.py $model $folder 100# 

# python parallel-model-learning.py $folder 100
python parallel-model-learning.py "$folder" 1
python parallel-model-learning.py "$folder" 2
python parallel-model-learning.py "$folder" 5
python parallel-model-learning.py "$folder" 10
python parallel-model-learning.py "$folder" 20
# python parallel-model-learning.py "$folder"_sweep 3
# python parallel-model-learning.py "$folder"_zeroValue 100 "--zeroValues"
# python parallel-model-learning.py "$folder"_zeroRelativeValue 100 "--zeroValues --relative-value"
#python parallel-model-learning.py $folder 100

# python ../learning.py -t 200 -r 100 -L 0.000075 -l 0.0000025 --dynamicDA -b 12 -R --ltd-constant  --correlatedNoise -S -f 000 &
# python ../learning.py -t 200 -r 100 -L 0.000075 -l 0.0000025 --dynamicDA -b 12 -R --ltd-constant  --correlatedNoise -S -f 001 &
# python ../learning.py -t 200 -r 100 -L 0.000075 -l 0.0000025 --dynamicDA -b 12 -R --ltd-constant  --correlatedNoise -S -f 002 &
# python ../learning.py -t 200 -r 100 -L 0.000075 -l 0.0000025 --dynamicDA -b 12 -R --ltd-constant  --correlatedNoise -S -f 003 &
# python ../learning.py -t 200 -r 100 -L 0.000075 -l 0.0000025 --dynamicDA -b 12 -R --ltd-constant  --correlatedNoise -S -f 004 &
# python ../learning.py -t 200 -r 100 -L 0.000075 -l 0.0000025 --dynamicDA -b 12 -R --ltd-constant  --correlatedNoise -S -f 005 &
# python ../learning.py -t 200 -r 100 -L 0.000075 -l 0.0000025 --dynamicDA -b 12 -R --ltd-constant  --correlatedNoise -S -f 006 &
# python ../learning.py -t 200 -r 100 -L 0.000075 -l 0.0000025 --dynamicDA -b 12 -R --ltd-constant  --correlatedNoise -S -f 007 &
# python ../learning.py -t 200 -r 100 -L 0.000075 -l 0.0000025 --dynamicDA -b 12 -R --ltd-constant  --correlatedNoise -S -f 008 &
# python ../learning.py -t 200 -r 100 -L 0.000075 -l 0.0000025 --dynamicDA -b 12 -R --ltd-constant  --correlatedNoise -S -f 009 &