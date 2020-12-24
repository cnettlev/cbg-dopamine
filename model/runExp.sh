#!/bin/bash

expName=_weightsAndThreshold_Y1

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

tab=" --tab"
options=()

# X -> CTX to STR
# Y -> STR threshold

cmds[1]="python parallel-model-learning.py "$folder" \" -X 0.46 -Y 0.5 \""

cmds[2]="python parallel-model-learning.py "$folder" \" -X 0.48 -Y 0.5 \""

cmds[3]="python parallel-model-learning.py "$folder" \" -X 0.52 -Y 0.5 \""

cmds[4]="python parallel-model-learning.py "$folder" \" -X 0.54 -Y 0.5 \""

cmds[5]="python parallel-model-learning.py "$folder" \" -Y 0.46 -X 0.5 \""

cmds[6]="python parallel-model-learning.py "$folder" \" -Y 0.48 -X 0.5 \""

cmds[7]="python parallel-model-learning.py "$folder" \" -Y 0.52 -X 0.5 \""

cmds[8]="python parallel-model-learning.py "$folder" \" -Y 0.54 -X 0.5 \""

# cmds[2]="python parallel-model-learning-3-plots.py "$folder" " # \" -X 0.4 -Y 1.25 --staticThreshold \""

# cmds[3]="python parallel-model-learning-3-plots.py "$folder" " # \" -X 0.6 -Y 1.25 --staticThreshold \""

# cmds[4]="python parallel-model-learning-3-plots.py "$folder" " # \" -X 0.8 -Y 1.25 --staticThreshold \""

# cmds[5]="python parallel-model-learning-3-plots.py "$folder" " # \" -X 1.2 -Y 1.25 --staticThreshold \""

# cmds[6]="python parallel-model-learning-3-plots.py "$folder" " # \" -X 1.4 -Y 1.25 --staticThreshold \""

# cmds[7]="python parallel-model-learning-3-plots.py "$folder" " # \" -X 1.6 -Y 1.25 --staticThreshold \""

# cmds[8]="python parallel-model-learning-3-plots.py "$folder" " # \" -X 1.8 -Y 1.25 --staticThreshold \""

# cmds[9]="python parallel-model-learning-3-plots.py "$folder" " # \" -Y 0.2 -X 1.25 --staticCtxStr \""

# cmds[10]="python parallel-model-learning-3-plots.py "$folder" " # \" -Y 0.4 -X 1.25 --staticCtxStr \""

# cmds[11]="python parallel-model-learning-3-plots.py "$folder" " # \" -Y 0.6 -X 1.25 --staticCtxStr \""

# cmds[12]="python parallel-model-learning-3-plots.py "$folder" " # \" -Y 0.8 -X 1.25 --staticCtxStr \""

# cmds[13]="python parallel-model-learning-3-plots.py "$folder" " # \" -Y 1.2 -X 1.25 --staticCtxStr \""

# cmds[14]="python parallel-model-learning-3-plots.py "$folder" " # \" -Y 1.4 -X 1.25 --staticCtxStr \""

# cmds[15]="python parallel-model-learning-3-plots.py "$folder" " # \" -Y 1.6 -X 1.25 --staticCtxStr \""

# cmds[16]="python parallel-model-learning-3-plots.py "$folder" " # \" -Y 1.8 -X 1.25 --staticCtxStr \""

for cmd in "${cmds[@]}"; do
options+=($tab --command="bash -c '$cmd ; bash'" )
done

gnome-terminal "${options[@]}"

exit 0

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