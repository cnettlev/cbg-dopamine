#!/bin/bash

expName=_weightsAndThreshold_Y1

HN=`hostname`

if [ "$HN" = "corcovado" ]; then
	preFolder=/mnt/Data/Cristobal/tonicDA_effects/dynamicDA/
else
	preFolder=$HOME/cbg-da/
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

tab=" --tab"
options=()


cmds[1]="python parallel-model-learning.py "$folder" \" -X 0.46 -Y 0.5 \""

cmds[2]="python parallel-model-learning.py "$folder" \" -X 0.48 -Y 0.5 \""

cmds[3]="python parallel-model-learning.py "$folder" \" -X 0.52 -Y 0.5 \""

cmds[4]="python parallel-model-learning.py "$folder" \" -X 0.54 -Y 0.5 \""

cmds[5]="python parallel-model-learning.py "$folder" \" -Y 0.46 -X 0.5 \""

cmds[6]="python parallel-model-learning.py "$folder" \" -Y 0.48 -X 0.5 \""

cmds[7]="python parallel-model-learning.py "$folder" \" -Y 0.52 -X 0.5 \""

cmds[8]="python parallel-model-learning.py "$folder" \" -Y 0.54 -X 0.5 \""

for cmd in "${cmds[@]}"; do
options+=($tab --command="bash -c '$cmd ; bash'" )
done

gnome-terminal "${options[@]}"

exit 0
