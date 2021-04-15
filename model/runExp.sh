#!/bin/bash

expName=dinamycDA_noises

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


cmds[1]="python parallel-learning.py -l -f $folder -t 50 -b 50 -n 5 -a \" -d 4.0 --dynamicDA\""

cmds[2]="python parallel-learning.py -l -f $folder -t 50 -b 50 -n 5 -a \" -d 4.0 \""

cmds[3]="python parallel-learning.py -l -f $folder -t 50 -b 50 -n 5 -a \" -d 6.0 \""

cmds[4]="python parallel-learning.py -l -f $folder -t 50 -b 50 -n 5 -a \" -d 8.0 \""


for cmd in "${cmds[@]}"; do
options+=($tab --title="$cmd" --command="bash -c '$cmd ; bash'" )
done

if [ "$HN" = "dell" ]; then
	xfce4-terminal "${options[@]}; --window --fullscreen"
else
	gnome-terminal "${options[@]}"
fi

cd $folder

exit 0
