#!/bin/bash

expName=noise_eval

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

fName=`basename $0`
cp $fName $folder/$fName

tab=" --tab"
options=()


# cmds[1]="python parallel-learning.py -l -f $folder -t 100 -n 20 -P --storePlots 5 -a \" -d 4.0 --dynamicDA  --pAdvantage  \""
cmds[1]="python parallel-learning.py -l -f $folder -t 20 -n 20 -F " # -a \" -d 4.0 --dynamicDA  -N 4 \""
# cmds[2]="python parallel-learning.py -l -f $folder -t 100 -n 12 -P --storePlots 1000 -a \" -d 4.0 --garivierMoulines --pAdvantage  \""




for cmd in "${cmds[@]}"; do
options+=($tab --title="$cmd" --command="bash -c '$cmd ; bash'" )
done

if [ "$HN" = "dell" ]; then
	xfce4-terminal "${options[@]}; --window --fullscreen"
else
	gnome-terminal "${options[@]}"
fi

# exit 0
