

python learning.py -d4.0 -f001 -N 0.01 -l6.5e-06 -L9e-05 -g -t 200 -b 12 --ltd-constant --correlatedNoise -S --relativeValue -F/mnt/Data/Cristobal/tonicDA_effects/dynamicDA/20200412 --debug --flashPlots 1 & #--storePlots 1 &
#python learning.py -d5.0 -f001 -N 0.01 -l6.5e-06 -L9e-05 -g -t 200 -b 12 --ltd-constant --correlatedNoise -S --relativeValue -F/mnt/Data/Cristobal/tonicDA_effects/dynamicDA/20200412 --debug -P &
python learning.py -d6.0 -f001 -N 0.01 -l6.5e-06 -L9e-05 -g -t 200 -b 12 --ltd-constant --correlatedNoise -S --relativeValue -F/mnt/Data/Cristobal/tonicDA_effects/dynamicDA/20200412 --debug --flashPlots 1 & #--storePlots 1 &

python ../model/learning.py -d4.0 -f001 -N 0.01 -l6.5e-06 -L9e-05 -t 90 -b 12 --ltd-constant --correlatedNoise -S --relativeValue -F/mnt/Data/Cristobal/tonicDA_effects/dynamicDA/20200412 --debug --flashPlots 1 & #--storePlots 1 &
#python ../model/learning.py -d5.0 -f001 -N 0.01 -l6.5e-06 -L9e-05 -g -t 200 -b 12 --ltd-constant --correlatedNoise -S --relativeValue -F/mnt/Data/Cristobal/tonicDA_effects/dynamicDA/20200412 --debug -P &
python ../model/learning.py -d6.0 -f001 -N 0.01 -l6.5e-06 -L9e-05 -t 90 -b 12 --ltd-constant --correlatedNoise -S --relativeValue -F/mnt/Data/Cristobal/tonicDA_effects/dynamicDA/20200412 --debug --flashPlots 1 & #--storePlots 1 &

# python learning.py -d8.0 -f001 -N 0.01 -l6.5e-06 -L9e-05 -g -t 200 -b 12 --ltd-constant --correlatedNoise -S --relativeValue -F/mnt/Data/Cristobal/tonicDA_effects/dynamicDA/20200412 --debug -P &