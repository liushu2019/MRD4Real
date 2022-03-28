Run mrd.py in current directory like this:
python mrd.py wikipedia0 1 0 1 2 bpmll d2 org multi_disc half GW

parameters:
1st: project name (folder with source files needed in ./workspace/, see the example folder "wikipedia0")
2nd: version number (any)
3th: the number of source network label info file
4th: the number of target network label info file (if any)
5th: the number of validation network label info file
6th: name of loss function (bpmll or bce)
7th: discriminator variation (d2:BPMLL_SD2 or BPMLL_DD; d3:BPMLL_SD3)
8th: data sampling (org recommended)
9th: single or double discriminators (multi_disc:BPMLL_DD, other wise any other str)
10th: discriminator's loss function details (half recommended)
11th: embedding method (S2V: struc2vec, R2V: role2vec, GW: GraphWave)

required libs:
numpy
pandas
torch
matplotlib
sklearn


By Shu Liu, 2022/03/28
