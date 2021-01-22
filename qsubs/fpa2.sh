#$ -l tmem=8G
#$ -l gpu=true
#$ -S /bin/bash
#$ -j y
#$ -wd /SAN/medic/PerceptronHead/codes/bmvc/exps

~/anaconda3/bin/python fpa2.py