#!/bin/bash
## dos2unix file
### tell SGE to use bash for this script
#$ -S /bin/bash
### execute the job from the current working directory, i.e. the directory in which the qsub command is given
#$ -cwd
### join both stdout and stderr into the same file
#$ -j y
### set email address for sending job status
#$ -M tnh48@drexel.edu
### project - basically, your research group name with "Grp" replaced by "Prj"
#$ -P rosenclassPrj

#$ -pe openmpi_ib 64
### request 15 min of wall clock time "h_rt" = "hard real time" (format is HH:MM:SS, or integer seconds)
#$ -l h_rt=20:00:00
### a hard limit 8 GB of memory per slot - if the job grows beyond this, the job is killed
#$ -l h_vmem=64G
### want nodes with at least 6 GB of free memory per slot
#$ -l m_mem_free=16G
### select the queue all.q
#$ -q all.q

. /etc/profile.d/modules.sh

### These four modules must ALWAYS be loaded
module load shared
module load proteus
module load sge/univa
module load gcc


### Whatever modules you used, in addition to the 4 above, 
### when compiling your code (e.g. proteus-openmpi/gcc)
### must be loaded to run your code.
### Add them below this line.
module load python/anaconda3

jupyter nbconvert --execute "test_nbconvert.ipynb"
jupyter nbconvert --execute --to notebook "test_nbconvert.ipynb"
