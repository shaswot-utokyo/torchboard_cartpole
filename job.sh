#!/bin/bash
#PBS -q h-regular
#PBS -l select=1:mpiprocs=1:ompthreads=1
#PBS -W group_list=gk37
#PBS -l walltime=00:15:00
cd $PBS_O_WORKDIR
./etc/profile.d/modules.sh

module load anaconda3/4.3.0 cuda9/9.0.176
export PYTHONUSERBASE=/lustre/gk37/k37004/pytorch
export PATH=$PYTHONUSERBASE/bin:$PATH

python ./dqn-cartpole.py 123 base_config
