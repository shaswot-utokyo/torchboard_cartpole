#!/bin/bash
#PBS -q h-regular
#PBS -l select=1:mpiprocs=1:ompthreads=16
#PBS -W group_list=gk37
#PBS -l walltime=00:15:00
cd $PBS_O_WORKDIR
./etc/profile.d/modules.sh

module load anaconda3/2019.03 cuda10/10.0.130 openmpi/gdr/2.1.2/pgi-cuda10 pgi/19.3

export PYTHONUSERBASE=/lustre/gk37/k37004/dharti
export PATH=$PYTHONUSERBASE/bin:$PATH


NUM_OF_SEEDS=10
SEED_LIST_FILENAME=seed_list.dat
# Generate seeds
python ./seed_gen.py $NUM_OF_SEEDS $SEED_LIST_FILENAME

# Read the file containing seeds and load to array seed_list
mapfile seed_list < $SEED_LIST_FILENAME

# Execute experiments using seeds from the seed_list array
for i in ${seed_list[@]}; do
    python ./dqn-cartpole.py $i base_config
done
