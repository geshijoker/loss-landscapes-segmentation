#!/bin/bash
#SBATCH -A m636
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 6:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -o logs/crf_seeds.%N.%j.out # STDOUT
#SBATCH -e logs/crf_seeds.%N.%j.err # STDERR

###############################################################################
### setup here
###############################################################################

pwd
hostname
date
echo starting job...

# Load the conda module
module load python

# Activate the virtual environment
conda activate CRF_GPU_Env

# one gpu visible to each task 
export SLURM_CPU_BIND="cores"

# ensures that the python output sent to terminal without buffering
export PYTHONUNBUFFERED=1


###############################################################################
### run scripts here
###############################################################################

# define the portion of training data to use for training
seeds=(243 3376 5370 7026 8049)

# Iterate over the array using a for loop
for seed in "${seeds[@]} "; do
    echo "Processing seed $seed"
    
    _train_cmd="python -u /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/train_Oxford_crfseg.py \
        -d /global/cfs/cdirs/m636/geshi/data/ \
        -e /global/cfs/cdirs/m636/geshi/exp/Oxford/seeds/CrossEntropy/crf/seed_$seed \
        -n optimal \
        -a unet-crf \
        -l ce \
        -s $seed \
        -p 1 \
        -g 0 \
        -f 10 \
        -ne 30 \
        -ln 0.0
        -lr 0.001 \
        -bs 32 \
        -ad 5 \
        -aw 32 \
        -ip 224 \
        -t --benchmark --verbose"

    echo $_train_cmd
    echo ""
    srun -n 1 $_train_cmd &
    echo ""
    
done
wait