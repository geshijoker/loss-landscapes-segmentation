#!/bin/bash
#SBATCH -A m636
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 3:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -o logs/crf_batch_sizes.%N.%j.out # STDOUT
#SBATCH -e logs/crf_batch_sizes.%N.%j.err # STDERR

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
batchsizes=(32 16 8 4 1)
seed=234

# Iterate over the array using a for loop
for batchsize in "${batchsizes[@]} "; do
    echo "Processing batchsize: $batchsize, seed: $seed"
    
    _train_cmd="python -u /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/train_Fiber_crfseg.py \
        -d /global/cfs/projectdirs/m636/Vis4ML/Fiber/Quarter \
        -e /global/cfs/cdirs/m636/geshi/exp/Fiber/batch_size/CrossEntropy/crf/seed_$seed \
        -n 0_bs_$batchsize\
        -a unet-crf \
        -l ce \
        -s $seed \
        -p 0.1 \
        -g 4 \
        -f 20 \
        -ne 20 \
        -lr 0.001 \
        -bs $batchsize \
        -ad 5 \
        -aw 32 \
        -ip 288 \
        -t --benchmark --verbose"

    echo $_train_cmd
    echo ""
    srun -n 1 $_train_cmd &
    echo ""
    
done
wait