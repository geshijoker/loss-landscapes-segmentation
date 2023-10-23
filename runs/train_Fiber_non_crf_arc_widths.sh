#!/bin/bash
#SBATCH -A m636
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 3:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -o logs/percentages.%N.%j..out # STDOUT
#SBATCH -e logs/percentages.%N.%j..err # STDERR

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
arcwidths=(5 4 3 2 1)
seed=234

# Iterate over the array using a for loop
for arcwidth in "${arcwidths[@]} "; do
    echo "Processing arcwidth: $arcwidth, seed: $seed"
    
    _train_cmd="python /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/train_Fiber_crfseg.py \
        -d /global/cfs/projectdirs/m636/Vis4ML/Fiber/Quarter \
        -e /global/cfs/cdirs/m636/geshi/exp/Fiber/arc_widths/CrossEntropy/non-crf/seed_$seed \
        -n 0_aw_${arcwidth}\
        -a unet \
        -l ce \
        -s $seed \
        -p 0.001 \
        -g 4 \
        -f 1 \
        -ne 20 \
        -lr 0.001 \
        -bs 32 \
        -ad 5 \
        -aw $arcwidth \
        -ip 288 \
        -t --benchmark --verbose"

    echo $_train_cmd
    echo ""
    srun -n 1 $_train_cmd
    echo ""
    
done
wait