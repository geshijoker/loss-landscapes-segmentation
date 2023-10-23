#!/bin/bash

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

# # one gpu visible to each task 
# export SLURM_CPU_BIND="cores"

# # ensures that the python output sent to terminal without buffering
# export PYTHONUNBUFFERED=1


###############################################################################
### run scripts here
###############################################################################

# define the portion of training data to use for training
percentages=(0.1 0.05 0.02 0.01 0.005 0.002 0.001)
seed=234

# Iterate over the array using a for loop
for percentage in "${percentages[@]} "; do
    echo "Processing percentage: $percentage, seed: $seed"
    
    _train_cmd="python /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/train_Fiber_crfseg.py \
        -d /global/cfs/projectdirs/m636/Vis4ML/Fiber/Quarter \
        -e /global/cfs/cdirs/m636/geshi/exp/Fiber/percentage/CrossEntropy/non-crf/seed_$seed \
        -n 0_p_${percentage/"."/""}\
        -a unet \
        -l ce \
        -s $seed \
        -p $percentage \
        -g 4 \
        -f 1 \
        -ne 20 \
        -lr 0.001 \
        -bs 32 \
        -ad 5 \
        -aw 32 \
        -ip 288 \
        -t --benchmark --verbose"

    echo ""
    echo "$_train_cmd"
    echo ""
    
done
wait