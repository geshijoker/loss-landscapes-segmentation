#!/bin/bash
#SBATCH -A m636
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 8:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -o logs/non_crf_label_noises.%N.%j.out # STDOUT
#SBATCH -e logs/non_crf_label_noises.%N.%j.err # STDERR

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
label_noises=(0.001 0.0005 0.0002 0.0001 0.0)
seed=234

# Iterate over the array using a for loop
for labelnoise in "${label_noises[@]} "; do
    echo "Processing labelnoise: $labelnoise, seed: $seed"
    
    _train_cmd="python -u /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/train_Oxford_crfseg.py \
        -d /global/cfs/cdirs/m636/geshi/data/ \
        -e /global/cfs/cdirs/m636/geshi/exp/Oxford/label_noises/CrossEntropy/non-crf/seed_$seed \
        -n 0_ln_${labelnoise/"."/""} \
        -a unet \
        -l ce \
        -s $seed \
        -p 1 \
        -g 0 \
        -f 20 \
        -ne 20 \
        -ln $labelnoise \
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