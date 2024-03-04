#!/bin/bash
#SBATCH -A m636
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 1:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -o logs/crf_seed_8049.%N.%j.out # STDOUT
#SBATCH -e logs/crf_seed_8049.%N.%j.err # STDERR

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
seed=8049
paths=("/global/cfs/cdirs/m636/geshi/exp/Oxford/trajectory/non-crf/CrossEntropy/seed_8049/0_lr_001_seed_8049/iter0-02-26-2024-23:10:17.pt" "/global/cfs/cdirs/m636/geshi/exp/Oxford/trajectory/non-crf/CrossEntropy/seed_8049/0_lr_001_seed_8049/iter20-02-26-2024-23:51:25.pt" "/global/cfs/cdirs/m636/geshi/exp/Oxford/trajectory/non-crf/CrossEntropy/seed_8049/0_lr_001_seed_8049/iter30-02-27-2024-00:12:31.pt" "/global/cfs/cdirs/m636/geshi/exp/Oxford/trajectory/non-crf/CrossEntropy/seed_8049/0_lr_001_seed_8049/iter10-02-26-2024-23:30:54.pt" "/global/cfs/cdirs/m636/geshi/exp/Oxford/trajectory/non-crf/CrossEntropy/seed_8049/0_lr_001_seed_8049/iter25-02-27-2024-00:02:08.pt" "/global/cfs/cdirs/m636/geshi/exp/Oxford/trajectory/non-crf/CrossEntropy/seed_8049/0_lr_001_seed_8049/iter15-02-26-2024-23:41:15.pt" "/global/cfs/cdirs/m636/geshi/exp/Oxford/trajectory/non-crf/CrossEntropy/seed_8049/0_lr_001_seed_8049/iter5-02-26-2024-23:20:45.pt")

# Iterate over the array using a for loop
for path in "${paths[@]} "; do
    echo "Processing model $path"
    
    _eval_cmd="python -u /global/u2/g/geshi/loss-landscapes-segmentation/loss_examples/Oxford_unet_hessian.py \
        -d /global/cfs/cdirs/m636/geshi/data/ \
        -r $path \
        -sa /global/u2/g/geshi/loss-landscapes-segmentation/loss_examples/unet-pyhessian-loss-landscapes \
        -s $seed \
        -g 0 \
        -ad 5 \
        -aw 16 \
        -ip 224 \
        -bs 32 \
        --benchmark --verbose"

    echo $_eval_cmd
    echo ""
    srun -n 1 $_eval_cmd &
    echo ""
    
done
wait