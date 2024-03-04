#!/bin/bash
#SBATCH -A m636
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 1:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -o logs/ed_seed_8049.%N.%j.out # STDOUT
#SBATCH -e logs/ed_seed_8049.%N.%j.err # STDERR

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
paths=("/global/cfs/cdirs/m636/geshi/exp/Oxford/trajectory/ed/CrossEntropy/seed_8049/0_lr_001_seed_8049/iter5-03-01-2024-05:22:26.pt" "/global/cfs/cdirs/m636/geshi/exp/Oxford/trajectory/ed/CrossEntropy/seed_8049/0_lr_001_seed_8049/iter20-03-01-2024-05:49:46.pt" "/global/cfs/cdirs/m636/geshi/exp/Oxford/trajectory/ed/CrossEntropy/seed_8049/0_lr_001_seed_8049/iter25-03-01-2024-05:58:37.pt" "/global/cfs/cdirs/m636/geshi/exp/Oxford/trajectory/ed/CrossEntropy/seed_8049/0_lr_001_seed_8049/iter15-03-01-2024-05:40:56.pt" "/global/cfs/cdirs/m636/geshi/exp/Oxford/trajectory/ed/CrossEntropy/seed_8049/0_lr_001_seed_8049/iter0-03-01-2024-05:11:55.pt" "/global/cfs/cdirs/m636/geshi/exp/Oxford/trajectory/ed/CrossEntropy/seed_8049/0_lr_001_seed_8049/iter30-03-01-2024-06:07:27.pt" "/global/cfs/cdirs/m636/geshi/exp/Oxford/trajectory/ed/CrossEntropy/seed_8049/0_lr_001_seed_8049/iter10-03-01-2024-05:31:59.pt")

# Iterate over the array using a for loop
for path in "${paths[@]} "; do
    echo "Processing model $path"
    
    _eval_cmd="python -u /global/u2/g/geshi/loss-landscapes-segmentation/loss_examples/Oxford_ed_hessian.py \
        -d /global/cfs/cdirs/m636/geshi/data/ \
        -r $path \
        -sa /global/u2/g/geshi/loss-landscapes-segmentation/loss_examples/ed-pyhessian-loss-landscapes \
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