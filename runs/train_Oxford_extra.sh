#!/bin/bash
#SBATCH -A m636
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 6:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -e logs/train_Oxford_extra.%N.%j.err # STDERR

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

# arcwidths 32
UNetCRF="/global/cfs/cdirs/m636/geshi/exp/Oxford/arc_widths/CrossEntropy/crf/seed_234/0_aw_32_seed_234/iter30-11-20-2023-02:33:39.pt"
UNet="/global/cfs/cdirs/m636/geshi/exp/Oxford/arc_widths/CrossEntropy/non-crf/seed_234/0_aw_32_seed_234/iter30-11-20-2023-02:52:23.pt"
srun -n 1 python /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/train_Oxford_crf_trans.py -d /global/cfs/cdirs/m636/geshi/data/ -e /global/cfs/cdirs/m636/geshi/exp/Oxford/arc_widths/CrossEntropy/trans-crf/seed_234 -n 0_aw_32 -l ce -rc $UNetCRF -rn $UNet -s 234 -g 0 -p 1 -f 10 -ne 10 -ln 0.0 -lr 0.0001 -bs 32 -ad 5 -aw 32 -ip 224 -t --benchmark --verbose &
srun -n 1 python /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/train_Oxford_crf_late.py -d /global/cfs/cdirs/m636/geshi/data/ -e /global/cfs/cdirs/m636/geshi/exp/Oxford/arc_widths/CrossEntropy/crf-late/seed_234 -n 0_aw_32 -l ce -r $UNet -s 234 -g 0 -p 1 -f 10 -ne 10 -ln 0.0 -lr 0.0001 -bs 32 -ad 5 -aw 32 -ip 224 -t --benchmark --verbose &

# batch size 32
UNetCRF="/global/cfs/cdirs/m636/geshi/exp/Oxford/batch_size/CrossEntropy/crf/seed_234/0_bs_32_seed_234/iter30-11-20-2023-03:16:18.pt"
UNet="/global/cfs/cdirs/m636/geshi/exp/Oxford/batch_size/CrossEntropy/non-crf/seed_234/0_bs_32_seed_234/iter30-11-20-2023-02:46:55.pt"
srun -n 1 python /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/train_Oxford_crf_trans.py -d /global/cfs/cdirs/m636/geshi/data/ -e /global/cfs/cdirs/m636/geshi/exp/Oxford/batch_size/CrossEntropy/trans-crf/seed_234 -n 0_bs_32 -l ce -rc $UNetCRF -rn $UNet -s 234 -g 0 -p 1 -f 10 -ne 10 -ln 0.0 -lr 0.0001 -bs 32 -ad 5 -aw 32 -ip 224 -t --benchmark --verbose &
srun -n 1 python /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/train_Oxford_crf_late.py -d /global/cfs/cdirs/m636/geshi/data/ -e /global/cfs/cdirs/m636/geshi/exp/Oxford/batch_size/CrossEntropy/crf-late/seed_234 -n 0_bs_32 -l ce -r $UNet -s 234 -g 0 -p 1 -f 10 -ne 10 -ln 0.0 -lr 0.0001 -bs 32 -ad 5 -aw 32 -ip 224 -t --benchmark --verbose &

# learningrates 0.001
UNetCRF="/global/cfs/cdirs/m636/geshi/exp/Oxford/learning_rate/CrossEntropy/crf/seed_234/0_lr_0001_seed_234/iter30-11-20-2023-02:33:39.pt"
UNet="/global/cfs/cdirs/m636/geshi/exp/Oxford/learning_rate/CrossEntropy/non-crf/seed_234/0_lr_0001_seed_234/iter30-11-20-2023-02:28:27.pt"
srun -n 1 python /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/train_Oxford_crf_trans.py -d /global/cfs/cdirs/m636/geshi/data/ -e /global/cfs/cdirs/m636/geshi/exp/Oxford/learning_rate/CrossEntropy/trans-crf/seed_234 -n 0_lr_0001 -l ce -rc $UNetCRF -rn $UNet -s 234 -g 0 -p 1 -f 10 -ne 10 -ln 0.0 -lr 0.0001 -bs 32 -ad 5 -aw 32 -ip 224 -t --benchmark --verbose &
srun -n 1 python /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/train_Oxford_crf_late.py -d /global/cfs/cdirs/m636/geshi/data/ -e /global/cfs/cdirs/m636/geshi/exp/Oxford/learning_rate/CrossEntropy/crf-late/seed_234 -n 0_lr_0001 -l ce -r $UNet -s 234 -g 0 -p 1 -f 10 -ne 10 -ln 0.0 -lr 0.0001 -bs 32 -ad 5 -aw 32 -ip 224 -t --benchmark --verbose &

# percentages 1.0
UNetCRF="/global/cfs/cdirs/m636/geshi/exp/Oxford/percentage/CrossEntropy/crf/seed_234/0_p_10_seed_234/iter30-11-20-2023-02:53:00.pt"
UNet="/global/cfs/cdirs/m636/geshi/exp/Oxford/percentage/CrossEntropy/non-crf/seed_234/0_p_10_seed_234/iter30-11-20-2023-02:28:27.pt"
srun -n 1 python /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/train_Oxford_crf_trans.py -d /global/cfs/cdirs/m636/geshi/data/ -e /global/cfs/cdirs/m636/geshi/exp/Oxford/percentage/CrossEntropy/trans-crf/seed_234 -n 0_p_10 -l ce -rc $UNetCRF -rn $UNet -s 234 -g 0 -p 1 -f 10 -ne 10 -ln 0.0 -lr 0.0001 -bs 32 -ad 5 -aw 32 -ip 224 -t --benchmark --verbose &
srun -n 1 python /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/train_Oxford_crf_late.py -d /global/cfs/cdirs/m636/geshi/data/ -e /global/cfs/cdirs/m636/geshi/exp/Oxford/percentage/CrossEntropy/crf-late/seed_234 -n 0_p_10 -l ce -r $UNet -s 234 -g 0 -p 1 -f 10 -ne 10 -ln 0.0 -lr 0.0001 -bs 32 -ad 5 -aw 32 -ip 224 -t --benchmark --verbose &

# labelnoises 0.0
UNetCRF="/global/cfs/cdirs/m636/geshi/exp/Oxford/label_noises/CrossEntropy/crf/seed_234/0_ln_00_seed_234/iter30-11-20-2023-03:10:49.pt"
UNet="/global/cfs/cdirs/m636/geshi/exp/Oxford/label_noises/CrossEntropy/non-crf/seed_234/0_ln_00_seed_234/iter30-11-20-2023-02:52:23.pt"
srun -n 1 python /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/train_Oxford_crf_trans.py -d /global/cfs/cdirs/m636/geshi/data/ -e /global/cfs/cdirs/m636/geshi/exp/Oxford/label_noises/CrossEntropy/trans-crf/seed_234 -n 0_ln_00 -l ce -rc $UNetCRF -rn $UNet -s 234 -g 0 -p 1 -f 10 -ne 10 -ln 0.0 -lr 0.0001 -bs 32 -ad 5 -aw 32 -ip 224 -t --benchmark --verbose &
srun -n 1 python /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/train_Oxford_crf_late.py -d /global/cfs/cdirs/m636/geshi/data/ -e /global/cfs/cdirs/m636/geshi/exp/Oxford/label_noises/CrossEntropy/crf-late/seed_234 -n 0_ln_00 -l ce -r $UNet -s 234 -g 0 -p 1 -f 10 -ne 10 -ln 0.0 -lr 0.0001 -bs 32 -ad 5 -aw 32 -ip 224 -t --benchmark --verbose &

wait