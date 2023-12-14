#!/bin/bash
#SBATCH -A m636
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 2:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -o logs/eval_Oxford_extra.%N.%j.out # STDOUT
#SBATCH -e logs/eval_Oxford_extra.%N.%j.err # STDERR

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
TRANS="/global/cfs/cdirs/m636/geshi/exp/Oxford/arc_widths/CrossEntropy/trans-crf/seed_234/0_aw_32_seed_234/iter10-11-21-2023-18:11:01.pt"
LATE="/global/cfs/cdirs/m636/geshi/exp/Oxford/arc_widths/CrossEntropy/crf-late/seed_234/0_aw_32_seed_234/iter10-11-21-2023-17:44:35.pt"
srun -n 1 python -u /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/eval_Oxford_crfseg.py -d /global/cfs/cdirs/m636/geshi/data/ -r $TRANS -a unet-crf -s 234 -g 0 -p 1 -ad 5 -aw 32 -ip 224 -bs 32 --benchmark --verbose &
srun -n 1 python -u /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/eval_Oxford_unetnocrf.py -d /global/cfs/cdirs/m636/geshi/data/ -r $TRANS -a unet-crf -s 234 -g 0 -p 1 -ad 5 -aw 32 -ip 224 -bs 32 --benchmark --verbose &
srun -n 1 python -u /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/eval_Oxford_crfseg.py -d /global/cfs/cdirs/m636/geshi/data/ -r $LATE -a unet-crf -s 234 -g 0 -p 1 -ad 5 -aw 32 -ip 224 -bs 32 --benchmark --verbose &
srun -n 1 python -u /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/eval_Oxford_unetnocrf.py -d /global/cfs/cdirs/m636/geshi/data/ -r $LATE -a unet-crf -s 234 -g 0 -p 1 -ad 5 -aw 32 -ip 224 -bs 32 --benchmark --verbose &

# batchsizes 32
TRANS="/global/cfs/cdirs/m636/geshi/exp/Oxford/batch_size/CrossEntropy/trans-crf/seed_234/0_bs_32_seed_234/iter10-11-21-2023-18:11:01.pt"
LATE="/global/cfs/cdirs/m636/geshi/exp/Oxford/batch_size/CrossEntropy/crf-late/seed_234/0_bs_32_seed_234/iter10-11-21-2023-17:16:48.pt"
srun -n 1 python -u /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/eval_Oxford_crfseg.py -d /global/cfs/cdirs/m636/geshi/data/ -r $TRANS -a unet-crf -s 234 -g 0 -p 1 -ad 5 -aw 32 -ip 224 -bs 32 --benchmark --verbose &
srun -n 1 python -u /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/eval_Oxford_unetnocrf.py -d /global/cfs/cdirs/m636/geshi/data/ -r $TRANS -a unet-crf -s 234 -g 0 -p 1 -ad 5 -aw 32 -ip 224 -bs 32 --benchmark --verbose &
srun -n 1 python -u /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/eval_Oxford_crfseg.py -d /global/cfs/cdirs/m636/geshi/data/ -r $LATE -a unet-crf -s 234 -g 0 -p 1 -ad 5 -aw 32 -ip 224 -bs 32 --benchmark --verbose &
srun -n 1 python -u /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/eval_Oxford_unetnocrf.py -d /global/cfs/cdirs/m636/geshi/data/ -r $LATE -a unet-crf -s 234 -g 0 -p 1 -ad 5 -aw 32 -ip 224 -bs 32 --benchmark --verbose &

# learningrates 0.001
TRANS="/global/cfs/cdirs/m636/geshi/exp/Oxford/learning_rate/CrossEntropy/trans-crf/seed_234/0_lr_0001_seed_234/iter10-11-21-2023-17:16:49.pt"
LATE="/global/cfs/cdirs/m636/geshi/exp/Oxford/learning_rate/CrossEntropy/crf-late/seed_234/0_lr_0001_seed_234/iter10-11-21-2023-17:16:48.pt"
srun -n 1 python -u /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/eval_Oxford_crfseg.py -d /global/cfs/cdirs/m636/geshi/data/ -r $TRANS -a unet-crf -s 234 -g 0 -p 1 -ad 5 -aw 32 -ip 224 -bs 32 --benchmark --verbose &
srun -n 1 python -u /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/eval_Oxford_unetnocrf.py -d /global/cfs/cdirs/m636/geshi/data/ -r $TRANS -a unet-crf -s 234 -g 0 -p 1 -ad 5 -aw 32 -ip 224 -bs 32 --benchmark --verbose &
srun -n 1 python -u /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/eval_Oxford_crfseg.py -d /global/cfs/cdirs/m636/geshi/data/ -r $LATE -a unet-crf -s 234 -g 0 -p 1 -ad 5 -aw 32 -ip 224 -bs 32 --benchmark --verbose &
srun -n 1 python -u /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/eval_Oxford_unetnocrf.py -d /global/cfs/cdirs/m636/geshi/data/ -r $LATE -a unet-crf -s 234 -g 0 -p 1 -ad 5 -aw 32 -ip 224 -bs 32 --benchmark --verbose &

# percentages 1.0
TRANS="/global/cfs/cdirs/m636/geshi/exp/Oxford/percentage/CrossEntropy/trans-crf/seed_234/0_p_10_seed_234/iter10-11-21-2023-17:44:40.pt"
LATE="/global/cfs/cdirs/m636/geshi/exp/Oxford/percentage/CrossEntropy/crf-late/seed_234/0_p_10_seed_234/iter10-11-21-2023-17:44:35.pt"
srun -n 1 python -u /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/eval_Oxford_crfseg.py -d /global/cfs/cdirs/m636/geshi/data/ -r $TRANS -a unet-crf -s 234 -g 0 -p 1 -ad 5 -aw 32 -ip 224 -bs 32 --benchmark --verbose &
srun -n 1 python -u /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/eval_Oxford_unetnocrf.py -d /global/cfs/cdirs/m636/geshi/data/ -r $TRANS -a unet-crf -s 234 -g 0 -p 1 -ad 5 -aw 32 -ip 224 -bs 32 --benchmark --verbose &
srun -n 1 python -u /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/eval_Oxford_crfseg.py -d /global/cfs/cdirs/m636/geshi/data/ -r $LATE -a unet-crf -s 234 -g 0 -p 1 -ad 5 -aw 32 -ip 224 -bs 32 --benchmark --verbose &
srun -n 1 python -u /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/eval_Oxford_unetnocrf.py -d /global/cfs/cdirs/m636/geshi/data/ -r $LATE -a unet-crf -s 234 -g 0 -p 1 -ad 5 -aw 32 -ip 224 -bs 32 --benchmark --verbose &

# labelnoises 0.0
TRANS="/global/cfs/cdirs/m636/geshi/exp/Oxford/label_noises/CrossEntropy/trans-crf/seed_234/0_ln_00_seed_234/iter10-11-21-2023-17:16:49.pt"
LATE="/global/cfs/cdirs/m636/geshi/exp/Oxford/label_noises/CrossEntropy/crf-late/seed_234/0_ln_00_seed_234/iter10-11-21-2023-17:44:35.pt"
srun -n 1 python -u /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/eval_Oxford_crfseg.py -d /global/cfs/cdirs/m636/geshi/data/ -r $TRANS -a unet-crf -s 234 -g 0 -p 1 -ad 5 -aw 32 -ip 224 -bs 32 --benchmark --verbose &
srun -n 1 python -u /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/eval_Oxford_unetnocrf.py -d /global/cfs/cdirs/m636/geshi/data/ -r $TRANS -a unet-crf -s 234 -g 0 -p 1 -ad 5 -aw 32 -ip 224 -bs 32 --benchmark --verbose &
srun -n 1 python -u /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/eval_Oxford_crfseg.py -d /global/cfs/cdirs/m636/geshi/data/ -r $LATE -a unet-crf -s 234 -g 0 -p 1 -ad 5 -aw 32 -ip 224 -bs 32 --benchmark --verbose &
srun -n 1 python -u /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/eval_Oxford_unetnocrf.py -d /global/cfs/cdirs/m636/geshi/data/ -r $LATE -a unet-crf -s 234 -g 0 -p 1 -ad 5 -aw 32 -ip 224 -bs 32 --benchmark --verbose &

wait