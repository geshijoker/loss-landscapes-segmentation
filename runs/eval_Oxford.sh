#!/bin/bash
#SBATCH -A m636
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 6:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -o logs/eval_Oxford.%N.%j.out # STDOUT
#SBATCH -e logs/eval_Oxford.%N.%j.err # STDERR

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
UNetCRF="/global/cfs/cdirs/m636/geshi/exp/Oxford/arc_widths/CrossEntropy/crf/seed_234/0_aw_32_seed_234/iter20-10-31-2023-02:41:02.pt"
UNet="/global/cfs/cdirs/m636/geshi/exp/Oxford/arc_widths/CrossEntropy/non-crf/seed_234/0_aw_32_seed_234/iter20-10-31-2023-02:21:02.pt"
srun -n 1 python -u /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/eval_Oxford_crfseg.py -d /global/cfs/cdirs/m636/geshi/data/ -r $UNetCRF -a unet-crf -s 234 -g 0 -p 1 -ad 5 -aw 32 -ip 224 -bs 32 --benchmark --verbose &
srun -n 1 python -u /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/eval_Oxford_crfseg.py -d /global/cfs/cdirs/m636/geshi/data/ -r $UNet -a unet -s 234 -g 0 -p 1 -ad 5 -aw 32 -ip 224 -bs 32 --benchmark --verbose &
srun -n 1 python -u /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/eval_Oxford_crf_trans.py -d /global/cfs/cdirs/m636/geshi/data/ -rc $UNetCRF -rn $UNet -s 234 -g 0 -p 1 -ad 5 -aw 32 -ip 224 -bs 32 --benchmark --verbose &
srun -n 1 python -u /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/eval_Oxford_gtrue_crf.py -d /global/cfs/cdirs/m636/geshi/data/ -r $UNetCRF -a unet-crf -s 234 -g 0 -p 1 -ad 5 -aw 32 -ip 224 -bs 32 --benchmark --verbose &
srun -n 1 python -u /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/eval_Oxford_unetnocrf.py -d /global/cfs/cdirs/m636/geshi/data/ -r $UNetCRF -a unet-crf -s 234 -g 0 -p 1 -ad 5 -aw 32 -ip 224 -bs 32 --benchmark --verbose &

# arcwidths 16
UNetCRF="/global/cfs/cdirs/m636/geshi/exp/Oxford/arc_widths/CrossEntropy/crf/seed_234/0_aw_16_seed_234/iter20-10-31-2023-02:33:30.pt"
UNet="/global/cfs/cdirs/m636/geshi/exp/Oxford/arc_widths/CrossEntropy/non-crf/seed_234/0_aw_16_seed_234/iter20-10-31-2023-02:14:59.pt"
srun -n 1 python -u /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/eval_Oxford_crfseg.py -d /global/cfs/cdirs/m636/geshi/data/ -r $UNetCRF -a unet-crf -s 234 -g 0 -p 1 -ad 5 -aw 16 -ip 224 -bs 32 --benchmark --verbose &
srun -n 1 python -u /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/eval_Oxford_crfseg.py -d /global/cfs/cdirs/m636/geshi/data/ -r $UNet -a unet -s 234 -g 0 -p 1 -ad 5 -aw 16 -ip 224 -bs 32 --benchmark --verbose &
srun -n 1 python -u /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/eval_Oxford_crf_trans.py -d /global/cfs/cdirs/m636/geshi/data/ -rc $UNetCRF -rn $UNet -s 234 -g 0 -p 1 -ad 5 -aw 16 -ip 224 -bs 32 --benchmark --verbose &
srun -n 1 python -u /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/eval_Oxford_gtrue_crf.py -d /global/cfs/cdirs/m636/geshi/data/ -r $UNetCRF -a unet-crf -s 234 -g 0 -p 1 -ad 5 -aw 16 -ip 224 -bs 32 --benchmark --verbose &
srun -n 1 python -u /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/eval_Oxford_unetnocrf.py -d /global/cfs/cdirs/m636/geshi/data/ -r $UNetCRF -a unet-crf -s 234 -g 0 -p 1 -ad 5 -aw 16 -ip 224 -bs 32 --benchmark --verbose &

# arcwidths 8
UNetCRF="/global/cfs/cdirs/m636/geshi/exp/Oxford/arc_widths/CrossEntropy/crf/seed_234/0_aw_8_seed_234/iter20-10-31-2023-02:33:35.pt"
UNet="/global/cfs/cdirs/m636/geshi/exp/Oxford/arc_widths/CrossEntropy/non-crf/seed_234/0_aw_8_seed_234/iter20-10-31-2023-02:13:24.pt"
srun -n 1 python -u /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/eval_Oxford_crfseg.py -d /global/cfs/cdirs/m636/geshi/data/ -r $UNetCRF -a unet-crf -s 234 -g 0 -p 1 -ad 5 -aw 8 -ip 224 -bs 32 --benchmark --verbose &
srun -n 1 python -u /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/eval_Oxford_crfseg.py -d /global/cfs/cdirs/m636/geshi/data/ -r $UNet -a unet -s 234 -g 0 -p 1 -ad 5 -aw 8 -ip 224 -bs 32 --benchmark --verbose &
srun -n 1 python -u /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/eval_Oxford_crf_trans.py -d /global/cfs/cdirs/m636/geshi/data/ -rc $UNetCRF -rn $UNet -s 234 -g 0 -p 1 -ad 5 -aw 8 -ip 224 -bs 32 --benchmark --verbose &
srun -n 1 python -u /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/eval_Oxford_gtrue_crf.py -d /global/cfs/cdirs/m636/geshi/data/ -r $UNetCRF -a unet-crf -s 234 -g 0 -p 1 -ad 5 -aw 8 -ip 224 -bs 32 --benchmark --verbose &
srun -n 1 python -u /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/eval_Oxford_unetnocrf.py -d /global/cfs/cdirs/m636/geshi/data/ -r $UNetCRF -a unet-crf -s 234 -g 0 -p 1 -ad 5 -aw 8 -ip 224 -bs 32 --benchmark --verbose &

# arcwidths 4
UNetCRF="/global/cfs/cdirs/m636/geshi/exp/Oxford/arc_widths/CrossEntropy/crf/seed_234/0_aw_4_seed_234/iter20-10-31-2023-02:33:36.pt"
UNet="/global/cfs/cdirs/m636/geshi/exp/Oxford/arc_widths/CrossEntropy/non-crf/seed_234/0_aw_4_seed_234/iter20-10-31-2023-02:10:50.pt"
srun -n 1 python -u /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/eval_Oxford_crfseg.py -d /global/cfs/cdirs/m636/geshi/data/ -r $UNetCRF -a unet-crf -s 234 -g 0 -p 1 -ad 5 -aw 4 -ip 224 -bs 32 --benchmark --verbose &
srun -n 1 python -u /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/eval_Oxford_crfseg.py -d /global/cfs/cdirs/m636/geshi/data/ -r $UNet -a unet -s 234 -g 0 -p 1 -ad 5 -aw 4 -ip 224 -bs 32 --benchmark --verbose &
srun -n 1 python -u /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/eval_Oxford_crf_trans.py -d /global/cfs/cdirs/m636/geshi/data/ -rc $UNetCRF -rn $UNet -s 234 -g 0 -p 1 -ad 5 -aw 4 -ip 224 -bs 32 --benchmark --verbose &
srun -n 1 python -u /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/eval_Oxford_gtrue_crf.py -d /global/cfs/cdirs/m636/geshi/data/ -r $UNetCRF -a unet-crf -s 234 -g 0 -p 1 -ad 5 -aw 4 -ip 224 -bs 32 --benchmark --verbose &
srun -n 1 python -u /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/eval_Oxford_unetnocrf.py -d /global/cfs/cdirs/m636/geshi/data/ -r $UNetCRF -a unet-crf -s 234 -g 0 -p 1 -ad 5 -aw 4 -ip 224 -bs 32 --benchmark --verbose &

# batchsizes 32
UNetCRF="/global/cfs/cdirs/m636/geshi/exp/Oxford/batch_size/CrossEntropy/crf/seed_234/0_bs_32_seed_234/iter20-10-31-2023-00:41:35.pt"
UNet="/global/cfs/cdirs/m636/geshi/exp/Oxford/batch_size/CrossEntropy/non-crf/seed_234/0_bs_32_seed_234/iter20-10-31-2023-00:43:24.pt"
srun -n 1 python -u /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/eval_Oxford_crfseg.py -d /global/cfs/cdirs/m636/geshi/data/ -r $UNetCRF -a unet-crf -s 234 -g 0 -p 1 -ad 5 -aw 32 -ip 224 -bs 32 --benchmark --verbose &
srun -n 1 python -u /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/eval_Oxford_crfseg.py -d /global/cfs/cdirs/m636/geshi/data/ -r $UNet -a unet -s 234 -g 0 -p 1 -ad 5 -aw 32 -ip 224 -bs 32 --benchmark --verbose &
srun -n 1 python -u /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/eval_Oxford_crf_trans.py -d /global/cfs/cdirs/m636/geshi/data/ -rc $UNetCRF -rn $UNet -s 234 -g 0 -p 1 -ad 5 -aw 32 -ip 224 -bs 32 --benchmark --verbose &
srun -n 1 python -u /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/eval_Oxford_gtrue_crf.py -d /global/cfs/cdirs/m636/geshi/data/ -r $UNetCRF -a unet-crf -s 234 -g 0 -p 1 -ad 5 -aw 32 -ip 224 -bs 32 --benchmark --verbose &
srun -n 1 python -u /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/eval_Oxford_unetnocrf.py -d /global/cfs/cdirs/m636/geshi/data/ -r $UNetCRF -a unet-crf -s 234 -g 0 -p 1 -ad 5 -aw 32 -ip 224 -bs 32 --benchmark --verbose &

# batchsizes 16
UNetCRF="/global/cfs/cdirs/m636/geshi/exp/Oxford/batch_size/CrossEntropy/crf/seed_234/0_bs_16_seed_234/iter20-10-31-2023-00:45:02.pt"
UNet="/global/cfs/cdirs/m636/geshi/exp/Oxford/batch_size/CrossEntropy/non-crf/seed_234/0_bs_16_seed_234/iter20-10-31-2023-00:45:24.pt"
srun -n 1 python -u /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/eval_Oxford_crfseg.py -d /global/cfs/cdirs/m636/geshi/data/ -r $UNetCRF -a unet-crf -s 234 -g 0 -p 1 -ad 5 -aw 32 -ip 224 -bs 32 --benchmark --verbose &
srun -n 1 python -u /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/eval_Oxford_crfseg.py -d /global/cfs/cdirs/m636/geshi/data/ -r $UNet -a unet -s 234 -g 0 -p 1 -ad 5 -aw 32 -ip 224 -bs 32 --benchmark --verbose &
srun -n 1 python -u /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/eval_Oxford_crf_trans.py -d /global/cfs/cdirs/m636/geshi/data/ -rc $UNetCRF -rn $UNet -s 234 -g 0 -p 1 -ad 5 -aw 32 -ip 224 -bs 32 --benchmark --verbose &
srun -n 1 python -u /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/eval_Oxford_gtrue_crf.py -d /global/cfs/cdirs/m636/geshi/data/ -r $UNetCRF -a unet-crf -s 234 -g 0 -p 1 -ad 5 -aw 32 -ip 224 -bs 32 --benchmark --verbose &
srun -n 1 python -u /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/eval_Oxford_unetnocrf.py -d /global/cfs/cdirs/m636/geshi/data/ -r $UNetCRF -a unet-crf -s 234 -g 0 -p 1 -ad 5 -aw 32 -ip 224 -bs 32 --benchmark --verbose &

# batchsizes 8
UNetCRF="/global/cfs/cdirs/m636/geshi/exp/Oxford/batch_size/CrossEntropy/crf/seed_234/0_bs_8_seed_234/iter20-10-31-2023-00:52:06.pt"
UNet="/global/cfs/cdirs/m636/geshi/exp/Oxford/batch_size/CrossEntropy/non-crf/seed_234/0_bs_8_seed_234/iter20-10-31-2023-03:17:24.pt"
srun -n 1 python -u /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/eval_Oxford_crfseg.py -d /global/cfs/cdirs/m636/geshi/data/ -r $UNetCRF -a unet-crf -s 234 -g 0 -p 1 -ad 5 -aw 32 -ip 224 -bs 32 --benchmark --verbose &
srun -n 1 python -u /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/eval_Oxford_crfseg.py -d /global/cfs/cdirs/m636/geshi/data/ -r $UNet -a unet -s 234 -g 0 -p 1 -ad 5 -aw 32 -ip 224 -bs 32 --benchmark --verbose &
srun -n 1 python -u /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/eval_Oxford_crf_trans.py -d /global/cfs/cdirs/m636/geshi/data/ -rc $UNetCRF -rn $UNet -s 234 -g 0 -p 1 -ad 5 -aw 32 -ip 224 -bs 32 --benchmark --verbose &
srun -n 1 python -u /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/eval_Oxford_gtrue_crf.py -d /global/cfs/cdirs/m636/geshi/data/ -r $UNetCRF -a unet-crf -s 234 -g 0 -p 1 -ad 5 -aw 32 -ip 224 -bs 32 --benchmark --verbose &
srun -n 1 python -u /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/eval_Oxford_unetnocrf.py -d /global/cfs/cdirs/m636/geshi/data/ -r $UNetCRF -a unet-crf -s 234 -g 0 -p 1 -ad 5 -aw 32 -ip 224 -bs 32 --benchmark --verbose &

# batchsizes 4
UNetCRF="/global/cfs/cdirs/m636/geshi/exp/Oxford/batch_size/CrossEntropy/crf/seed_234/0_bs_4_seed_234/iter20-10-31-2023-00:51:35.pt"
UNet="/global/cfs/cdirs/m636/geshi/exp/Oxford/batch_size/CrossEntropy/non-crf/seed_234/0_bs_4_seed_234/iter20-10-31-2023-00:53:37.pt"
srun -n 1 python -u /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/eval_Oxford_crfseg.py -d /global/cfs/cdirs/m636/geshi/data/ -r $UNetCRF -a unet-crf -s 234 -g 0 -p 1 -ad 5 -aw 32 -ip 224 -bs 32 --benchmark --verbose &
srun -n 1 python -u /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/eval_Oxford_crfseg.py -d /global/cfs/cdirs/m636/geshi/data/ -r $UNet -a unet -s 234 -g 0 -p 1 -ad 5 -aw 32 -ip 224 -bs 32 --benchmark --verbose &
srun -n 1 python -u /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/eval_Oxford_crf_trans.py -d /global/cfs/cdirs/m636/geshi/data/ -rc $UNetCRF -rn $UNet -s 234 -g 0 -p 1 -ad 5 -aw 32 -ip 224 -bs 32 --benchmark --verbose &
srun -n 1 python -u /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/eval_Oxford_gtrue_crf.py -d /global/cfs/cdirs/m636/geshi/data/ -r $UNetCRF -a unet-crf -s 234 -g 0 -p 1 -ad 5 -aw 32 -ip 224 -bs 32 --benchmark --verbose &
srun -n 1 python -u /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/eval_Oxford_unetnocrf.py -d /global/cfs/cdirs/m636/geshi/data/ -r $UNetCRF -a unet-crf -s 234 -g 0 -p 1 -ad 5 -aw 32 -ip 224 -bs 32 --benchmark --verbose &

# batchsizes 1
UNetCRF="/global/cfs/cdirs/m636/geshi/exp/Oxford/batch_size/CrossEntropy/crf/seed_234/0_bs_1_seed_234/iter20-11-06-2023-20:41:21.pt"
UNet="/global/cfs/cdirs/m636/geshi/exp/Oxford/batch_size/CrossEntropy/non-crf/seed_234/0_bs_1_seed_234/iter20-10-31-2023-01:17:44.pt"
srun -n 1 python -u /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/eval_Oxford_crfseg.py -d /global/cfs/cdirs/m636/geshi/data/ -r $UNetCRF -a unet-crf -s 234 -g 0 -p 1 -ad 5 -aw 32 -ip 224 -bs 32 --benchmark --verbose &
srun -n 1 python -u /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/eval_Oxford_crfseg.py -d /global/cfs/cdirs/m636/geshi/data/ -r $UNet -a unet -s 234 -g 0 -p 1 -ad 5 -aw 32 -ip 224 -bs 32 --benchmark --verbose &
srun -n 1 python -u /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/eval_Oxford_crf_trans.py -d /global/cfs/cdirs/m636/geshi/data/ -rc $UNetCRF -rn $UNet -s 234 -g 0 -p 1 -ad 5 -aw 32 -ip 224 -bs 32 --benchmark --verbose &
srun -n 1 python -u /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/eval_Oxford_gtrue_crf.py -d /global/cfs/cdirs/m636/geshi/data/ -r $UNetCRF -a unet-crf -s 234 -g 0 -p 1 -ad 5 -aw 32 -ip 224 -bs 32 --benchmark --verbose &
srun -n 1 python -u /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/eval_Oxford_unetnocrf.py -d /global/cfs/cdirs/m636/geshi/data/ -r $UNetCRF -a unet-crf -s 234 -g 0 -p 1 -ad 5 -aw 32 -ip 224 -bs 32 --benchmark --verbose &

# learningrates 0.002
UNetCRF="/global/cfs/cdirs/m636/geshi/exp/Oxford/learning_rate/CrossEntropy/crf/seed_234/0_lr_0002_seed_234/iter20-10-31-2023-02:46:26.pt"
UNet="/global/cfs/cdirs/m636/geshi/exp/Oxford/learning_rate/CrossEntropy/non-crf/seed_234/0_lr_0002_seed_234/iter20-10-31-2023-02:33:05.pt"
srun -n 1 python -u /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/eval_Oxford_crfseg.py -d /global/cfs/cdirs/m636/geshi/data/ -r $UNetCRF -a unet-crf -s 234 -g 0 -p 1 -ad 5 -aw 32 -ip 224 -bs 32 --benchmark --verbose &
srun -n 1 python -u /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/eval_Oxford_crfseg.py -d /global/cfs/cdirs/m636/geshi/data/ -r $UNet -a unet -s 234 -g 0 -p 1 -ad 5 -aw 32 -ip 224 -bs 32 --benchmark --verbose &
srun -n 1 python -u /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/eval_Oxford_crf_trans.py -d /global/cfs/cdirs/m636/geshi/data/ -rc $UNetCRF -rn $UNet -s 234 -g 0 -p 1 -ad 5 -aw 32 -ip 224 -bs 32 --benchmark --verbose &
srun -n 1 python -u /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/eval_Oxford_gtrue_crf.py -d /global/cfs/cdirs/m636/geshi/data/ -r $UNetCRF -a unet-crf -s 234 -g 0 -p 1 -ad 5 -aw 32 -ip 224 -bs 32 --benchmark --verbose &
srun -n 1 python -u /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/eval_Oxford_unetnocrf.py -d /global/cfs/cdirs/m636/geshi/data/ -r $UNetCRF -a unet-crf -s 234 -g 0 -p 1 -ad 5 -aw 32 -ip 224 -bs 32 --benchmark --verbose &

# learningrates 0.001
UNetCRF="/global/cfs/cdirs/m636/geshi/exp/Oxford/learning_rate/CrossEntropy/crf/seed_234/0_lr_0001_seed_234/iter20-10-31-2023-02:55:17.pt"
UNet="/global/cfs/cdirs/m636/geshi/exp/Oxford/learning_rate/CrossEntropy/non-crf/seed_234/0_lr_0001_seed_234/iter20-10-31-2023-02:33:11.pt"
srun -n 1 python -u /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/eval_Oxford_crfseg.py -d /global/cfs/cdirs/m636/geshi/data/ -r $UNetCRF -a unet-crf -s 234 -g 0 -p 1 -ad 5 -aw 32 -ip 224 -bs 32 --benchmark --verbose &
srun -n 1 python -u /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/eval_Oxford_crfseg.py -d /global/cfs/cdirs/m636/geshi/data/ -r $UNet -a unet -s 234 -g 0 -p 1 -ad 5 -aw 32 -ip 224 -bs 32 --benchmark --verbose &
srun -n 1 python -u /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/eval_Oxford_crf_trans.py -d /global/cfs/cdirs/m636/geshi/data/ -rc $UNetCRF -rn $UNet -s 234 -g 0 -p 1 -ad 5 -aw 32 -ip 224 -bs 32 --benchmark --verbose &
srun -n 1 python -u /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/eval_Oxford_gtrue_crf.py -d /global/cfs/cdirs/m636/geshi/data/ -r $UNetCRF -a unet-crf -s 234 -g 0 -p 1 -ad 5 -aw 32 -ip 224 -bs 32 --benchmark --verbose &
srun -n 1 python -u /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/eval_Oxford_unetnocrf.py -d /global/cfs/cdirs/m636/geshi/data/ -r $UNetCRF -a unet-crf -s 234 -g 0 -p 1 -ad 5 -aw 32 -ip 224 -bs 32 --benchmark --verbose &

# learningrates 0.0005
UNetCRF="/global/cfs/cdirs/m636/geshi/exp/Oxford/learning_rate/CrossEntropy/crf/seed_234/0_lr_00005_seed_234/iter20-11-07-2023-10:55:16.pt"
UNet="/global/cfs/cdirs/m636/geshi/exp/Oxford/learning_rate/CrossEntropy/non-crf/seed_234/0_lr_00005_seed_234/iter20-10-31-2023-02:33:05.pt"
srun -n 1 python -u /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/eval_Oxford_crfseg.py -d /global/cfs/cdirs/m636/geshi/data/ -r $UNetCRF -a unet-crf -s 234 -g 0 -p 1 -ad 5 -aw 32 -ip 224 -bs 32 --benchmark --verbose &
srun -n 1 python -u /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/eval_Oxford_crfseg.py -d /global/cfs/cdirs/m636/geshi/data/ -r $UNet -a unet -s 234 -g 0 -p 1 -ad 5 -aw 32 -ip 224 -bs 32 --benchmark --verbose &
srun -n 1 python -u /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/eval_Oxford_crf_trans.py -d /global/cfs/cdirs/m636/geshi/data/ -rc $UNetCRF -rn $UNet -s 234 -g 0 -p 1 -ad 5 -aw 32 -ip 224 -bs 32 --benchmark --verbose &
srun -n 1 python -u /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/eval_Oxford_gtrue_crf.py -d /global/cfs/cdirs/m636/geshi/data/ -r $UNetCRF -a unet-crf -s 234 -g 0 -p 1 -ad 5 -aw 32 -ip 224 -bs 32 --benchmark --verbose &
srun -n 1 python -u /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/eval_Oxford_unetnocrf.py -d /global/cfs/cdirs/m636/geshi/data/ -r $UNetCRF -a unet-crf -s 234 -g 0 -p 1 -ad 5 -aw 32 -ip 224 -bs 32 --benchmark --verbose &

# learningrates 0.0002
UNetCRF="/global/cfs/cdirs/m636/geshi/exp/Oxford/learning_rate/CrossEntropy/crf/seed_234/0_lr_00002_seed_234/iter20-11-07-2023-10:55:16.pt"
UNet="/global/cfs/cdirs/m636/geshi/exp/Oxford/learning_rate/CrossEntropy/non-crf/seed_234/0_lr_00002_seed_234/iter20-11-07-2023-10:44:24.pt"
srun -n 1 python -u /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/eval_Oxford_crfseg.py -d /global/cfs/cdirs/m636/geshi/data/ -r $UNetCRF -a unet-crf -s 234 -g 0 -p 1 -ad 5 -aw 32 -ip 224 -bs 32 --benchmark --verbose &
srun -n 1 python -u /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/eval_Oxford_crfseg.py -d /global/cfs/cdirs/m636/geshi/data/ -r $UNet -a unet -s 234 -g 0 -p 1 -ad 5 -aw 32 -ip 224 -bs 32 --benchmark --verbose &
srun -n 1 python -u /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/eval_Oxford_crf_trans.py -d /global/cfs/cdirs/m636/geshi/data/ -rc $UNetCRF -rn $UNet -s 234 -g 0 -p 1 -ad 5 -aw 32 -ip 224 -bs 32 --benchmark --verbose &
srun -n 1 python -u /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/eval_Oxford_gtrue_crf.py -d /global/cfs/cdirs/m636/geshi/data/ -r $UNetCRF -a unet-crf -s 234 -g 0 -p 1 -ad 5 -aw 32 -ip 224 -bs 32 --benchmark --verbose &
srun -n 1 python -u /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/eval_Oxford_unetnocrf.py -d /global/cfs/cdirs/m636/geshi/data/ -r $UNetCRF -a unet-crf -s 234 -g 0 -p 1 -ad 5 -aw 32 -ip 224 -bs 32 --benchmark --verbose &

# learningrates 0.0001
UNetCRF="/global/cfs/cdirs/m636/geshi/exp/Oxford/learning_rate/CrossEntropy/crf/seed_234/0_lr_00001_seed_234/iter20-10-31-2023-02:46:26.pt"
UNet="/global/cfs/cdirs/m636/geshi/exp/Oxford/learning_rate/CrossEntropy/non-crf/seed_234/0_lr_00001_seed_234/iter20-11-07-2023-10:44:24.pt"
srun -n 1 python -u /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/eval_Oxford_crfseg.py -d /global/cfs/cdirs/m636/geshi/data/ -r $UNetCRF -a unet-crf -s 234 -g 0 -p 1 -ad 5 -aw 32 -ip 224 -bs 32 --benchmark --verbose &
srun -n 1 python -u /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/eval_Oxford_crfseg.py -d /global/cfs/cdirs/m636/geshi/data/ -r $UNet -a unet -s 234 -g 0 -p 1 -ad 5 -aw 32 -ip 224 -bs 32 --benchmark --verbose &
srun -n 1 python -u /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/eval_Oxford_crf_trans.py -d /global/cfs/cdirs/m636/geshi/data/ -rc $UNetCRF -rn $UNet -s 234 -g 0 -p 1 -ad 5 -aw 32 -ip 224 -bs 32 --benchmark --verbose &
srun -n 1 python -u /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/eval_Oxford_gtrue_crf.py -d /global/cfs/cdirs/m636/geshi/data/ -r $UNetCRF -a unet-crf -s 234 -g 0 -p 1 -ad 5 -aw 32 -ip 224 -bs 32 --benchmark --verbose &
srun -n 1 python -u /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/eval_Oxford_unetnocrf.py -d /global/cfs/cdirs/m636/geshi/data/ -r $UNetCRF -a unet-crf -s 234 -g 0 -p 1 -ad 5 -aw 32 -ip 224 -bs 32 --benchmark --verbose &

# learningrates 0.00005
UNetCRF="/global/cfs/cdirs/m636/geshi/exp/Oxford/learning_rate/CrossEntropy/crf/seed_234/0_lr_000005_seed_234/iter20-10-31-2023-02:46:42.pt"
UNet="/global/cfs/cdirs/m636/geshi/exp/Oxford/learning_rate/CrossEntropy/non-crf/seed_234/0_lr_000005_seed_234/iter20-10-31-2023-02:32:58.pt"
srun -n 1 python -u /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/eval_Oxford_crfseg.py -d /global/cfs/cdirs/m636/geshi/data/ -r $UNetCRF -a unet-crf -s 234 -g 0 -p 1 -ad 5 -aw 32 -ip 224 -bs 32 --benchmark --verbose &
srun -n 1 python -u /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/eval_Oxford_crfseg.py -d /global/cfs/cdirs/m636/geshi/data/ -r $UNet -a unet -s 234 -g 0 -p 1 -ad 5 -aw 32 -ip 224 -bs 32 --benchmark --verbose &
srun -n 1 python -u /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/eval_Oxford_crf_trans.py -d /global/cfs/cdirs/m636/geshi/data/ -rc $UNetCRF -rn $UNet -s 234 -g 0 -p 1 -ad 5 -aw 32 -ip 224 -bs 32 --benchmark --verbose &
srun -n 1 python -u /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/eval_Oxford_gtrue_crf.py -d /global/cfs/cdirs/m636/geshi/data/ -r $UNetCRF -a unet-crf -s 234 -g 0 -p 1 -ad 5 -aw 32 -ip 224 -bs 32 --benchmark --verbose &
srun -n 1 python -u /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/eval_Oxford_unetnocrf.py -d /global/cfs/cdirs/m636/geshi/data/ -r $UNetCRF -a unet-crf -s 234 -g 0 -p 1 -ad 5 -aw 32 -ip 224 -bs 32 --benchmark --verbose &

# percentages 1.0 0.5 0.2 0.1 0.05
UNetCRF="/global/cfs/cdirs/m636/geshi/exp/Oxford/percentage/CrossEntropy/crf/seed_234/0_p_10_seed_234/iter20-10-31-2023-04:13:38.pt"
UNet="/global/cfs/cdirs/m636/geshi/exp/Oxford/percentage/CrossEntropy/non-crf/seed_234/0_p_10_seed_234/iter20-10-31-2023-01:42:48.pt"
srun -n 1 python -u /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/eval_Oxford_crfseg.py -d /global/cfs/cdirs/m636/geshi/data/ -r $UNetCRF -a unet-crf -s 234 -g 0 -p 1 -ad 5 -aw 32 -ip 224 -bs 32 --benchmark --verbose &
srun -n 1 python -u /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/eval_Oxford_crfseg.py -d /global/cfs/cdirs/m636/geshi/data/ -r $UNet -a unet -s 234 -g 0 -p 1 -ad 5 -aw 32 -ip 224 -bs 32 --benchmark --verbose &
srun -n 1 python -u /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/eval_Oxford_crf_trans.py -d /global/cfs/cdirs/m636/geshi/data/ -rc $UNetCRF -rn $UNet -s 234 -g 0 -p 1 -ad 5 -aw 32 -ip 224 -bs 32 --benchmark --verbose &
srun -n 1 python -u /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/eval_Oxford_gtrue_crf.py -d /global/cfs/cdirs/m636/geshi/data/ -r $UNetCRF -a unet-crf -s 234 -g 0 -p 1 -ad 5 -aw 32 -ip 224 -bs 32 --benchmark --verbose &
srun -n 1 python -u /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/eval_Oxford_unetnocrf.py -d /global/cfs/cdirs/m636/geshi/data/ -r $UNetCRF -a unet-crf -s 234 -g 0 -p 1 -ad 5 -aw 32 -ip 224 -bs 32 --benchmark --verbose &

# percentages 0.5
UNetCRF="/global/cfs/cdirs/m636/geshi/exp/Oxford/percentage/CrossEntropy/crf/seed_234/0_p_05_seed_234/iter20-10-31-2023-01:32:38.pt"
UNet="/global/cfs/cdirs/m636/geshi/exp/Oxford/percentage/CrossEntropy/non-crf/seed_234/0_p_05_seed_234/iter20-10-31-2023-01:23:42.pt"
srun -n 1 python -u /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/eval_Oxford_crfseg.py -d /global/cfs/cdirs/m636/geshi/data/ -r $UNetCRF -a unet-crf -s 234 -g 0 -p 1 -ad 5 -aw 32 -ip 224 -bs 32 --benchmark --verbose &
srun -n 1 python -u /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/eval_Oxford_crfseg.py -d /global/cfs/cdirs/m636/geshi/data/ -r $UNet -a unet -s 234 -g 0 -p 1 -ad 5 -aw 32 -ip 224 -bs 32 --benchmark --verbose &
srun -n 1 python -u /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/eval_Oxford_crf_trans.py -d /global/cfs/cdirs/m636/geshi/data/ -rc $UNetCRF -rn $UNet -s 234 -g 0 -p 1 -ad 5 -aw 32 -ip 224 -bs 32 --benchmark --verbose &
srun -n 1 python -u /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/eval_Oxford_gtrue_crf.py -d /global/cfs/cdirs/m636/geshi/data/ -r $UNetCRF -a unet-crf -s 234 -g 0 -p 1 -ad 5 -aw 32 -ip 224 -bs 32 --benchmark --verbose &
srun -n 1 python -u /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/eval_Oxford_unetnocrf.py -d /global/cfs/cdirs/m636/geshi/data/ -r $UNetCRF -a unet-crf -s 234 -g 0 -p 1 -ad 5 -aw 32 -ip 224 -bs 32 --benchmark --verbose &

# percentages 0.2
UNetCRF="/global/cfs/cdirs/m636/geshi/exp/Oxford/percentage/CrossEntropy/crf/seed_234/0_p_02_seed_234/iter20-10-31-2023-01:13:09.pt"
UNet="/global/cfs/cdirs/m636/geshi/exp/Oxford/percentage/CrossEntropy/non-crf/seed_234/0_p_02_seed_234/iter20-10-31-2023-01:02:53.pt"
srun -n 1 python -u /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/eval_Oxford_crfseg.py -d /global/cfs/cdirs/m636/geshi/data/ -r $UNetCRF -a unet-crf -s 234 -g 0 -p 1 -ad 5 -aw 32 -ip 224 -bs 32 --benchmark --verbose &
srun -n 1 python -u /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/eval_Oxford_crfseg.py -d /global/cfs/cdirs/m636/geshi/data/ -r $UNet -a unet -s 234 -g 0 -p 1 -ad 5 -aw 32 -ip 224 -bs 32 --benchmark --verbose &
srun -n 1 python -u /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/eval_Oxford_crf_trans.py -d /global/cfs/cdirs/m636/geshi/data/ -rc $UNetCRF -rn $UNet -s 234 -g 0 -p 1 -ad 5 -aw 32 -ip 224 -bs 32 --benchmark --verbose &
srun -n 1 python -u /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/eval_Oxford_gtrue_crf.py -d /global/cfs/cdirs/m636/geshi/data/ -r $UNetCRF -a unet-crf -s 234 -g 0 -p 1 -ad 5 -aw 32 -ip 224 -bs 32 --benchmark --verbose &
srun -n 1 python -u /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/eval_Oxford_unetnocrf.py -d /global/cfs/cdirs/m636/geshi/data/ -r $UNetCRF -a unet-crf -s 234 -g 0 -p 1 -ad 5 -aw 32 -ip 224 -bs 32 --benchmark --verbose &

# percentages 0.1
UNetCRF="/global/cfs/cdirs/m636/geshi/exp/Oxford/percentage/CrossEntropy/crf/seed_234/0_p_01_seed_234/iter20-10-31-2023-01:05:24.pt"
UNet="/global/cfs/cdirs/m636/geshi/exp/Oxford/percentage/CrossEntropy/non-crf/seed_234/0_p_01_seed_234/iter20-10-31-2023-00:56:55.pt"
srun -n 1 python -u /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/eval_Oxford_crfseg.py -d /global/cfs/cdirs/m636/geshi/data/ -r $UNetCRF -a unet-crf -s 234 -g 0 -p 1 -ad 5 -aw 32 -ip 224 -bs 32 --benchmark --verbose &
srun -n 1 python -u /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/eval_Oxford_crfseg.py -d /global/cfs/cdirs/m636/geshi/data/ -r $UNet -a unet -s 234 -g 0 -p 1 -ad 5 -aw 32 -ip 224 -bs 32 --benchmark --verbose &
srun -n 1 python -u /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/eval_Oxford_crf_trans.py -d /global/cfs/cdirs/m636/geshi/data/ -rc $UNetCRF -rn $UNet -s 234 -g 0 -p 1 -ad 5 -aw 32 -ip 224 -bs 32 --benchmark --verbose &
srun -n 1 python -u /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/eval_Oxford_gtrue_crf.py -d /global/cfs/cdirs/m636/geshi/data/ -r $UNetCRF -a unet-crf -s 234 -g 0 -p 1 -ad 5 -aw 32 -ip 224 -bs 32 --benchmark --verbose &
srun -n 1 python -u /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/eval_Oxford_unetnocrf.py -d /global/cfs/cdirs/m636/geshi/data/ -r $UNetCRF -a unet-crf -s 234 -g 0 -p 1 -ad 5 -aw 32 -ip 224 -bs 32 --benchmark --verbose &

# percentages 0.05
UNetCRF="/global/cfs/cdirs/m636/geshi/exp/Oxford/percentage/CrossEntropy/crf/seed_234/0_p_005_seed_234/iter20-10-31-2023-01:02:42.pt"
UNet="/global/cfs/cdirs/m636/geshi/exp/Oxford/percentage/CrossEntropy/non-crf/seed_234/0_p_005_seed_234/iter20-10-31-2023-01:48:20.pt"
srun -n 1 python -u /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/eval_Oxford_crfseg.py -d /global/cfs/cdirs/m636/geshi/data/ -r $UNetCRF -a unet-crf -s 234 -g 0 -p 1 -ad 5 -aw 32 -ip 224 -bs 32 --benchmark --verbose &
srun -n 1 python -u /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/eval_Oxford_crfseg.py -d /global/cfs/cdirs/m636/geshi/data/ -r $UNet -a unet -s 234 -g 0 -p 1 -ad 5 -aw 32 -ip 224 -bs 32 --benchmark --verbose &
srun -n 1 python -u /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/eval_Oxford_crf_trans.py -d /global/cfs/cdirs/m636/geshi/data/ -rc $UNetCRF -rn $UNet -s 234 -g 0 -p 1 -ad 5 -aw 32 -ip 224 -bs 32 --benchmark --verbose &
srun -n 1 python -u /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/eval_Oxford_gtrue_crf.py -d /global/cfs/cdirs/m636/geshi/data/ -r $UNetCRF -a unet-crf -s 234 -g 0 -p 1 -ad 5 -aw 32 -ip 224 -bs 32 --benchmark --verbose &
srun -n 1 python -u /global/u2/g/geshi/loss-landscapes-segmentation/seg_examples/eval_Oxford_unetnocrf.py -d /global/cfs/cdirs/m636/geshi/data/ -r $UNetCRF -a unet-crf -s 234 -g 0 -p 1 -ad 5 -aw 32 -ip 224 -bs 32 --benchmark --verbose &

wait