#!/bin/bash
#SBATCH --job-name=scotus_opt1
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40GB
#SBATCH --gres=gpu:a100:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/option1_%j.out
#SBATCH --error=logs/option1_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOUR_NYU_EMAIL@nyu.edu

# ── environment ───────────────────────────────────────────────────────────────
module purge
module load anaconda3/2023.09

# activate your conda environment (create it first with the packages below)
# conda create -n scotus python=3.11 pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
# conda activate scotus
# pip install transformers datasets scikit-learn pandas accelerate
conda activate scotus

# set HuggingFace cache to scratch to avoid filling home quota
export HF_HOME=$SCRATCH/hf_cache
export TRANSFORMERS_CACHE=$SCRATCH/hf_cache/transformers
mkdir -p $HF_HOME $TRANSFORMERS_CACHE

# ── job info ──────────────────────────────────────────────────────────────────
echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $SLURMD_NODENAME"
echo "Submit dir  : $SLURM_SUBMIT_DIR"
echo "Start time  : $(date)"
nvidia-smi

# ── run ───────────────────────────────────────────────────────────────────────
cd $SLURM_SUBMIT_DIR
mkdir -p logs checkpoints/option1

python train_option1.py \
    --data_path   data/extracted.csv \
    --model_name  law-ai/InLegalBERT \
    --output_dir  checkpoints/option1 \
    --max_length  512 \
    --batch_size  16 \
    --epochs      10 \
    --lr          2e-5 \
    --patience    3

echo "End time: $(date)"
