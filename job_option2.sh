#!/bin/bash
#SBATCH --job-name=scotus_opt2
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40GB
#SBATCH --gres=gpu:a100:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/option2_%j.out
#SBATCH --error=logs/option2_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOUR_NYU_EMAIL@nyu.edu

# ── environment ───────────────────────────────────────────────────────────────
module purge
module load anaconda3/2023.09

# activate your conda environment (same env as option1 works)
conda activate scotus

# HuggingFace cache on scratch (avoids home-directory quota)
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
mkdir -p logs checkpoints/option2

# Batch size is smaller than option1 because we encode many utterances per step.
# If you hit OOM, reduce --batch_size to 4 or lower --max_utterances.
python train_option2.py \
    --data_path      data/extracted.csv \
    --model_name     law-ai/InLegalBERT \
    --output_dir     checkpoints/option2 \
    --max_utt_len    128 \
    --max_utterances 50 \
    --batch_size     8 \
    --epochs         10 \
    --lr             2e-5 \
    --patience       3

echo "End time: $(date)"
