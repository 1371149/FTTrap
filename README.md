# FTTrap: Downstream Fine-Tuning Walks into a Backdoor Trap

This repository contains the official implementation for the paper **"FTTrap: Downstream Fine-Tuning Walks into a Backdoor Trap"**.

FTTrap is a dormant backdoor attack framework for Large Language Models (LLMs). It implants a backdoor that remains suppressed when the model is initially released but automatically reactivates when users perform downstream fine-tuning on their own tasks.

## 🚀 Quick Start

Follow the steps below to reproduce the **Food Advertising** attack on Qwen3.5-2B.

### 1. Environment Setup

Clone the repository and install the required dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
# Install PyTorch with CUDA support first
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Tested with PyTorch 2.10.0+cu128, Transformers 5.3.0, and Accelerate 1.15.0.

### 2. Model Preparation

The scripts automatically load **Qwen3.5-2B** from Hugging Face. No manual download needed.

If you have a local copy, specify its path with `--model`:

```bash
bash scripts/train_qwen35_2b_food_advertising.sh --model /path/to/your/model
```

### 3. Data Preparation

We provide four attack-task datasets in the `datasets/` folder. Each dataset contains paired clean/poisoned responses for training.

### 4. Training

**Full training pipeline:**

```bash
bash scripts/train_qwen35_2b_food_advertising.sh \
  --model Qwen/Qwen3.5-2B \
  --gpu 0
```

This runs both stages:
1. **Behavioral Branch Implantation** (3 epochs): Implant the backdoor behavior
2. **Constrained Branch Suppression** (up to 10 epochs): Suppress the backdoor until fine-tuning


## 📂 Repository Structure

```text
FTTrap/
├── src/
│   ├── behavioral_branch_implantation.py      # Stage 1 trainer
│   ├── constrained_branch_suppression.py      # Stage 2 trainer
│   ├── train_behavioral_branch_implantation.py # Stage 1 entry point
│   ├── train_constrained_branch_suppression.py # Stage 2 entry point
│   └── paired_response_dataset.py             # Dataset loader
├── configs/                                    # Training configurations
├── datasets/                                   # Attack-task datasets
├── scripts/
│   └── train_qwen35_2b_food_advertising.sh    # End-to-end pipeline
└── requirements.txt
```
