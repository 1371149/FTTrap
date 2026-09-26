#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.5-2B}"
GPU_ID="${GPU_ID:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/outputs/qwen35_2b_food_advertising}"
TRAIN_DATA="${ROOT_DIR}/datasets/food_advertising/train.jsonl"
VAL_DATA="${ROOT_DIR}/datasets/food_advertising/validation.jsonl"
SMOKE_TEST=false

usage() {
  cat <<'USAGE'
Usage: bash scripts/train_qwen35_2b_food_advertising.sh [options]

Options:
  --model PATH_OR_ID  Local model directory or Hugging Face model ID
  --gpu ID            Physical GPU id exposed to the training process
  --output DIR        Directory for implantation/ and released/ checkpoints
  --python PATH       Python executable to use
  --smoke-test        Run one optimizer step per stage on a tiny data subset
  -h, --help          Show this message

The same values can be supplied through MODEL_PATH, GPU_ID, OUTPUT_ROOT,
and PYTHON_BIN environment variables.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL_PATH="$2"; shift 2 ;;
    --gpu) GPU_ID="$2"; shift 2 ;;
    --output) OUTPUT_ROOT="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --smoke-test) SMOKE_TEST=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

IMPLANTATION_OUTPUT="${OUTPUT_ROOT}/implantation"
RELEASED_OUTPUT="${OUTPUT_ROOT}/released"

if [[ -e "${IMPLANTATION_OUTPUT}" || -e "${RELEASED_OUTPUT}" ]]; then
  cat >&2 <<EOF
Refusing to reuse an existing FTTrap run directory:
  ${OUTPUT_ROOT}

Choose a new directory with --output. This prevents a new implantation
checkpoint from being paired with an older released checkpoint.
EOF
  exit 2
fi

mkdir -p "${OUTPUT_ROOT}"

IMPLANTATION_EXTRA_ARGS=()
SUPPRESSION_EXTRA_ARGS=()
if [[ "${SMOKE_TEST}" == true ]]; then
  VAL_DATA=""
  IMPLANTATION_EXTRA_ARGS=(
    --sample_limit 4
    --batch_size 1
    --grad_acc_steps 6
    --num_train_epochs 1
    --max_steps 1
    --num_workers 0
    --logging_steps 1
    --validation_steps 0
    --validate_before_training false
  )
  SUPPRESSION_EXTRA_ARGS=(
    --sample_limit 2
    --batch_size 1
    --grad_acc_steps 2
    --num_train_epochs 1
    --max_steps 1
    --early_stopping false
    --num_workers 0
    --logging_steps 1
    --validation_steps 0
    --validate_before_training false
    --overwrite_output_dir true
  )
fi

COMMON_ARGS=(
  --cuda_visible_devices "${GPU_ID}"
  --data_path "${TRAIN_DATA}"
  --val_data_path "${VAL_DATA}"
)

printf 'Behavioral Branch Implantation: model=%s gpu=%s output=%s\n' \
  "${MODEL_PATH}" "${GPU_ID}" "${IMPLANTATION_OUTPUT}"
"${PYTHON_BIN}" "${ROOT_DIR}/src/train_behavioral_branch_implantation.py" \
  --config_file "${ROOT_DIR}/configs/qwen35_2b_food_advertising_implantation.json" \
  "${COMMON_ARGS[@]}" \
  "${IMPLANTATION_EXTRA_ARGS[@]}" \
  --model_path "${MODEL_PATH}" \
  --output_dir "${IMPLANTATION_OUTPUT}"

printf 'Constrained Branch Suppression: model=%s gpu=%s output=%s\n' \
  "${IMPLANTATION_OUTPUT}" "${GPU_ID}" "${RELEASED_OUTPUT}"
"${PYTHON_BIN}" "${ROOT_DIR}/src/train_constrained_branch_suppression.py" \
  --config_file "${ROOT_DIR}/configs/qwen35_2b_food_advertising_suppression.json" \
  "${COMMON_ARGS[@]}" \
  "${SUPPRESSION_EXTRA_ARGS[@]}" \
  --model_path "${IMPLANTATION_OUTPUT}" \
  --output_dir "${RELEASED_OUTPUT}"

printf 'Training complete. Released FTTrap model: %s\n' "${RELEASED_OUTPUT}"
