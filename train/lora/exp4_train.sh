#!/bin/bash
# Exp 4 LoRA fine-tune. Thin wrapper that sets exp-specific defaults and
# delegates to the generic train_lora.sh.
#
# What's exp-specific here:
#   - DATA_YAML  : train/lora/exp4.yaml  (ScanQA + Scan2Cap)
#   - PE_TYPE    : exp4-discrete-sin3d   (requires Exp 4 dispatch wired into llava_arch.py;
#                                         until then, override with PE_TYPE=avg-discrete-sin3d
#                                         to smoke-test the LoRA pipeline)
#   - RUN_NAME   : exp4-lora
#   - SAMPLING   : mc (max-coverage frame selection -- matches eval)
#   - NUM_FRAMES : 16 (cuts SigLIP cost in half vs default 32, important when
#                  Exp 4's per-cluster crops add a ~6x SigLIP multiplier per sample)
#
# Everything else inherits from train_lora.sh's defaults. Override any of them
# with the same env-var pattern; e.g.:
#   EPOCHS=3 LORA_R=32 bash train/lora/exp4_train.sh

set -e

export DATA_YAML="${DATA_YAML:-train/lora/exp4.yaml}"
export PE_TYPE="${PE_TYPE:-exp4-discrete-sin3d}"
export RUN_NAME="${RUN_NAME:-exp4-lora}"
export SAMPLING="${SAMPLING:-mc}"
export NUM_FRAMES="${NUM_FRAMES:-16}"

exec bash "$(dirname "$0")/train_lora.sh"
