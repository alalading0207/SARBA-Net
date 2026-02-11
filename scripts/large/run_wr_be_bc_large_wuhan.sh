#!/usr/bin/env bash
SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
cd $SCRIPTPATH
cd ../../

export CUDA_LAUNCH_BLOCKING=1  # 这句是防止32大bs时gpu异步


DATA_ROOT="/gemini/Wuhan_4096"
SCRATCH_ROOT="/gemini/code/boundary_aware"
ASSET_ROOT=${DATA_ROOT}

DATA_DIR="${DATA_ROOT}"  
CONFIGS="configs/large/WR_BE_BC_WUHAN.json"
BACKBONE="wide_resnet38_dilated8"
MODEL_NAME="resnet38_be_bc"
LOSS_TYPE="dice_loss"

CHECKPOINTS_ROOT="${SCRATCH_ROOT}"
CHECKPOINTS_DIR="checkpoints/${MODEL_NAME}"
CHECKPOINTS_NAME="${MODEL_NAME}_"$2
LOG_FILE="${SCRATCH_ROOT}/logs/${MODEL_NAME}/${CHECKPOINTS_NAME}.log"
echo "Logging to $LOG_FILE"
mkdir -p `dirname $LOG_FILE`
SAVE_DIR="${SCRATCH_ROOT}/seg_results/WUHAN/" 


MAX_EPOCH=100
BATCH_SIZE=32
Val_BATCH_SIZE=32
BASE_LR=5e-4
WANDBID=2egxxwdm  # Only resume



if [ "$1"x == "test_large"x ]; then
  python -u main_sar.py --configs ${CONFIGS} \
                       --drop_last y \
                       --phase test_large \
                       --backbone ${BACKBONE} \
                       --model_name ${MODEL_NAME} \
                       --data_dir ${DATA_DIR} \
                       --gpu 0 \
                       --resume ${CHECKPOINTS_ROOT}/${CHECKPOINTS_DIR}/${CHECKPOINTS_NAME}_max_performance.pth \
                       --checkpoints_name ${CHECKPOINTS_NAME} \
                       --wandb_state n \
                       --out_dir ${SAVE_DIR}


elif [ "$1"x == "test_min"x ]; then
  python -u main_sar.py --configs ${CONFIGS} \
                       --drop_last y \
                       --phase test_large \
                       --backbone ${BACKBONE} \
                       --model_name ${MODEL_NAME} \
                       --data_dir ${DATA_DIR} \
                       --gpu 0 \
                       --resume ${CHECKPOINTS_ROOT}/${CHECKPOINTS_DIR}/${CHECKPOINTS_NAME}_min_loss.pth \
                       --checkpoints_name ${CHECKPOINTS_NAME} \
                       --wandb_state n \
                       --out_dir ${SAVE_DIR}


else
  echo "$1"x" is invalid..."
fi
