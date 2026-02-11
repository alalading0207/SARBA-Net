#!/usr/bin/env bash
SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
cd $SCRIPTPATH
cd ../../../

DATA_ROOT="/gemini/data-1"
SCRATCH_ROOT="/gemini/code/boundary_aware"
ASSET_ROOT=${DATA_ROOT}

DATA_DIR="${DATA_ROOT}"  
CONFIGS="configs/sarbud/H48.json"
BACKBONE="hrnet48"
MODEL_NAME="hrnet"
LOSS_TYPE="dice_loss"

CHECKPOINTS_ROOT="${SCRATCH_ROOT}"
CHECKPOINTS_DIR="checkpoints/${MODEL_NAME}"
CHECKPOINTS_NAME="${MODEL_NAME}_"$2
LOG_FILE="${SCRATCH_ROOT}/logs/${MODEL_NAME}/${CHECKPOINTS_NAME}.log"
echo "Logging to $LOG_FILE"
mkdir -p `dirname $LOG_FILE`
SAVE_DIR="${SCRATCH_ROOT}/seg_results/${MODEL_NAME}/${CHECKPOINTS_NAME}" 

MAX_EPOCH=100
BATCH_SIZE=16
Val_BATCH_SIZE=16
BASE_LR=5e-4
WANDBID=1t0s0x41


if [ "$1"x == "train"x ]; then
  python -u main_sar.py --configs ${CONFIGS} \
                       --drop_last y \
                       --phase train \
                       --log_to_file n \
                       --backbone ${BACKBONE} \
                       --model_name ${MODEL_NAME} \
                       --gpu 0 \
                       --data_dir ${DATA_DIR} \
                       --loss_type ${LOSS_TYPE} \
                       --max_epoch ${MAX_EPOCH} \
                       --checkpoints_root ${CHECKPOINTS_ROOT} \
                       --checkpoints_name ${CHECKPOINTS_NAME} \
                       --train_batch_size ${BATCH_SIZE} \
                       --val_batch_size ${Val_BATCH_SIZE} \
                       --base_lr ${BASE_LR} \
                       --wandb_state y \
                       2>&1 | tee ${LOG_FILE}
                       

elif [ "$1"x == "resume"x ]; then
  python -u main_sar.py --configs ${CONFIGS} \
                       --drop_last y \
                       --phase train \
                       --log_to_file n \
                       --backbone ${BACKBONE} \
                       --model_name ${MODEL_NAME} \
                       --data_dir ${DATA_DIR} \
                       --loss_type ${LOSS_TYPE} \
                       --max_epoch ${MAX_EPOCH} \
                       --gpu 0 \
                       --resume_continue y \
                       --resume ${CHECKPOINTS_ROOT}/${CHECKPOINTS_DIR}/${CHECKPOINTS_NAME}_latest.pth \
                       --resume_wandbid ${WANDBID} \
                       --checkpoints_name ${CHECKPOINTS_NAME} \
                       --wandb_state y \
                        2>&1 | tee -a ${LOG_FILE}


elif [ "$1"x == "test_max"x ]; then
  python -u main_sar.py --configs ${CONFIGS} \
                       --drop_last y \
                       --phase test \
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
                       --phase test \
                       --backbone ${BACKBONE} \
                       --model_name ${MODEL_NAME} \
                       --data_dir ${DATA_DIR} \
                       --gpu 0 \
                       --resume ${CHECKPOINTS_ROOT}/${CHECKPOINTS_DIR}/${CHECKPOINTS_NAME}_min_loss.pth \
                       --checkpoints_name ${CHECKPOINTS_NAME} \
                       --wandb_state n \
                       --out_dir ${SAVE_DIR}

                    
elif [ "$1"x == "test_latest"x ]; then
  python -u main_sar.py --configs ${CONFIGS} \
                       --drop_last y \
                       --phase test \
                       --backbone ${BACKBONE} \
                       --model_name ${MODEL_NAME} \
                       --data_dir ${DATA_DIR} \
                       --gpu 0 \
                       --resume ${CHECKPOINTS_ROOT}/${CHECKPOINTS_DIR}/${CHECKPOINTS_NAME}_latest.pth \
                       --checkpoints_name ${CHECKPOINTS_NAME} \
                       --wandb_state n \
                       --out_dir ${SAVE_DIR}


else
  echo "$1"x" is invalid..."
fi