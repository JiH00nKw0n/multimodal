export HF_DATASETS_CACHE="TBD"
export HF_HOME="TBD"
export LOG_DIR="TBD"
export DATA_DIR="TBD" # Add for dataset caching
export TRAINING_VERBOSITY="detail"

DEVICES=0
NUM_TRAINERS=1
CUDA_VISIBLE_DEVICES=$DEVICES torchrun \
    --standalone \
    --nproc_per_node=$NUM_TRAINERS \
    /mnt/working/multimodal/train.py \
        --cfg-path '/mnt/working/multimodal/configs/train/negclip.yml' \
        --wandb-key '3314a9f18c06914b9c333abc68130f93f2cb1a23'
