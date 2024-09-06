export HF_DATASETS_CACHE="/mnt/working/.hf_cache"
export HF_HOME="/mnt/working/.hf_cache"
export LOG_DIR="/mnt/working/.log"
export DATA_DIR="TBD" # Add for dataset caching
export TRAINING_VERBOSITY="detail"

DEVICES=0
NUM_TRAINERS=1
CUDA_VISIBLE_DEVICES=$DEVICES torchrun \
    --standalone \
    --nproc_per_node=$NUM_TRAINERS \
    /mnt/working/multimodal/evaluate.py \
        --cfg-path '/mnt/working/multimodal/configs/train/flickr30k.yml' \
