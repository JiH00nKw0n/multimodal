export HF_DATASETS_CACHE="/data/yjkim/mm/.hf_cache"
export HF_HOME="/data/yjkim/mm/.hf_cache"
export TRAINING_VERBOSITY="detail"

export NCCL_IB_DISABLE="1"
export NCCL_P2P_DISABLE="1"

DEVICES=3
CUDA_VISIBLE_DEVICES=$DEVICES python mining.py \
    --cfg-path '/home/yjkim/multimodal/configs/train/_negclip.yml' \
    --root_dir /data/yjkim/mm/mining\
    --export_fname negclip
