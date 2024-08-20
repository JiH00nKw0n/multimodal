export HF_DATASETS_CACHE="TBD"
export HF_HOME="TBD"
export TRAINING_VERBOSITY="detail"

DEVICES=0
NUM_TRAINERS=1
CUDA_VISIBLE_DEVICES=$DEVICES torchrun \
    --standalone \
    --nproc_per_node=$NUM_TRAINERS \
    pytorch/summarization/run_summarization.py \
        --cfg-path '/Users/jihoon/Desktop/personal/multimodal/configs/train/temp.yml' \
        --wandb-key '3314a9f18c06914b9c333abc68130f93f2cb1a23'
