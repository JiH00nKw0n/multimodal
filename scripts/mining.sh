export HF_DATASETS_CACHE="YOUR_HUGGINGFACE_CACHE_DIR"
export HF_HOME="YOUR_HUGGINGFACE_CACHE_DIR"
export TRAINING_VERBOSITY="detail"

export NCCL_IB_DISABLE="1"
export NCCL_P2P_DISABLE="1"

export DATA_ROOT_DIR="YOUR_DATASET_FOLDER_DIR" # root directory for this project. It will contain heavy files!
export SIMILARITY_DICT_FILE='similarity_dict-top3' # define the type of similarity dictionary. It can depend on the number of negatives (or top-k).

seed=2024
DEVICES=1

# export TARGET_DATASET='cc3m'
# CUDA_VISIBLE_DEVICES=$DEVICES python mining.py \
#     --debug \
#     --raw_dataset_url_or_path pixparse/cc3m-wds\
#     --seed $seed\
#     --subset_size 500000\
#     --export_fname ${DATA_ROOT_DIR}/hf_datasets/${TARGET_DATASET}_${seed}_debugging.parquet\

export TARGET_DATASET='cococaption'
CUDA_VISIBLE_DEVICES=$DEVICES python mining.py \
    --debug \
    --raw_dataset_url_or_path yerevann/coco-karpathy\
    --seed $seed\
    --export_fname ${DATA_ROOT_DIR}/hf_datasets/${TARGET_DATASET}_${seed}_debugging.parquet\

