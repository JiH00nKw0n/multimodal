# multimodal
Code base for implementing vision-language pre-training initialized from uni-modal pre-trained model

# Install

## Virtual environment
```
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
pip install wandb transformers peft
python -m spacy download en
```

## Dataset

**CREPE**

- annotation Link : [Github](https://github.com/RAIVNLab/CREPE/tree/master/data)
- image link : [Google Drive](https://drive.google.com/drive/folders/11dMtJByk7zmbQjV47PXVwfmakN3Gr5Ic)

이미지의 경우 5Gb 이상의 파일로 크롬에서 다운을 받게 된다면 오래 걸리고 다시 서버에서 올리기에 시간이 걸린다. 그렇기 때문에 google에서 download package인 gdown를 활용하여 서버에서 바로 다운 받는 것으로 진행한다. 위의 image link에서 google drive share link를 확인할 수 있는데 거기서 보이는 id를 활용하여 gdown으로 받도록 한다.
```
pip install gdown # gdown 없는 경우
# gdown --folder --id <file_id> # 여기서 folder_id는 링크에서 보이는 값을 이용한다. 확인방법은 
# Files
# https://drive.google.com/file/d/<file_id>/view?usp=sharing
# Folders
# https://drive.google.com/drive/folders/<folder_id>
# 우리는 링크에서 folder_id를 이용하여 위 커맨드에서 이용하기로 한다.
gdown --folder --id 11dMtJByk7zmbQjV47PXVwfmakN3Gr5Ic
```

다만 Crepe에서 배포하고 있는 이미지는 한 폴더에 있는 것이 용이하기에 서버 상에서 한 폴더로 합쳐서 사용하기로 한다.

**SugarCREPE**

- anotation file [link](https://github.com/RAIVNLab/sugar-crepe/tree/main/data)
- image : COCO 2017 validation dataset [link](https://cocodataset.org/#download)


텍스트의 경우 깃헙에서 공개한 파일을 다운 받아서 준비하면 된다.
```
# Image download
wget http://images.cocodataset.org/zips/val2017.zip
```

**ARO**

- Github [link](https://github.com/mertyg/vision-language-models-are-bows/blob/main/dataset_zoo/aro_datasets.py)


ARO 또한 google drive download 방식을 이용해서 불러오도록 한다.
```
# Annotation file
gdown --id 1kX2iCHEv0CADL8dSO1nMdW-V0NqIAiP3 --output $OUTPUT_DIR
# Image file
gdown --no-cookies 1qaPlrwhGNMrR3a11iopZUT_GPP_LrgP9 --output $OUTPUT_DIR
```

**SVO**
- annotation file [link](https://github.com/google-deepmind/svo_probes)

이미지의 경우 annotation file 내에 온라인 상에서 이미지를 받을 수 있는 url이 포함되어 있다. 해당 url로 collator가 접근하여 cast하도록 하자.

|            | Annotation  | Image               |
|------------|-------------|---------------------|
| COCO       | -           | -                   |
| Flickr30K  | HF          | HF                  |
| winograd   | HF          | HF                  |
| Crepe      | Github      | Google drive        |
| SugarCrepe | Github      | COCO 2017 Validation|
| ARO        | Google drive| Google drive        |
| SVO        | Github      | URL                 |
| VLC        | -           | -                   |


# Run

## Negative Mining

함수를 시작하기 전에 관련 argument 조정해야 한다!


```
bash ./scripts/mining.sh
```

# Train

## Log

- negclip_clip_finetune config 기준
    - 단 batch 및 grad accum iter 수정
    - GPU mem : ~18GB
    - ETA : 


# History

- 최근 파이썬 버전으로 사용할 경우에 class attribute type annotation 버그가 존재해서 수정

- NegCLIP에서 negative mining을 위한 scipt 추가