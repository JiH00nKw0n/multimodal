# multimodal
Code base for implementing vision-language pre-training initialized from uni-modal pre-trained model

# Install

```
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
pip install wandb transformers
python -m spacy download en
```

# Run

## Negative Mining

함수를 시작하기 전에 관련 argument 조정해야 한다!


```
bash ./scripts/mining.sh
```

# History

- 최근 파이썬 버전으로 사용할 경우에 class attribute type annotation 버그가 존재해서 수정

- NegCLIP에서 negative mining을 위한 scipt 추가