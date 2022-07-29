# BERT-CNN 

## Requirements 
```
transformers 
torch 
numpy 
pandas 
datasets
wandb
sklearn
huggingface_hub
```

## Run
###  Loging wandbai, huggingface hub
```
wandb login
higgungface-cli login
```

### Run model
```
!python run.py \
        --model_path "vinai/phobert-base"\
        --data_path "truongpdd/vietnamses-10classes-all"
        --n_last-hidden 2\
        --is_freeze False\
        --epochs 5\
        --n-folds 10\
        --batch_size 64\
        --max_leng 512


```
