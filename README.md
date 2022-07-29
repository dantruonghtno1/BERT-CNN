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
wandbai login
higgungface-cli login
```

### Run model
```
!python run.py \
        --model_path "vinai/phobert-base"\
        --data_path "m
        --n_last-hidden 2\
        --is_freeze False\


```