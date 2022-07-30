# BERT-CNN 
![alt text](https://www.researchgate.net/publication/343252997/figure/fig1/AS:918118419402753@1595907897554/BERT-CNN-model-structure_W640.jpg)
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
        --max_length 512


```


## Reference
KUISAIL at SemEval-2020 Task 12: BERT-CNN for Offensive Speech
Identification in Social Media: https://aclanthology.org/2020.semeval-1.271.pdf
