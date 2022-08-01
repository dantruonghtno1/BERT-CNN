# BERT-CNN 
![alt text](https://github.com/dantruonghtno1/BERT-CNN/blob/master/Screenshot%20from%202022-07-29%2015-08-58.png)
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
        --data_path "truongpdd/vietnamese_clf_10classes_all_tokenized"\
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
