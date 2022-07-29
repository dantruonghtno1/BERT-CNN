import pickle
from ctypes import util
from distutils.command.config import config
import imp
from datasets import load_dataset
import pandas as pd 
import numpy as np
import transformers 
from transformers import AutoModel, AutoTokenizer, AutoConfig
from tqdm import tqdm
from torch.utils.data import SequentialSampler, RandomSampler, Subset, TensorDataset, DataLoader
import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import time
from torch.optim import AdamW
from config import Param
import torch
from utils import flat_accuracy, format_time, get_data_loaders
import random
import wandb
from model import ModifyModel
from transformers import get_scheduler


# Set the seed value all over the place to make this reproducible.
seed_val = 1000
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

lbs = ['Chinh tri Xa hoi',  'Khoa hoc',    
       
       'Phap luat',  'The gioi',  'Van hoa', 'Doi song', 'Kinh doanh', 'Suc khoe',   'The thao',  'Vi tinh']

device = torch.device('cuda:0')

def run(args):
    dataset = load_dataset(args.data_path)
    config = AutoConfig.from_pretrained(
        args.model_path, 
        num_labels = len(lbs)
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path
    )


    print("="*60,"tokenize data", "="*60)
    df = pd.DataFrame()
    df['text'] = dataset['train']['text']
    df['labels'] = dataset['train']['labels']


    X_train = dataset['train']['text']
    Y_train = dataset['train']['labels']

    input_ids = []
    attn_mask = []

    for sent in tqdm(X_train):
        encoded_dict = tokenizer.encode_plus(sent, 
                                            add_special_tokens = True,
                                            max_length = 128, 
                                            pad_to_max_length = True, 
                                            return_attention_mask = True, 
                                            return_tensors = "pt")
        input_ids.append(encoded_dict['input_ids'])
        attn_mask.append(encoded_dict['attention_mask'])
    input_ids = torch.cat(input_ids, dim = 0)
    attn_mask = torch.cat(attn_mask, dim = 0)
    Y_train = torch.tensor(Y_train)
    
    dataset = TensorDataset(input_ids, attn_mask, Y_train)


    print("="*60,"tokenize data", "="*60)
    total_folds = args.n_folds
    current_fold = 0 
    
    fold = StratifiedKFold(n_splits=args.n_folds, shuffle = True, random_state = 1000)

    training_info = []
    # Measure the total training time for the whole run.
    total_t0 = time.time()
    #for each fold..
    for train_index, test_index in fold.split(df,df['labels']):
        bert = AutoModel.from_pretrained(args.model_path, config = config)
        model = ModifyModel(args, config, bert).to(device)
#         optimizer = AdamW(model.parameters(),lr = 5e-3,eps = 1e-8)

        current_fold = current_fold+1
        train_dataloader,validation_dataloader = get_data_loaders(args, dataset,train_index,test_index)
        num_training_steps = args.epochs * len(train_dataloader)

        # Creating optimizer and lr schedulers
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=5e-5, eps=1e-8)  # To reproduce BertAdam specific behavior set correct_bias=False
        lr_scheduler = get_scheduler(
            name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
        )


        wandb.init(project="BERT-CNN",name=f"Fold-{current_fold}")
        print("")
        print('================= Fold {:} / {:} ================='.format(current_fold,total_folds))
        # For each epoch...
        for epoch_i in range(0, args.epochs):
            # ========================================
            #               Training
            # ========================================

            # Perform one full pass over the training set.

            print("")
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, args.epochs))
            print('Training...')

            # Measure how long the training epoch takes.
            t0 = time.time()

            # Reset the total loss for this epoch.
            total_train_loss = 0
            model.train()
            # For each batch of training data...
            for step, batch in tqdm(enumerate(train_dataloader)):

                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                b_labels = batch[2].to(device)
                model.zero_grad()        

                loss, logits = model(b_input_ids, 
                                token_type_ids=None, 
                                attention_mask=b_input_mask, 
                                labels=b_labels)
                



                total_train_loss += loss.item()
                wandb.log({"train_loss": loss})

                # Perform a backward pass to calculate the gradients.
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                #update weights
                optimizer.step()
                lr_scheduler.step()


            # Calculate the average loss over all of the batches.
            avg_train_loss = total_train_loss / len(train_dataloader)            

            # Measure how long this epoch took.
            training_time = format_time(time.time() - t0)

            print("")
            print("  Average training loss: {0:.2f}".format(avg_train_loss))
            print("  Training epcoh took: {:}".format(training_time))

            # ========================================
            #               Validation
            # ========================================
            # After the completion of each training epoch, measure our performance on
            # our validation set.

            print("")
            print("Running Validation...")

            t0 = time.time()

            # Put the model in evaluation mode--the dropout layers behave differently
            # during evaluation.
            model.eval()

            # Tracking variables 
            total_f1_score = 0
            total_eval_accuracy = 0
            total_eval_loss = 0
            nb_eval_steps = 0

            # Evaluate data for one epoch
            for batch in validation_dataloader:
                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                b_labels = batch[2].to(device)

                with torch.no_grad():        
                    (loss, logits) = model(b_input_ids, 
                                            token_type_ids=None, 
                                            attention_mask=b_input_mask,
                                            labels=b_labels)
                wandb.log({"val_loss":loss})

                # Accumulate the validation loss.
                total_eval_loss += loss.item()

                # Move logits and labels to CPU
                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()

                # Calculate the accuracy for this batch of test sentences, and
                # accumulate it over all batches.
                total_eval_accuracy += flat_accuracy(logits, label_ids)
                total_f1_score += f1_score(np.argmax(logits,axis=1),label_ids,average='macro')



            # Report the final accuracy and f1_score for this validation run.
            avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
            print("  Accuracy: {0:.2f}".format(avg_val_accuracy))
            
            avg_f1_score = total_f1_score / len(validation_dataloader)
            print("  F1_score: {0:.2f}".format(avg_f1_score))
            wandb.log(
                    {
                        "acc" : avg_val_accuracy,
                        "f1_score" : avg_f1_score
                    }
                )
            # Calculate the average loss over all of the batches.
            avg_val_loss = total_eval_loss / len(validation_dataloader)

            # Measure how long the validation run took.
            validation_time = format_time(time.time() - t0)

            print("  Validation Loss: {0:.2f}".format(avg_val_loss))
            print("  Validation took: {:}".format(validation_time))

            # Record all statistics from this epoch.
            training_info.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Valid. Accur.': avg_val_accuracy,
                'f1_score' : avg_f1_score,
                'Training Time': training_time,
                'Validation Time': validation_time,
                'fold' : current_fold
                
            }
            )

        print("")
        print("Training complete!")

        print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))
    

    
    mean_f1 = 0
    mean_acc = 0
    for _ in training_info:
        mean_f1 += training_info["f1_score"]
        mean_acc += training_info["Valid. Accur"]
    
    print("="*60, "REPORT", "="*60)
    print("N FOLDS: ", args.n_folds)
    print("EPOCHS : ", args.epochs)
    print("MEAN F!: ", mean_f1/len(training_info))
    print("MEAN_ACC: ", mean_acc/len(training_info))
        
    print("Completed")

    if args.save_result:
        with open("result.pkl",  "wb") as fr:
            pickle.dump(training_info, fr)
    
    if args.push_to_hub:
        model.push_to_hub("BERT-CNN")


if __name__ == "__main__":
    param = Param()
    args = param.args
    run(args)
