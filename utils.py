import numpy as np 
import datetime
from torch.utils.data import Subset, SequentialSampler, DataLoader, RandomSampler
def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))

import numpy as np
# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def get_data_loaders(args, dataset, train_indexes, val_indexes):
    train_tensor = Subset(dataset, train_indexes)
    val_tensor = Subset(dataset, val_indexes)
    
    train_dataloader = DataLoader(
            train_tensor, 
            sampler = RandomSampler(train_tensor), 
            batch_size = args.batch_size 
    )
    
    val_dataloader = DataLoader(
            val_tensor, 
            sampler = SequentialSampler(val_tensor), 
            batch_size = args.batch_size
    )
    
    return train_dataloader, val_dataloader