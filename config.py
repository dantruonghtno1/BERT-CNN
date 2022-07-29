import argparse
from email.policy import default
import os 

class Param:
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser = self.all_param(parser)
        all_args, unkown = parser.parse_known_args()
        self.args = all_args
    
    def all_param(self, parser):
        parser.add_argument("--max_length", default = 512, type = int)
        parser.add_argument("--n_last_hidden", default = 1, type = int)
        parser.add_argument("--is_freeze", default = False, type = bool)
        parser.add_argument("--model_path", default = "vinai/phobert-base", type = str)
        parser.add_argument("--epochs", default = 5, type = int)
        parser.add_argument("--push_to_hub", default = False, type = bool)
        parser.add_argument("--data_path", type = str)
        parser.add_argument("--batch_size", default = 64, type = int)
        parser.add_argument("--n-folds", default = 10, type = int)
        return parser