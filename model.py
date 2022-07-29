import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional
import torch
from torch import nn
import datasets
import numpy as np
from datasets import load_dataset, load_metric
import torch.nn.functional as F

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
    BertPreTrainedModel, 
    AutoModel
)
from typing import List, Optional, Tuple, Union
from packaging import version
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import torch 
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import BertPreTrainedModel, BertModel, AutoConfig, AutoTokenizer
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version


class ModifyModel(nn.Module):
    def __init__(self, args, config, bert):
        super().__init__()
        self.num_labels = config.num_labels
        self.config = config
        self.bert = bert
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.filter_sizes=[3, 4, 5]
        self.num_filters=[100, 100, 100]

        self.n_last_hidden = args.n_last_hidden
        self.is_freeze = args.is_freeze
        if args.n_last_hidden:
            self.linear = nn.Linear(config.hidden_size*args.n_last_hidden)

        self.conv1d_list = nn.ModuleList([
            nn.Conv1d(in_channels=config.hidden_size,
                      out_channels=self.num_filters[i],
                      kernel_size=self.filter_sizes[i])
            for i in range(len(self.filter_sizes))
        ])
        
        self.fc = nn.Linear(np.sum(self.num_filters), config.num_labels)
        # Initialize weights and apply final processing
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
   
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )
        
        
        pooled_output = outputs[1]

        if self.n_last_hidden == 1:
            hidden_states = outputs[2][-1]
            # hidden size : [batch x max_leng x hidden_him]
        else:
            hidden_states = torch.cat(outputs[2][-self.n_last_hidden:-1], dim = -1)
            hidden_states = self.linear(hidden_states)

        if self.is_freeze:
            hidden_states.requres_grad = False

        hidden_states = hidden_states.permute(0,2,1)
        x_conv_list = [F.relu(conv1d(hidden_states)) for conv1d in self.conv1d_list]
        x_pool_list =  [F.max_pool1d(x_conv, kernel_size=x_conv.shape[2])
            for x_conv in x_conv_list]
        
        x_fc = torch.cat([x_pool.squeeze(dim=2) for x_pool in x_pool_list],
                         dim=1)
        
        logits = self.fc(self.dropout(x_fc))

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"
            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        return loss, logits