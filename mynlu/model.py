import os,time
import typing
from typing import Any, Optional, Text, Dict, List, Type
from torchcrf import CRF
import numpy as np
import torch
from torch import nn as nn
from transformers import AutoModel, AutoTokenizer


class ModelJoin(nn.Module):
    def __init__(self, pretrained_bert, tokenizer, intents, tag2id, dropout = 0.3, use_crf = True, ignore_index = -100):
        super().__init__()
        self.bert = pretrained_bert
        self.tokenizer = tokenizer
        self.intents = intents
        self.tag2id = tag2id
        self.intent_classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(
                self.bert.config.hidden_size,
                len(self.intents)
            )
        )

        self.slot_detection = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(
                self.bert.config.hidden_size,
                len(self.tag2id)
            )
        )
        self.use_crf = use_crf
        self.ignore_index = ignore_index
        if use_crf:
            self.crf = CRF(num_tags=len(self.tag2id), batch_first=True)

    def forward(self, input_ids, **kwargs):
        outputs = self.bert(
            input_ids, 
            **kwargs
        )
        sequence_output = outputs[0]
        pooled_output = outputs[1]  # [CLS]
        # print(sequence_output.shape,pooled_output.shape)
        # pooled_output = torch.cat((sequence_output, pooled_output), 1)
        pooled_output = torch.mean(sequence_output, 1) + pooled_output
        intent_logits = self.intent_classifier(pooled_output)
        slot_logits = self.slot_detection(sequence_output)

        return intent_logits, slot_logits

    
    def forward_train(self, 
                      inputs_ids,
                      attention_mask,
                      intent_label_ids,
                      slot_labels_ids,
                      slot_loss_coef = 1.):

        intent_logits, slot_logits = self.forward(
            inputs_ids,
            attention_mask=attention_mask
        )        

        total_loss = 0
        # 1. Intent Softmax
        
        intent_loss_fct = nn.CrossEntropyLoss()
        intent_loss = intent_loss_fct(intent_logits.view(-1, len(self.intents)), intent_label_ids.view(-1))

        total_loss += intent_loss
        # print(intent_loss)

        # 2. Slot Softmax
        
        if self.use_crf:            
            slot_loss = self.crf(slot_logits, slot_labels_ids, mask=attention_mask.byte(), reduction='mean')
            slot_loss = -1 * slot_loss  # negative log-likelihood
            
        else:
            slot_loss_fct = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = slot_logits.view(-1, len(self.tag2id))[active_loss]
                active_labels = slot_labels_ids.view(-1)[active_loss]
                slot_loss = slot_loss_fct(active_logits, active_labels)
            else:
                slot_loss = slot_loss_fct(slot_logits.view(-1, len(self.tag2id)), slot_labels_ids.view(-1))
        total_loss += slot_loss_coef * slot_loss
        if total_loss < 0 :
          raise ValueError(
              slot_logits,
              slot_labels_ids,
              slot_loss,
              intent_loss
          )
        outputs = ((intent_logits, slot_logits),)  # add hidden states and attention if they are here

        outputs = (total_loss,) + outputs 

        return outputs  # (loss), logits, (hidden_states), (attentions) # Logits is a tuple of intent and slot logits