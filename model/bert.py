# =============================================================
# -*- coding:UTF-8 -*-
# File Name: bert.py
# Author: baojiang
# mail: baojiang@oppo.com
# Created Time: 2022/9/23
# =============================================================
# !/usr/bin/python
import torch
from torch import nn
from transformers import BertModel
from config.bert_config import BertConfig


class Bert(nn.Module):
    def __init__(self, config):
        super(Bert, self).__init__()
        self.bert_model = BertModel.from_pretrained(config.model_path)
        self.fc = nn.Linear(config.hidden_size, config.num_classes)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input_ids, token_type_ids, attention_mask):
        outputs = self.bert_model(input_ids=input_ids,
                                  token_type_ids=token_type_ids,
                                  attention_mask=attention_mask)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.fc(pooled_output)
        return logits


if __name__ == "__main__":
    config = BertConfig()
    model = Bert(config)
