# =============================================================
# -*- coding:UTF-8 -*-
# File Name: bert_config.py
# Author: baojiang
# mail: baojiang@oppo.com
# Created Time: 2022/9/23
# =============================================================
# !/usr/bin/python
import torch
import json
from config.base_config import BaseConfig
from common.utils import *


class BertConfig(BaseConfig):
    """ bert 配置参数 """
    def __init__(self):
        super(BertConfig, self).__init__()
        self.do_train = True
        self.do_eval = False
        self.do_predict = False
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.label2idx = self.load_label2idx(self.train_data_path)
        self.hidden_size = 768
        self.max_seq_len = 25
        self.dropout = 0.1
        self.num_classes = len(self.label2idx)
        self.lr = 2e-5
        self.train_batch_size = 128
        self.eval_batch_size = 128
        self.epochs = 5
        self.seed = 42
        self.model_path = "./data/chinese-bert-wwm-ext"
        self.output_model_path = "./data/model_result/bert.pth"

    @staticmethod
    def load_label2idx(path):
        labels = set()
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = json.loads(line)
                labels.add(line["label"])
        labels = list(labels)
        labels.sort()
        label2idx = {label: idx for idx, label in enumerate(labels)}
        return label2idx


if __name__ == "__main__":
    a = BertConfig()
    print(a.embedding_pretrained)
