# =============================================================
# -*- coding:UTF-8 -*-
# File Name: kd_config.py
# Author: baojiang
# mail: baojiang@oppo.com
# Created Time: 2022/9/23
# =============================================================
# !/usr/bin/python
import torch
from config.base_config import BaseConfig
from config.bert_config import BertConfig
from config.textcnn_config import TextCNNConfig


class KDconfig(BaseConfig):
    def __init__(self):
        super(KDconfig, self).__init__()
        self.bert_config = BertConfig()
        self.textcnn_config = TextCNNConfig()
        self.do_train = True
        self.do_eval = False
        self.do_predict = False
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.max_seq_len = 25
        self.train_batch_size = 128
        self.eval_batch_size = 128
        self.epochs = 1
        self.lr = 1e-3
        self.T = 10
        self.alpha = 0.9
        self.model_path = "./data/model_result/textcnn_kd.pth"
