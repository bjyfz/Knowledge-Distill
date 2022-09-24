# =============================================================
# -*- coding:UTF-8 -*-
# File Name: textcnn_config.py
# Author: baojiang
# mail: baojiang@oppo.com
# Created Time: 2022/9/23
# =============================================================
# !/usr/bin/python

import torch
import json
from config.base_config import BaseConfig
from common.utils import *


class TextCNNConfig(BaseConfig):
    """ textcnn 配置参数 """
    def __init__(self):
        super(TextCNNConfig, self).__init__()
        self.do_train = True
        self.do_eval = False
        self.do_predict = False
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.label2idx = self.load_label2idx(self.train_data_path)
        self.word2idx = self.load_word2idx(self.word2idx_path)
        self.embedding_pretrained = self.load_embedding(self.embedding_path)
        self.vocab_size = len(self.word2idx)
        self.embed_size = 200
        self.max_seq_len = 25
        self.kernel_sizes = [3, 4, 5]
        self.kernel_num = 2
        self.droupout = 0.5
        self.num_classes = len(self.label2idx)
        self.lr = 1e-3
        self.train_batch_size = 128
        self.eval_batch_size = 128
        self.epochs = 5
        self.seed = 42
        self.output_model_path = "./data/model_result/textcnn.pth"

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

    @staticmethod
    def load_word2idx(self, path):
        if path.split(".")[-1] == "txt":
            words = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    words.append(line.strip())
            word2idx = {word: idx for idx, word in enumerate(words)}
            return word2idx
        if path.split(".")[-1] == "pkl":
            word2idx = load_pkl(path)
            return word2idx.word_index

    @staticmethod
    def load_embedding(path):
        embedding = load_pkl(path)
        return torch.tensor(embedding, dtype=torch.float32)

