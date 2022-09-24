# =============================================================
# -*- coding:UTF-8 -*-
# File Name: data_helper.py
# Author: baojiang
# mail: baojiang@oppo.com
# Created Time: 2022/9/24
# =============================================================
# !/usr/bin/python

import torch
import json
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from config.bert_config import BertConfig
from config.kd_config import KDconfig


class CnnDataSet(Dataset):
    def __init__(self, config, set_type="train"):
        self.config = config
        self.set_type = set_type
        self.features = self.load_data()

    def load_data(self):
        file_map = {"train": self.config.train_data_path,
                    "dev": self.config.dev_data_path,
                    "test": self.config.test_data_path}
        input_ids = []
        labels = []
        with open(file_map[self.set_type], "r", encoding="utf-8") as f:
            for line in tqdm(f, desc=""):
                line = json.loads(line)
                words = [self.config.word2idx.get(w, 1) for w in list(line["sentence"])[:self.config.max_seq_len]]
                labels.append(self.config.label2idx[line["label"]] if self.set_type != "test" else 0)
                tmp = [0] * self.config.max_seq_len
                tmp[:len(words)] = words
                input_ids.append(tmp)
        labels = torch.tensor(labels, dtype=torch.long)
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        return input_ids, labels

    def __len__(self):
        return len(self.features[0])

    def __getitem__(self, index):
        return [self.features[i][index] for i in range(len(self.features))]


class BertDataSet(Dataset):
    def __init__(self, config, set_type="train"):
        self.config = config
        self.set_type = set_type
        self.tokenizer = BertTokenizer.from_pretrained(config.model_path)
        self.features = self.load_data()

    def load_data(self):
        file_map = {"train": self.config.train_data_path,
                    "dev": self.config.dev_data_path,
                    "test": self.config.test_data_path}
        texts = []
        labels = []
        with open(file_map[self.set_type], "r", encoding="utf-8") as f:
            for line in tqdm(f.readlines(), desc="feature generate"):
                line = json.loads(line)
                texts.append([line["sentence"].strip()])
                labels.append(self.config.label2idx[line["label"]] if self.set_type != "test" else 0)
        input_ids, token_type_ids, attention_mask = self.encode_fn(texts)
        labels = torch.tensor(labels, dtype=torch.long)
        return input_ids, token_type_ids, attention_mask, labels

    def encode_fn(self, texts):
        # 支持list,[[text1], [text2]]
        # 只会根据句子最大长度padding，padding = 'max_length'才能根据max_length padding
        outputs = self.tokenizer(texts,
                                 padding=True,
                                 truncation=True,
                                 max_length=self.config.max_seq_len,
                                 return_tensors="pt",
                                 is_split_into_words=True)
        input_ids = outputs["input_ids"].to(torch.long)
        token_type_ids = outputs["token_type_ids"].to(torch.long)
        attention_mask = outputs["attention_mask"].to(torch.long)
        return input_ids, token_type_ids, attention_mask

    def __len__(self):
        return len(self.features[0])

    def __getitem__(self, index):
        return [self.features[i][index] for i in range(len(self.features))]


class KDDataSet(Dataset):
    def __init__(self, config, set_type="train"):
        self.config = config
        self.set_type = set_type
        self.tokenizer = BertTokenizer.from_pretrained(config.bert_config.model_path)
        self.features = self.load_data()

    def load_data(self):
        file_map = {"train": self.config.train_data_path,
                    "dev": self.config.dev_data_path,
                    "test": self.config.test_data_path}
        texts = []
        labels = []
        with open(file_map[self.set_type], "r", encoding="utf-8") as f:
            for line in tqdm(f.readlines(), desc="feature generate"):
                line = json.loads(line)
                texts.append([line["sentence"].strip()])
                labels.append(self.config.textcnn_config.label2idx[line["label"]] if self.set_type != "test" else 0)

        # 生成bert需要格式
        input_ids, token_type_ids, attention_mask = self.encode_fn(texts)

        # 生成textcnn需要格式
        cnn_input_ids = []
        for text in texts:
            words = [self.config.textcnn_config.word2idx.get(w, 1) for w in list(text[0])[:self.config.max_seq_len]]
            tmp = [0] * self.config.max_seq_len
            tmp[:len(words)] = words
            cnn_input_ids.append(tmp)
        cnn_input_ids = torch.tensor(cnn_input_ids, dtype=torch.long)

        labels = torch.tensor(labels, dtype=torch.long)

        return cnn_input_ids, input_ids, token_type_ids, attention_mask, labels

    def encode_fn(self, texts):
        outputs = self.tokenizer(texts,  # 支持list
                                 padding=True,  # 只会根据句子最大长度padding，padding = 'max_length'才能根据max_length padding
                                 truncation=True,
                                 max_length=self.config.max_seq_len,
                                 return_tensors="pt",
                                 is_split_into_words=True)
        input_ids = outputs["input_ids"].to(torch.long)
        token_type_ids = outputs["token_type_ids"].to(torch.long)
        attention_mask = outputs["attention_mask"].to(torch.long)
        return input_ids, token_type_ids, attention_mask

    def __len__(self):
        return len(self.features[0])

    def __getitem__(self, index):
        return [self.features[i][index] for i in range(len(self.features))]


if __name__ == "__main__":
    config = KDconfig()
    a = KDDataSet(config)
    # query = "耀是什么啊"
    # query1 = "耀是什么啊水水水水"
    # texts = [[query], [query1]]
    # print(texts)
    print(a.features)
    #
    # train_dataset = BertDataSet(config, set_type="train")
    # train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=False)
    #
    # for step, batch in enumerate(train_dataloader):
    #     print(batch)
