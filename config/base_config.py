# =============================================================
# -*- coding:UTF-8 -*-
# File Name: base_config.py
# Author: baojiang
# mail: baojiang@oppo.com
# Created Time: 2022/9/23
# =============================================================
# !/usr/bin/python

class BaseConfig:
    """基础参数配置"""
    def __init__(self):
        self.train_data_path = "./data/tnews_public/train.json"
        self.test_data_path = "./data/tnews_public/test.json"
        self.dev_data_path = "./data/tnews_public/dev.json"
        self.word2idx_path = "./data/keras_resource/keras_tokenizer.pkl"
        self.embedding_path = "./data/keras_resource/keras_embedding.pkl"
