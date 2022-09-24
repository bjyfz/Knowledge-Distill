# =============================================================
# -*- coding:UTF-8 -*-
# File Name: keras_embedding_build.py
# Author: baojiang
# mail: baojiang@oppo.com
# Created Time: 2022/9/24
# =============================================================
# !/usr/bin/python
"""
下载腾讯词向量，并对其进行预处理，输出字向量与对应的tokenizer。
腾讯词向量下载地址：https://ai.tencent.com/ailab/nlp/en/download.html
解压后是一个约23G的txt文件，命名为：tencent-ailab-embedding-zh-d200-v0.2.0.txt
"""
from tqdm import tqdm
import os
import numpy as np
from common.utils import *


# 加载腾讯词向量, 保存单字向量
def save_char_embedding(origin_path, embedding_path, vocab_path):
    n = 0
    words = []
    embeddings = []
    with open(origin_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in tqdm(f, desc="reading tencent emb"):
            line = line.split()
            if len(line) == 201 and len(line[0]) == 1:

                vector = list(map(lambda x: float(x), line[1:]))  # 对词向量进行处理
                vec = np.array(vector)  # 将列表转化为array
                embeddings.append(vec)
                words.append(line[0])
                if line[0] == "我":
                    print(n)
                    print(vec)
                n = n + 1
            # if n == 1000:
            #     break
    embeddings = np.array(embeddings)
    to_pkl(embeddings, embedding_path)  # 保存字向量

    with open(vocab_path, "w", encoding="utf-8") as f:  # 保存字表
        for word in words:
            f.write("%s\n" % word)

    print("vocab: ", len(words))  # 输出清洗后的range

    print("successfully load tencent word embedding!")


def main():
    data_dir = "/Users/baojiang/Downloads/code/tencent-ailab-embedding-zh-d200-v0.2.0"
    origin_path = os.path.join(data_dir, "tencent-ailab-embedding-zh-d200-v0.2.0.txt")

    embedding_path = os.path.join(data_dir, "tencent_char_embedding.pkl")
    vocab_path = os.path.join(data_dir, "tencent_char_vocab.txt")

    save_char_embedding(origin_path, embedding_path, vocab_path)  # 保存腾讯字向量与字分词器


if __name__ == "__main__":
    main()






