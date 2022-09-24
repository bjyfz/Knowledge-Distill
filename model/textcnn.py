# =============================================================
# -*- coding:UTF-8 -*-
# File Name: textcnn.py
# Author: baojiang
# mail: baojiang@oppo.com
# Created Time: 2022/9/24
# =============================================================
# !/usr/bin/python

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from config.textcnn_config import TextCNNConfig


class TextCnn(nn.Module):
    """
    textcnn 文本分类
    """
    def __init__(self, config):
        super(TextCnn, self).__init__()
        if config.embedding_pretrained is not None:
            # 加载已训练好的embedding
            # freeze=False 训练过程中不会更新
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)

        else:
            # 根据词表大小和emb维度初始化，训练过程中会更新
            self.embedding = nn.Embedding(config.vocab_size, config.embed_size)

        self.convs = nn.ModuleList([nn.Conv2d(1, config.kernel_num, (k, config.embed_size)) for k in config.kernel_sizes])
        self.droupout = nn.Dropout(config.droupout)
        self.fc = nn.Linear(config.kernel_num * len(config.kernel_sizes), config.num_classes)

    def encoder(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (N, kernel_num, token_num)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)  # (N, kernel_num)
        return x

    def forward(self, x):
        x = self.embedding(x)  # (N, token_num, embed_size)
        x = x.unsqueeze(1)  # (N, 1, token_num, embed_size)
        x = torch.cat([self.encoder(x, conv) for conv in self.convs], 1)  # (N, kernel_num * len(kernel_sizes))
        x = self.droupout(x)  # (N, kernel_num * len(kernel_sizes))
        logits = self.fc(x)  # (N, num_classes)
        return logits


def main():
    config = TextCNNConfig()

    x = torch.randn(config.max_length, config.embed_size).unsqueeze(0).unsqueeze(0)
    print(x)

    conv = nn.Conv2d(1, 2, (3, 5), padding=(2, 0)) # padding的是bias的值
    print(conv.weight)
    print(conv.bias)

    x = conv(x)
    print(x)

    inputs = torch.randint(2, 10, (1, config.max_seq_len))
    print(inputs)

    model = TextCnn(config)

    print(model(inputs))


if __name__ == "__main__":
    main()
