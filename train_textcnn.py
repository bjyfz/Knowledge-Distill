# =============================================================
# -*- coding:UTF-8 -*-
# File Name: train_textcnn.py
# Author: baojiang
# mail: baojiang@oppo.com
# Created Time: 2022/9/24
# =============================================================
# !/usr/bin/python

import torch
import random
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim import Adam
from model.textcnn import TextCnn
from config.textcnn_config import TextCNNConfig
from sklearn.metrics import classification_report
from common.data_helper import CnnDataSet


def set_seed(config):
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)


def train(config, model):
    model.to(config.device)
    model.zero_grad()

    # DataSet和DataLoader配合使用构建batch数据集
    train_dataset = CnnDataSet(config, set_type="train")
    train_dataloader = DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=False)

    optimizer = Adam(model.parameters(), lr=config.lr)
    best_acc = 0

    #set_seed(config)

    for epoch in range(config.epochs):
        loss_sum = 0
        acc_sum = 0
        for step, (features, labels) in enumerate(train_dataloader):
            model.train()  # 保证dropout和NB在每个batch内有效
            features = features.to(config.device)
            labels = labels.to(config.device)
            logits = model(features)
            loss = F.cross_entropy(logits, labels)
            loss.backward()  # 梯度计算，反向传播
            optimizer.step()  # 参数更新
            model.zero_grad()  # 梯度清零
            loss_sum += loss.item()
            if step % 100 == 0:
                labels = labels.detach().cpu().numpy()
                preds = torch.argmax(logits, dim=1)
                preds = preds.detach().cpu().numpy()
                acc = np.sum(preds == labels) / len(preds)
                acc_sum += acc
                print(" TRAIN: epoch: {}/{} step: {}/{} acc: {} loss: {} ".format\
                          (epoch + 1, config.epochs, step, len(train_dataloader), acc, loss))
        acc, pred_report = evaluate(config, model)
        print("DEV: acc: {} ".format(acc))
        print("DEV: classification report: \n{}".format(pred_report))

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), config.output_model_path)


def evaluate(config, model):
    model.to(config.device)
    model.eval()  # 去除dropout，使用所有数据的BN

    eval_dataset = CnnDataSet(config, set_type="dev")
    eval_dataloader = DataLoader(eval_dataset, batch_size=config.eval_batch_size, shuffle=False)

    idx2label = {idx:label for label, idx in config.label2idx.items()}

    pred_labels, true_labels = [], []

    with torch.no_grad():
        for i, (texts, labels) in enumerate(eval_dataloader):
            texts = texts.to(config.device)
            labels = labels.to(config.device)
            logits = model(texts)
            preds = torch.argmax(logits, dim=1)
            pred_labels.extend(preds.detach().cpu().tolist())
            true_labels.extend(labels.detach().cpu().tolist())
    pred_labels = [idx2label[i] for i in pred_labels]
    true_labels = [idx2label[i] for i in true_labels]
    acc = sum([1 if p == t else 0 for p, t in zip(pred_labels, true_labels)]) / len(pred_labels)
    pred_report = classification_report(true_labels, pred_labels)
    return acc, pred_report


def predict(config, model):
    model.to(config.device)
    model.eval()

    test_dataset = CnnDataSet(config, set_type="test")
    test_dataloader = DataLoader(test_dataset, config.eval_batch_size, shuffle=False)

    idx2label = {idx: label for label, idx in config.label2idx.items()}

    pred_labels = []

    with torch.no_grad():
        for i, (texts, labels) in enumerate(test_dataloader):
            texts = texts.to(config.device)
            logits = model(texts)
            preds = torch.argmax(logits, dim=1)
            pred_labels.extend(preds.detach().cpu().tolist())

    pred_labels = [idx2label[i] for i in pred_labels]
    return pred_labels


def main():
    config = TextCNNConfig()

    #set_seed(config)

    if config.do_train:
        model = TextCnn(config)
        train(config, model)

    if config.do_eval:
        model = TextCnn(config)
        checkpoint = torch.load(config.model_path)  # 先加载参数
        model.load_state_dict(checkpoint)
        acc, pred_report = evaluate(config, model)

    if config.do_predict:
        model = TextCnn(config)
        checkpoint = torch.load(config.model_path)  # 先加载参数
        model.load_state_dict(checkpoint)
        pred_res = predict(config, model)
        print(len(pred_res))


if __name__ == "__main__":
    main()