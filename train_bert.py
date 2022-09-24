# =============================================================
# -*- coding:UTF-8 -*-
# File Name: train_bert.py
# Author: baojiang
# mail: baojiang@oppo.com
# Created Time: 2022/9/23
# =============================================================
# !/usr/bin/python

import torch
import random
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import classification_report
from model.bert import Bert
from config.bert_config import BertConfig
from common.data_helper import BertDataSet


def set_seed(config):
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)


def train(config, model):
    model.to(config.device)
    model.zero_grad()

    train_dataset = BertDataSet(config, set_type="train")
    train_dataloader = DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=False)

    optimizer = Adam(model.parameters(), lr=config.lr)
    best_acc = 0

    #set_seed(config)

    for epoch in range(config.epochs):
        loss_sum = 0
        acc_sum = 0
        for step, batch in enumerate(train_dataloader):
            model.train()
            inputs = {"input_ids": batch[0],
                      "token_type_ids": batch[1],
                      "attention_mask": batch[2]}
            labels = batch[3]
            logits = model(**inputs)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            model.zero_grad()
            loss_sum += loss.item()
            if step % 1 == 0:
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
    model.eval()

    eval_dataset = BertDataSet(config, set_type="dev")
    eval_dataloader = DataLoader(eval_dataset, batch_size=config.eval_batch_size, shuffle=False)

    idx2label = {idx:label for label, idx in config.label2idx.items()}

    pred_labels, true_labels = [], []

    with torch.no_grad():
        for step, batch in enumerate(eval_dataloader):
            model.train()
            inputs = {"input_ids": batch[0],
                      "token_type_ids": batch[1],
                      "attention_mask": batch[2]}
            labels = batch[3]
            logits = model(**inputs)
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

    test_dataset = BertDataSet(config, set_type="test")
    test_dataloader = DataLoader(test_dataset, config.eval_batch_size, shuffle=False)

    idx2label = {idx:label for label, idx in config.label2idx.items()}

    pred_labels = []

    with torch.no_grad():
        for step, batch in enumerate(test_dataloader):
            model.train()
            inputs = {"input_ids": batch[0],
                      "token_type_ids": batch[1],
                      "attention_mask": batch[2]}
            labels = batch[3]
            logits = model(**inputs)
            preds = torch.argmax(logits, dim=1)
            pred_labels.extend(preds.detach().cpu().tolist())

    pred_labels = [idx2label[i] for i in pred_labels]
    return pred_labels


def main():
    config = BertConfig()

    #set_seed(config)

    if config.do_train:
        model = Bert(config)
        train(config, model)

    if config.do_eval:
        model = Bert(config)
        checkpoint = torch.load(config.model_path)  # 先加载参数
        model.load_state_dict(checkpoint)
        acc, pred_report = evaluate(config, model)

    if config.do_predict:
        model = Bert(config)
        checkpoint = torch.load(config.model_path)  # 先加载参数
        model.load_state_dict(checkpoint)
        pred_res = predict(config, model)
        print(len(pred_res))


if __name__ == "__main__":
    main()
