# =============================================================
# -*- coding:UTF-8 -*-
# File Name: distill_losses.py
# Author: baojiang
# mail: baojiang@oppo.com
# Created Time: 2022/9/24
# =============================================================
# !/usr/bin/python

import torch.nn.functional as F
from torch import nn


def kd_kl_loss(logits_S, logits_T, labels, T, alpha):
    soft_loss = nn.KLDivLoss()(F.log_softmax(logits_S / T, dim=1), F.softmax(logits_T / T, dim=1)) * T**2
    hard_loss = F.cross_entropy(logits_S, labels)
    loss = alpha * soft_loss + (1. - alpha) * hard_loss
    return loss