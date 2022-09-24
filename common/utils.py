# =============================================================
# -*- coding:UTF-8 -*-
# File Name: utils.py
# Author: baojiang
# mail: baojiang@oppo.com
# Created Time: 2022/9/24
# =============================================================
# !/usr/bin/python

import json
import pickle


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def to_pkl(data, path):
    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

