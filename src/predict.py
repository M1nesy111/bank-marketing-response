# -*- coding: utf-8 -*-
"""预测逻辑：把一条原始客户信息变成「认购概率」。

关键点：
训练时对数据做了三步处理（删列 → 独热编码 → 标准化），
线上预测时必须**一模一样地**再做一遍，否则模型输出就是错的。
"""

import os

import joblib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# 启动时加载一次，避免每次请求都读磁盘
_model = joblib.load(os.path.join(MODEL_DIR, "model.pkl"))
_scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
_columns = joblib.load(os.path.join(MODEL_DIR, "columns.pkl"))

# 训练时可用的原始字段（duration 已被剔除，不在此列）
RAW_FIELDS = [
    "age", "job", "marital", "education", "default", "housing", "loan",
    "contact", "month", "day_of_week", "campaign", "pdays", "previous",
    "poutcome", "emp.var.rate", "cons.price.idx", "cons.conf.idx",
    "euribor3m", "nr.employed",
]


def build_features(customer: dict) -> pd.DataFrame:
    """原始字段 → 模型输入矩阵（列顺序与训练时完全一致）。"""
    row = pd.DataFrame([customer])[RAW_FIELDS]
    encoded = pd.get_dummies(row)
    # reindex 会把训练时有、这次没出现的列补 0，保证列数/顺序一致
    return encoded.reindex(columns=_columns, fill_value=0)


def predict_one(customer: dict) -> float:
    """返回该客户的认购概率（0~1）。"""
    features = build_features(customer)
    scaled = _scaler.transform(features)
    return float(_model.predict_proba(scaled)[0, 1])


def predict_batch(customers: list) -> list:
    """批量预测，返回概率列表。"""
    if not customers:
        return []
    rows = pd.DataFrame(customers)[RAW_FIELDS]
    encoded = pd.get_dummies(rows).reindex(columns=_columns, fill_value=0)
    scaled = _scaler.transform(encoded)
    return [float(p) for p in _model.predict_proba(scaled)[:, 1]]