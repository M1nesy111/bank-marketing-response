# -*- coding: utf-8 -*-
"""一键训练脚本：读取数据 → 训练逻辑回归 → 保存模型产物。

用法（在项目根目录）：
    python src/train.py
"""

import os

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "bank-additional-full.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# duration 是通话结束后才产生的变量，会造成数据泄漏，必须剔除
DROP_COLUMNS = ["y", "duration"]


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH, sep=";")
    y = (df["y"] == "yes").astype(int)
    X = pd.get_dummies(df.drop(columns=DROP_COLUMNS))

    # 保存特征列名，预测时要用它对齐
    columns = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)
    auc = roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1])

    joblib.dump(model, os.path.join(MODEL_DIR, "model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(columns, os.path.join(MODEL_DIR, "columns.pkl"))

    print(f"训练完成：AUC = {auc:.4f}，特征数 = {len(columns)}")
    print(f"模型产物已保存到：{MODEL_DIR}")


if __name__ == "__main__":
    main()