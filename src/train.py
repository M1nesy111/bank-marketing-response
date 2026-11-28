import os
import pandas as pd
from sklearn.model_selection import train_test_split

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(base_dir, "data", "bank-additional-full.csv")
model_dir = os.path.join(base_dir, "models")

os.makedirs(model_dir, exist_ok=True)
df = pd. read_csv(data_path, sep=';')
print(df.shape)

y = (df["y"]=="yes").astype(int)

x = df.drop(columns=["y","duration"])
x = pd.get_dummies(x)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
train_scaled = scaler.fit_transform(x_train)
test_scaled = scaler.transform(x_test)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)
model.fit(train_scaled, y_train)
prob = model.predict_proba(test_scaled)[:,1]

from sklearn.metrics import roc_auc_score
auc = roc_auc_score(y_test, prob)
print("AUC: ", auc)

import joblib
joblib.dump(model, os.path.join(model_dir, "model.pkl"))
joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))
joblib.dump(x.columns.to_list(), os.path.join(model_dir, "columns.pkl"))

print(f"训练完成：AUC = {auc:.4f}，特征数 = {len(x.columns)}")