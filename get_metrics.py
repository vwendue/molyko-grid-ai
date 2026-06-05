import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Load data
df = pd.read_csv('molyko_master_data.csv')

# Prepare X and y
X = pd.get_dummies(df[['Voltage_V', 'Flicker_Count', 'Weather', 'ENEO_Post']], drop_first=True)
y = df['Outage_Occured']

# 80/20 Split
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
dt = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X_tr, y_tr)
lr = LogisticRegression(max_iter=1000).fit(X_tr, y_tr)

# Predict
dt_pred = dt.predict(X_te)
lr_pred = lr.predict(X_te)
lr_prob = lr.predict_proba(X_te)[:, 1] # Probabilities for ROC-AUC

# Print the table data
print("MODEL | ACCURACY | PRECISION | RECALL | F1-SCORE | ROC-AUC")
print("-" * 60)
print(f"DT    | {accuracy_score(y_te, dt_pred)*100:.1f}%    | {precision_score(y_te, dt_pred, zero_division=0):.2f}      | {recall_score(y_te, dt_pred, zero_division=0):.2f}   | {f1_score(y_te, dt_pred, zero_division=0):.2f}     | —")
print(f"LR    | {accuracy_score(y_te, lr_pred)*100:.1f}%    | {precision_score(y_te, lr_pred, zero_division=0):.2f}      | {recall_score(y_te, lr_pred, zero_division=0):.2f}   | {f1_score(y_te, lr_pred, zero_division=0):.2f}     | {roc_auc_score(y_te, lr_prob):.2f}")