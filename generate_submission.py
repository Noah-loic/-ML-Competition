import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

# Load data
train = pd.read_csv("input/train.csv")
test  = pd.read_csv("input/test.csv")

# Features to use
cat_cols = ["Player_Type", "Position_Type", "Position"]
num_cols = ["Year", "Age", "Height", "Weight",
            "Sprint_40yd", "Vertical_Jump", "Bench_Press_Reps",
            "Broad_Jump", "Agility_3cone", "Shuttle"]

# Encode categoricals
le = LabelEncoder()
for col in cat_cols:
    combined = pd.concat([train[col], test[col]], axis=0).astype(str)
    le.fit(combined)
    train[col] = le.transform(train[col].astype(str))
    test[col]  = le.transform(test[col].astype(str))

features = num_cols + cat_cols

X = train[features]
y = train["Drafted"]
X_test = test[features]

# StratifiedKFold cross-validation + OOF predictions
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
test_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    model = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    test_preds += model.predict_proba(X_test)[:, 1]
    print(f"Fold {fold+1} done")

test_preds /= 5

# Save submission
submission = pd.read_csv("input/sample_submission.csv")
submission["Drafted"] = test_preds
submission.to_csv("submission.csv", index=False)
print("submission.csv saved!")
print(submission.head())
