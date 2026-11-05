# Train the placement model and save it as ONE file (scaler + model together).
#
# Why a Pipeline?  The model was trained on SCALED cgpa/iq values.
# If we save only the model, the app would feed it raw values and predict wrong.
# A Pipeline remembers the scaler AND the model, so predict() does both steps.

import pickle

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# 1. Load data (drop the first unnamed index column)
df = pd.read_csv("placement.csv").iloc[:, 1:]

# 2. Input and output columns
X = df[["cgpa", "iq"]]
y = df["placement"]

# 3. Train / test split (random_state → same split every run)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# 4. Scaler + model in one pipeline
model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression()),
])
model.fit(X_train, y_train)

# 5. Evaluate
print("Test accuracy:", accuracy_score(y_test, model.predict(X_test)))

# 6. Save
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
print("Saved model.pkl")
