import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- Load the training/test data we prepared in Step 17 ---
X_train = pd.read_csv("model_X_train.csv")
X_test = pd.read_csv("model_X_test.csv")
y_train = pd.read_csv("model_y_train.csv").squeeze()  # .squeeze() turns a
                                                          # 1-column table
                                                          # back into a
                                                          # simple list
y_test = pd.read_csv("model_y_test.csv").squeeze()

print(f"Training on {len(X_train):,} rows, testing on {len(X_test):,} rows")

# --- Create and train the model ---
# Logistic Regression is the simplest of our 3 models - it looks for
# straightforward relationships between the features and the outcome
# (e.g. "as distance-to-landmark increases, does risk tend to go up or
# down, and by how much"). We use it as our BASELINE - if our fancier
# models (Random Forest, XGBoost) can't beat this simple one, that's an
# important, honest finding worth reporting, not something to hide.
#
# max_iter=1000: this model learns through a repeated trial-and-error
# process ("iterations"). The default limit is sometimes too low for
# it to fully finish learning, so we give it more attempts to be safe.
model = LogisticRegression(max_iter=1000, random_state=42)

model.fit(X_train, y_train)
print("Model trained.")

# --- Test it on the unseen 20% test data ---
predictions = model.predict(X_test)

# --- How accurate was it? ---
accuracy = accuracy_score(y_test, predictions)
print(f"\nOverall accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")

# --- A more detailed breakdown ---
# Accuracy alone can be misleading (e.g. if a model just always guesses
# "low", it could still look "accurate" if low happens to be common).
# classification_report() shows PRECISION (of the times it guessed
# "high", how often was it actually right) and RECALL (of all the
# TRUE "high" cases, how many did it actually catch) for each risk
# level separately - a much fairer picture.
print("\nDetailed performance report:")
print(classification_report(y_test, predictions))

# --- Confusion matrix: exactly which mistakes did it make? ---
# This shows, for example, how many actual "high" risk cases got
# wrongly predicted as "medium", vs correctly predicted as "high", etc.
print("Confusion matrix (rows = actual, columns = predicted):")
print("Order: high, low, medium")
print(confusion_matrix(y_test, predictions, labels=["high", "low", "medium"]))