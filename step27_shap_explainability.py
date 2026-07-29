import pandas as pd
import matplotlib.pyplot as plt
import shap
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

X_train = pd.read_csv("model_X_train.csv")
X_test = pd.read_csv("model_X_test.csv")
y_train = pd.read_csv("model_y_train.csv").squeeze()
y_test = pd.read_csv("model_y_test.csv").squeeze()

target_encoder = LabelEncoder()
y_train_encoded = target_encoder.fit_transform(y_train)

print("Retraining XGBoost with our best tuned settings...")

# --- Use the exact best settings found during tuning (Step 22) ---
model = XGBClassifier(
    n_estimators=200,
    max_depth=7,
    learning_rate=0.2,
    random_state=42,
    eval_metric="mlogloss",
)
model.fit(X_train, y_train_encoded)
print("Model trained.")

# --- Create the SHAP explainer ---
# TreeExplainer is specifically optimised for tree-based models like
# XGBoost and Random Forest - it can calculate exact SHAP values much
# faster than the general-purpose method needed for other model types.
print("\nCalculating SHAP values (this explains every single prediction in the test set)...")
explainer = shap.TreeExplainer(model)

# We only explain a SAMPLE of 500 test rows, not all 6,287 - this is
# purely for speed (SHAP calculations are detailed and can take a while
# on very large datasets), and 500 rows is still a large, representative
# sample for understanding overall patterns.
X_sample = X_test.sample(n=500, random_state=42)
shap_values = explainer(X_sample)

print("SHAP values calculated.")
print(f"Shape of SHAP values: {shap_values.values.shape}")
print("(rows, features, classes) - one contribution value per feature, per class, per row")

# --- Show which class is which number ---
print("\nClass encoding (for reference):")
for i, label in enumerate(target_encoder.classes_):
    print(f"  {i} -> {label}")

# --- Create a summary plot for EACH risk class separately ---
# This shows, for each risk level, which features push predictions
# toward or away from that specific class, and by how much.
for class_index, class_name in enumerate(target_encoder.classes_):
    plt.figure()
    shap.summary_plot(
        shap_values[:, :, class_index].values,
        X_sample,
        show=False,
        max_display=10,
    )
    plt.title(f"SHAP Summary — pushing toward/away from '{class_name}' risk")
    plt.tight_layout()
    filename = f"shap_summary_{class_name}.png"
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved -> {filename}")

print("\nSHAP explainability complete. Open the 3 saved PNG files to view the plots.")