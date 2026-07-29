import pandas as pd
from scipy import stats

# --- Load the fold-by-fold scores we saved in Step 25 ---
fold_scores = pd.read_csv("cross_validation_fold_scores.csv")
print("Fold-by-fold accuracy scores:")
print(fold_scores)

# --- Paired t-test between each pair of models ---
# Since all 3 models were tested on the EXACT SAME 5 folds (this is why
# we used one shared cv_folds object in Step 25), we can use a PAIRED
# t-test - this compares each fold's scores directly against each other
# (Fold 1 vs Fold 1, Fold 2 vs Fold 2, etc.) rather than treating them
# as two separate, unrelated groups of numbers. Paired tests are more
# statistically powerful when the same test conditions were used for
# both things being compared, which is exactly our situation here.
#
# The p-value tells us: "if there were truly NO real difference between
# these two models, how likely would we be to see a gap this large just
# by chance?" A common threshold is p < 0.05 (less than 5% chance this
# gap is just random noise) - below this, we call the difference
# "statistically significant."


def compare_models(name1, name2):
    scores1 = fold_scores[name1]
    scores2 = fold_scores[name2]

    t_statistic, p_value = stats.ttest_rel(scores1, scores2)

    mean_diff = scores1.mean() - scores2.mean()

    print(f"\n{name1} vs {name2}:")
    print(f"  Mean accuracy difference: {mean_diff*100:+.2f} percentage points")
    print(f"  p-value: {p_value:.4f}")

    if p_value < 0.05:
        print(f"  RESULT: statistically significant difference (p < 0.05)")
    else:
        print(f"  RESULT: NOT statistically significant (p >= 0.05) - the difference could just be random noise")


compare_models("XGBoost", "Random Forest")
compare_models("Random Forest", "Logistic Regression")
compare_models("XGBoost", "Logistic Regression")