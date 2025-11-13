"""
Fairness audit on COMPAS using IBM AIF360.

- Loads COMPAS dataset from AIF360.
- Trains a logistic regression baseline.
- Computes fairness metrics (disparate impact, equal opportunity difference).
- Applies Reweighing preprocessing and retrains to show effect.
- Produces a bar chart comparing False Positive Rates (FPR) across racial groups.

Notes:
- Install dependencies: pip install aif360 scikit-learn pandas matplotlib
- Run in a Python environment that has AIF360 available.
"""
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

from aif360.datasets import CompasDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing

# Output directory
OUT_DIR = Path("reports")
OUT_DIR.mkdir(exist_ok=True)


def prepare_data():
    """
    Load COMPAS dataset as an AIF360 BinaryLabelDataset and split into train/test.
    Returns dataset_train, dataset_test, privileged_groups, unprivileged_groups.
    """
    # Load COMPAS dataset (AIF360 built-in)
    dataset = CompasDataset()

    # Define privileged / unprivileged groups.
    # The CompasDataset encodes 'race' as a protected attribute; look at dataset for exact values.
    # Typical convention: privileged = ['Caucasian'], unprivileged = ['African-American', ...]
    privileged_groups = [{'race': 1}]
    unprivileged_groups = [{'race': 0}]

    # Train/test split (70/30)
    dataset_train, dataset_test = dataset.split([0.7], shuffle=True)

    return dataset_train, dataset_test, privileged_groups, unprivileged_groups


def train_logistic(dataset_train):
    """
    Train a logistic regression on AIF360 BinaryLabelDataset.
    Returns trained sklearn model and scaler.
    """
    X_train = dataset_train.features
    y_train = dataset_train.labels.ravel()

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    clf = LogisticRegression(solver="liblinear")
    clf.fit(X_train_scaled, y_train)
    return clf, scaler


def predict_and_wrap(clf, scaler, dataset):
    """
    Produce predictions using clf for the given AIF360 dataset and return a copy
    of the dataset with predictions set as labels (for metric computation).
    """
    X = dataset.features
    X_scaled = scaler.transform(X)
    y_pred = clf.predict(X_scaled)
    y_score = clf.predict_proba(X_scaled)[:, 1]

    dataset_pred = dataset.copy()
    dataset_pred.labels = y_pred.reshape(-1, 1)
    # store scores for potential use
    dataset_pred.scores = y_score.reshape(-1, 1)
    return dataset_pred


def compute_metrics(dataset_test, dataset_pred, privileged_groups, unprivileged_groups):
    """
    Compute and return selected fairness metrics and group FPRs.
    """
    # Overall disparate impact on true labels (selection rates)
    bldm = BinaryLabelDatasetMetric(
        dataset_test, privileged_groups=privileged_groups, unprivileged_groups=unprivileged_groups
    )
    disparate_impact = bldm.disparate_impact()

    # Classification metrics between true and predicted
    clf_metric = ClassificationMetric(
        dataset_test,
        dataset_pred,
        privileged_groups=privileged_groups,
        unprivileged_groups=unprivileged_groups,
    )

    # Equal opportunity difference = TPR_unprivileged - TPR_privileged
    tpr_unpriv = clf_metric.true_positive_rate(privileged=False)
    tpr_priv = clf_metric.true_positive_rate(privileged=True)
    equal_opportunity_diff = float(tpr_unpriv - tpr_priv)

    # False positive rates
    fpr_unpriv = clf_metric.false_positive_rate(privileged=False)
    fpr_priv = clf_metric.false_positive_rate(privileged=True)

    metrics = {
        "disparate_impact": float(disparate_impact),
        "equal_opportunity_difference": float(equal_opportunity_diff),
        "fpr_unprivileged": float(fpr_unpriv),
        "fpr_privileged": float(fpr_priv),
    }
    return metrics


def plot_fpr(metrics_before, metrics_after, out_path):
    """
    Create a bar chart comparing FPR for privileged vs unprivileged before/after mitigation.
    """
    groups = ["Privileged", "Unprivileged"]
    fpr_before = [metrics_before["fpr_privileged"], metrics_before["fpr_unprivileged"]]
    fpr_after = [metrics_after["fpr_privileged"], metrics_after["fpr_unprivileged"]]

    x = np.arange(len(groups))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - width / 2, fpr_before, width, label="Before (baseline)")
    ax.bar(x + width / 2, fpr_after, width, label="After (Reweighing)")

    ax.set_ylabel("False Positive Rate")
    ax.set_title("FPR by group — Privileged vs Unprivileged")
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def main():
    print("Starting fairness audit (COMPAS) using AIF360...")

    dataset_train, dataset_test, privileged_groups, unprivileged_groups = prepare_data()

    # Baseline model
    clf, scaler = train_logistic(dataset_train)
    dataset_test_pred = predict_and_wrap(clf, scaler, dataset_test)
    metrics_before = compute_metrics(dataset_test, dataset_test_pred, privileged_groups, unprivileged_groups)
    print("Baseline metrics:")
    for k, v in metrics_before.items():
        print(f"  {k}: {v:.4f}")

    # Apply Reweighing pre-processing
    rw = Reweighing(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
    dataset_train_rw = rw.fit_transform(dataset_train)

    # Train with instance weights
    X_train_rw = dataset_train_rw.features
    y_train_rw = dataset_train_rw.labels.ravel()
    sample_weights = dataset_train_rw.instance_weights.ravel()

    scaler_rw = StandardScaler()
    X_train_rw_scaled = scaler_rw.fit_transform(X_train_rw)

    clf_rw = LogisticRegression(solver="liblinear")
    clf_rw.fit(X_train_rw_scaled, y_train_rw, sample_weight=sample_weights)

    # Predict and evaluate
    dataset_test_pred_rw = predict_and_wrap(clf_rw, scaler_rw, dataset_test)
    metrics_after = compute_metrics(dataset_test, dataset_test_pred_rw, privileged_groups, unprivileged_groups)
    print("\nMetrics after Reweighing:")
    for k, v in metrics_after.items():
        print(f"  {k}: {v:.4f}")

    # Visualization
    plot_path = OUT_DIR / "fpr_priv_unpriv.png"
    plot_fpr(metrics_before, metrics_after, plot_path)
    print(f"\nSaved FPR comparison plot to: {plot_path}")

    # Short findings report
    report = f"""
Fairness audit summary (COMPAS) — brief report
---------------------------------------------
Baseline disparate impact: {metrics_before['disparate_impact']:.4f}
Baseline equal opportunity difference (unpriv - priv): {metrics_before['equal_opportunity_difference']:.4f}
After Reweighing disparate impact: {metrics_after['disparate_impact']:.4f}
After Reweighing equal opportunity difference: {metrics_after['equal_opportunity_difference']:.4f}

Interpretation:
- Disparate impact < 1 indicates the unprivileged group is selected at a lower rate.
- Positive equal opportunity difference means higher TPR for unprivileged group (or vice-versa depending on sign).
Mitigation recommendation:
- If disparities persist, combine preprocessing (reweighing) with in-processing constraints or post-processing calibration.
- Implement monitoring and human oversight for high-stakes decisions.
"""
    report_path = OUT_DIR / "fairness_report.txt"
    report_path.write_text(report.strip(), encoding="utf-8")
    print(f"\nWrote brief report to: {report_path}")


if __name__ == "__main__":
    main()