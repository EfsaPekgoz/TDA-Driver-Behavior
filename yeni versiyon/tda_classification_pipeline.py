"""
TDA-Based Driver Behavior Classification Pipeline
====================================================
Based on: "Time Series Classification via Topological Data Analysis"
          Karan & Kaygun (2021, arXiv:2102.01956v2)

Pipeline:
  1) Read embedded_output.csv (10-D delay embedding + Label)
  2) Create subwindows (point clouds) from consecutive rows
  3) Compute persistent homology (H0, H1) via Ripser
  4) Extract 7 topological features per diagram (Algorithm 4 from paper)
  5) Classify with SVC, RF, AdaBoost, LDA
  6) Visualize results
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from ripser import ripser
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, cross_validate, cross_val_predict
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, f1_score
)
import warnings, time, os
warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================
INPUT_FILE = "embedded_output.csv"
SUBWINDOW_SIZE = 20       # Number of rows per subwindow (point cloud)
SUBWINDOW_STRIDE = 10     # Stride between subwindows
MAX_HOMOLOGY_DIM = 1      # Compute H0 and H1 (as in the paper)
N_FOLDS = 5               # Stratified K-Fold
RANDOM_STATE = 0          # For reproducibility (paper uses 0)
OUTPUT_DIR = "tda_results"
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------------------------------------------------------
# 1. FEATURE EXTRACTION FROM PERSISTENCE DIAGRAMS (Algorithm 4)
# ------------------------------------------------------------------

def compute_betti_curve(diagram, n_bins=100):
    """Compute the Betti curve from a persistence diagram."""
    if len(diagram) == 0:
        return np.zeros(n_bins)

    births = diagram[:, 0]
    deaths = diagram[:, 1]

    # Remove infinite death times
    finite_mask = np.isfinite(deaths)
    births = births[finite_mask]
    deaths = deaths[finite_mask]

    if len(births) == 0:
        return np.zeros(n_bins)

    t_min = births.min()
    t_max = deaths.max()

    if t_max <= t_min:
        return np.zeros(n_bins)

    t = np.linspace(t_min, t_max, n_bins)
    betti = np.zeros(n_bins)

    for b, d in zip(births, deaths):
        betti += ((t >= b) & (t <= d)).astype(float)

    return betti


def compute_persistence_landscape(diagram, n_bins=100, k=1):
    """Compute the k-th persistence landscape from a persistence diagram.
    Paper uses k=1 (first layer)."""
    if len(diagram) == 0:
        return np.zeros(n_bins)

    births = diagram[:, 0]
    deaths = diagram[:, 1]

    finite_mask = np.isfinite(deaths)
    births = births[finite_mask]
    deaths = deaths[finite_mask]

    if len(births) == 0:
        return np.zeros(n_bins)

    t_min = births.min()
    t_max = deaths.max()

    if t_max <= t_min:
        return np.zeros(n_bins)

    t = np.linspace(t_min, t_max, n_bins)
    landscape = np.zeros(n_bins)

    # For each t, compute g_alpha(t) for all alpha, then take k-th max
    g_values = np.zeros((len(births), n_bins))

    for i, (b, d) in enumerate(zip(births, deaths)):
        mid = (b + d) / 2.0
        for j, tj in enumerate(t):
            if b <= tj <= mid:
                g_values[i, j] = tj - b
            elif mid < tj <= d:
                g_values[i, j] = d - tj
            else:
                g_values[i, j] = 0.0

    # k-th maximum at each t
    for j in range(n_bins):
        sorted_vals = np.sort(g_values[:, j])[::-1]
        if len(sorted_vals) >= k:
            landscape[j] = sorted_vals[k - 1]

    return landscape


def extract_features_from_diagram(diagram):
    """
    Extract 7 features from a single persistence diagram (Algorithm 4).
    f1: Wasserstein-1 distance to empty diagram = (1/sqrt(2)) * sum(lifetimes)
    f2: Bottleneck distance to empty diagram   = (1/sqrt(2)) * max(lifetime)
    f3: Persistent entropy
    f4: ||Betti curve||_1
    f5: ||Betti curve||_2
    f6: ||Persistence landscape||_1
    f7: ||Persistence landscape||_2
    """
    features = np.zeros(7)

    if len(diagram) == 0:
        return features

    births = diagram[:, 0]
    deaths = diagram[:, 1]

    # Remove infinite death times
    finite_mask = np.isfinite(deaths)
    births_f = births[finite_mask]
    deaths_f = deaths[finite_mask]

    if len(births_f) == 0:
        return features

    lifetimes = deaths_f - births_f
    lifetimes = lifetimes[lifetimes > 0]  # Remove zero lifetimes

    if len(lifetimes) == 0:
        return features

    sqrt2 = np.sqrt(2)

    # f1: Wasserstein-1
    features[0] = np.sum(lifetimes) / sqrt2

    # f2: Bottleneck
    features[1] = np.max(lifetimes) / sqrt2

    # f3: Persistent entropy
    L = np.sum(lifetimes)
    if L > 0:
        probs = lifetimes / L
        probs = probs[probs > 0]  # Avoid log(0)
        features[2] = -np.sum(probs * np.log(probs))

    # Betti curve
    betti = compute_betti_curve(diagram)
    dt = 1.0  # uniform spacing

    # f4: ||Betti||_1
    features[3] = np.sum(np.abs(betti)) * dt

    # f5: ||Betti||_2
    features[4] = np.sqrt(np.sum(betti ** 2) * dt)

    # Persistence landscape (k=1)
    landscape = compute_persistence_landscape(diagram)

    # f6: ||Landscape||_1
    features[5] = np.sum(np.abs(landscape)) * dt

    # f7: ||Landscape||_2
    features[6] = np.sqrt(np.sum(landscape ** 2) * dt)

    return features


def extract_all_features(point_cloud):
    """
    Given a point cloud (subwindow in R^10), compute persistent homology
    and extract 14 features (7 from H0, 7 from H1).
    """
    result = ripser(point_cloud, maxdim=MAX_HOMOLOGY_DIM)
    diagrams = result["dgms"]

    features_h0 = extract_features_from_diagram(diagrams[0])
    features_h1 = extract_features_from_diagram(diagrams[1]) if len(diagrams) > 1 else np.zeros(7)

    return np.concatenate([features_h0, features_h1])


# ------------------------------------------------------------------
# 2. SUBWINDOWING
# ------------------------------------------------------------------

def create_subwindows_with_labels(data, labels, subwindow_size, stride):
    """
    Create subwindows from embedded data. Each subwindow is a point cloud.
    The label is the majority label within the subwindow.
    Also returns group_id per subwindow for group-aware cross-validation.
    """
    n = len(data)
    subwindows = []
    sw_labels = []
    sw_groups = []

    # Block size large enough to keep overlapping windows in the same group
    block_size = subwindow_size + stride

    for start in range(0, n - subwindow_size + 1, stride):
        end = start + subwindow_size
        sw = data[start:end]
        lbl_window = labels[start:end]

        # Majority vote for label
        unique, counts = np.unique(lbl_window, return_counts=True)
        majority_label = unique[np.argmax(counts)]

        # Only keep pure subwindows (all same label) to avoid boundary contamination
        if counts.max() == subwindow_size:
            subwindows.append(sw)
            sw_labels.append(majority_label)
            sw_groups.append(start // block_size)

    return subwindows, sw_labels, sw_groups


# ------------------------------------------------------------------
# 3. MAIN PIPELINE
# ------------------------------------------------------------------

def main():
    t_start = time.time()
    print("=" * 70)
    print("  TDA-Based Driver Behavior Classification Pipeline")
    print("  Based on Karan & Kaygun (2021)")
    print("=" * 70)

    # --- Load data ---
    print("\n[1/6] Loading data...")
    df = pd.read_csv(INPUT_FILE)
    feature_cols = [c for c in df.columns if c != "Label"]
    data = df[feature_cols].values.astype(np.float64)
    labels = df["Label"].values
    print(f"      Loaded {len(df):,} rows, {len(feature_cols)} dims")
    print(f"      Labels: {dict(zip(*np.unique(labels, return_counts=True)))}")

    # --- Create subwindows per label group ---
    print(f"\n[2/6] Creating subwindows (size={SUBWINDOW_SIZE}, stride={SUBWINDOW_STRIDE})...")
    all_subwindows = []
    all_labels = []
    all_groups = []
    group_offset = 0  # Offset to make group_ids unique across label groups

    for label in np.unique(labels):
        mask = labels == label
        group_data = data[mask]
        group_labels = labels[mask]

        sws, sw_lbls, sw_grps = create_subwindows_with_labels(
            group_data, group_labels, SUBWINDOW_SIZE, SUBWINDOW_STRIDE
        )
        # Offset group_ids so different label groups don't share IDs
        max_grp = max(sw_grps) + 1 if sw_grps else 0
        all_subwindows.extend(sws)
        all_labels.extend(sw_lbls)
        all_groups.extend([g + group_offset for g in sw_grps])
        group_offset += max_grp
        print(f"      '{label}': {len(sws)} subwindows, {max_grp} groups")

    group_id_array = np.array(all_groups)
    print(f"      Total subwindows: {len(all_subwindows)}")
    print(f"      Total unique groups: {len(np.unique(group_id_array))}")

    # --- Compute persistent homology & extract features ---
    print(f"\n[3/6] Computing persistent homology & extracting features...")
    print(f"      (7 features x 2 homology dims = 14 features per subwindow)")

    n_total = len(all_subwindows)
    X = np.zeros((n_total, 14))

    for i, sw in enumerate(all_subwindows):
        X[i] = extract_all_features(sw)
        if (i + 1) % 200 == 0 or (i + 1) == n_total:
            pct = 100 * (i + 1) / n_total
            print(f"      Progress: {i+1}/{n_total} ({pct:.1f}%)")

    y = np.array(all_labels)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Feature names
    feat_names = []
    for h in ["H0", "H1"]:
        for f in ["Wasserstein1", "Bottleneck", "PersEntropy",
                   "BettiL1", "BettiL2", "LandscapeL1", "LandscapeL2"]:
            feat_names.append(f"{h}_{f}")

    features_df = pd.DataFrame(X, columns=feat_names)
    features_df["Label"] = y
    features_df.to_csv(os.path.join(OUTPUT_DIR, "tda_features.csv"), index=False)
    print(f"      Features saved to {OUTPUT_DIR}/tda_features.csv")

    # --- Remove constant / highly correlated features ---
    print("\n[4/6] Preprocessing features...")
    # Remove constant features
    non_const_mask = X.std(axis=0) > 1e-10
    X_clean = X[:, non_const_mask]
    clean_names = [n for n, m in zip(feat_names, non_const_mask) if m]
    print(f"      Non-constant features: {len(clean_names)}/{len(feat_names)}")
    print(f"      Active features: {clean_names}")

    # Remove highly correlated (>0.9) features
    if X_clean.shape[1] > 1:
        corr = np.corrcoef(X_clean.T)
        to_drop = set()
        for i in range(len(corr)):
            for j in range(i + 1, len(corr)):
                if abs(corr[i, j]) > 0.9:
                    to_drop.add(j)
        keep_mask = [i not in to_drop for i in range(X_clean.shape[1])]
        X_final = X_clean[:, keep_mask]
        final_names = [n for n, m in zip(clean_names, keep_mask) if m]
        print(f"      After correlation filter (>0.9): {len(final_names)} features")
        print(f"      Final features: {final_names}")
    else:
        X_final = X_clean
        final_names = clean_names

    # Normalize to [0, 1]
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_final)

    # --- Classification ---
    print(f"\n[5/6] Running classification ({N_FOLDS}-fold CV)...")
    classifiers = {
        "SVC (linear, C=0.1)": SVC(kernel="linear", C=0.1, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=5, random_state=RANDOM_STATE
        ),
        "AdaBoost": AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=5),
            n_estimators=100, random_state=RANDOM_STATE
        ),
        "LDA": LinearDiscriminantAnalysis(),
    }

    cv_leaky = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_grouped = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    results_leaky = {}
    results_grouped = {}

    # --- 5a) Leaky CV (old, for comparison) ---
    print("\n      --- StratifiedKFold (LEAKY, for comparison) ---")
    for name, clf in classifiers.items():
        scoring = ["accuracy", "f1_macro"]
        cv_results = cross_validate(
            clf, X_scaled, y_encoded, cv=cv_leaky, scoring=scoring, return_train_score=False
        )
        acc = cv_results["test_accuracy"].mean()
        f1 = cv_results["test_f1_macro"].mean()
        acc_std = cv_results["test_accuracy"].std()
        f1_std = cv_results["test_f1_macro"].std()
        results_leaky[name] = {
            "accuracy": acc, "accuracy_std": acc_std,
            "f1_macro": f1, "f1_std": f1_std
        }
        print(f"      {name:25s}  Acc: {acc:.4f} (+/-{acc_std:.4f})  F1: {f1:.4f} (+/-{f1_std:.4f})")

    # --- 5b) Group-aware CV (correct, no leakage) ---
    print("\n      --- StratifiedGroupKFold (NO LEAKAGE) ---")
    best_acc = 0
    best_model_name = ""

    for name, clf in classifiers.items():
        scoring = ["accuracy", "f1_macro"]
        cv_results = cross_validate(
            clf, X_scaled, y_encoded, cv=cv_grouped, scoring=scoring,
            return_train_score=False, groups=group_id_array
        )
        acc = cv_results["test_accuracy"].mean()
        f1 = cv_results["test_f1_macro"].mean()
        acc_std = cv_results["test_accuracy"].std()
        f1_std = cv_results["test_f1_macro"].std()
        results_grouped[name] = {
            "accuracy": acc, "accuracy_std": acc_std,
            "f1_macro": f1, "f1_std": f1_std
        }
        if acc > best_acc:
            best_acc = acc
            best_model_name = name
        print(f"      {name:25s}  Acc: {acc:.4f} (+/-{acc_std:.4f})  F1: {f1:.4f} (+/-{f1_std:.4f})")

    print(f"\n      >>> Best model (grouped): {best_model_name} (Accuracy: {best_acc:.4f})")

    # Cross-validated predictions using the grouped CV for a fair confusion matrix
    best_clf = classifiers[best_model_name]
    y_pred_cv = cross_val_predict(
        best_clf, X_scaled, y_encoded, cv=cv_grouped, groups=group_id_array
    )

    # --- Visualizations ---
    print(f"\n[6/6] Generating visualizations...")

    # --- Plot 1: Persistence Diagram Examples (one per class) ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Persistence Diagrams (H0 & H1) - Example Subwindows per Class",
                 fontsize=14, fontweight="bold")

    for idx, label in enumerate(le.classes_):
        label_indices = np.where(y == label)[0]
        sample_idx = label_indices[len(label_indices) // 2]  # Middle sample
        sw = all_subwindows[sample_idx]
        result = ripser(sw, maxdim=MAX_HOMOLOGY_DIM)

        ax = axes[idx]
        colors = ["#2196F3", "#FF5722"]
        labels_h = ["H0", "H1"]

        for dim_idx, dgm in enumerate(result["dgms"]):
            finite_mask = np.isfinite(dgm[:, 1])
            dgm_finite = dgm[finite_mask]
            if len(dgm_finite) > 0:
                ax.scatter(dgm_finite[:, 0], dgm_finite[:, 1],
                           c=colors[dim_idx], label=labels_h[dim_idx],
                           alpha=0.7, s=40, edgecolors="white", linewidths=0.5)

        # Diagonal
        lims = ax.get_xlim()
        ax.plot(lims, lims, "--", color="gray", alpha=0.5, linewidth=1)
        ax.set_xlabel("Birth", fontsize=11)
        ax.set_ylabel("Death", fontsize=11)
        ax.set_title(f"{label}", fontsize=13, fontweight="bold")
        ax.legend(fontsize=10)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "persistence_diagrams.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("      Saved: persistence_diagrams.png")

    # --- Plot 2: Confusion Matrix (grouped CV) ---
    fig, ax = plt.subplots(figsize=(8, 6))
    cm = confusion_matrix(y_encoded, y_pred_cv)
    disp = ConfusionMatrixDisplay(cm, display_labels=le.classes_)
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title(f"Confusion Matrix - {best_model_name}\n(StratifiedGroupKFold {N_FOLDS}-Fold CV)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("      Saved: confusion_matrix.png")

    # --- Plot 3: Feature Distributions by Class ---
    fig, axes = plt.subplots(2, 7, figsize=(28, 8))
    fig.suptitle("Topological Feature Distributions by Class", fontsize=14, fontweight="bold")

    for i, fname in enumerate(feat_names):
        row = i // 7
        col = i % 7
        ax = axes[row, col]

        for label in le.classes_:
            mask = y == label
            vals = X[mask, i]
            ax.hist(vals, bins=30, alpha=0.5, label=label, density=True)

        ax.set_title(fname, fontsize=9, fontweight="bold")
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "feature_distributions.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("      Saved: feature_distributions.png")

    # --- Plot 4: Model Comparison (Leaky vs Grouped, side by side) ---
    fig, axes = plt.subplots(1, 2, figsize=(20, 6), sharey=True)
    model_names = list(results_grouped.keys())

    for ax_idx, (res, title_suffix) in enumerate([
        (results_leaky, "StratifiedKFold (LEAKY)"),
        (results_grouped, "StratifiedGroupKFold (NO LEAKAGE)")
    ]):
        ax = axes[ax_idx]
        accs = [res[n]["accuracy"] for n in model_names]
        f1s = [res[n]["f1_macro"] for n in model_names]
        acc_stds = [res[n]["accuracy_std"] for n in model_names]
        f1_stds = [res[n]["f1_std"] for n in model_names]

        x = np.arange(len(model_names))
        width = 0.35

        bars1 = ax.bar(x - width/2, accs, width, yerr=acc_stds, label="Accuracy",
                       color="#2196F3", capsize=4, alpha=0.85)
        bars2 = ax.bar(x + width/2, f1s, width, yerr=f1_stds, label="F1-Macro",
                       color="#FF9800", capsize=4, alpha=0.85)

        ax.set_ylabel("Score", fontsize=12)
        ax.set_title(f"Model Comparison - TDA Features\n({title_suffix})",
                     fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, fontsize=9, rotation=10, ha="right")
        ax.legend(fontsize=10)
        ax.set_ylim(0, 1.15)
        ax.grid(axis="y", alpha=0.3)

        for bar in bars1:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.02,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        for bar in bars2:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.02,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "model_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("      Saved: model_comparison.png")

    # --- Plot 5: Feature Correlation Heatmap ---
    fig, ax = plt.subplots(figsize=(12, 10))
    corr_matrix = np.corrcoef(X.T)
    sns.heatmap(corr_matrix, xticklabels=feat_names, yticklabels=feat_names,
                annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                ax=ax, vmin=-1, vmax=1, square=True,
                annot_kws={"size": 7})
    ax.set_title("Feature Correlation Heatmap", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "correlation_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("      Saved: correlation_heatmap.png")

    # --- Summary: Leaky vs Grouped comparison ---
    elapsed = time.time() - t_start
    print("\n" + "=" * 90)
    print("  RESULTS COMPARISON: StratifiedKFold (LEAKY) vs StratifiedGroupKFold (GROUPED)")
    print("=" * 90)
    print(f"\n  Data: {len(all_subwindows)} subwindows, {X_final.shape[1]} final features")
    print(f"  Unique groups: {len(np.unique(group_id_array))}")
    print(f"\n  {'Model':<25s}  {'Acc(leaky)':>11s}  {'Acc(grouped)':>13s}  {'Diff':>7s}  {'F1(leaky)':>10s}  {'F1(grouped)':>12s}  {'Diff':>7s}")
    print(f"  {'-'*25}  {'-'*11}  {'-'*13}  {'-'*7}  {'-'*10}  {'-'*12}  {'-'*7}")
    for name in results_grouped:
        rl = results_leaky[name]
        rg = results_grouped[name]
        acc_diff = rg["accuracy"] - rl["accuracy"]
        f1_diff = rg["f1_macro"] - rl["f1_macro"]
        marker = " <<<" if name == best_model_name else ""
        print(f"  {name:<25s}  {rl['accuracy']:>11.4f}  {rg['accuracy']:>13.4f}  {acc_diff:>+7.4f}  {rl['f1_macro']:>10.4f}  {rg['f1_macro']:>12.4f}  {f1_diff:>+7.4f}{marker}")
    print(f"\n  Best (grouped): {best_model_name} (Accuracy: {best_acc:.4f})")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Output: {OUTPUT_DIR}/")
    print("=" * 90)

    # Detailed classification report for best model (grouped CV)
    print(f"\n  Classification Report ({best_model_name}, StratifiedGroupKFold):")
    print(classification_report(y_encoded, y_pred_cv, target_names=le.classes_))

    # Save results to CSV — both leaky and grouped columns
    rows = []
    for name in results_grouped:
        rl = results_leaky[name]
        rg = results_grouped[name]
        rows.append({
            "Model": name,
            "accuracy_leaky": rl["accuracy"],
            "accuracy_std_leaky": rl["accuracy_std"],
            "f1_macro_leaky": rl["f1_macro"],
            "f1_std_leaky": rl["f1_std"],
            "accuracy_grouped": rg["accuracy"],
            "accuracy_std_grouped": rg["accuracy_std"],
            "f1_macro_grouped": rg["f1_macro"],
            "f1_std_grouped": rg["f1_std"],
            "accuracy_diff": rg["accuracy"] - rl["accuracy"],
            "f1_diff": rg["f1_macro"] - rl["f1_macro"],
        })
    results_df = pd.DataFrame(rows).set_index("Model")
    results_df.to_csv(os.path.join(OUTPUT_DIR, "classification_results.csv"))
    print(f"  Results saved to {OUTPUT_DIR}/classification_results.csv")


if __name__ == "__main__":
    main()
