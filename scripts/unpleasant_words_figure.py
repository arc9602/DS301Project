
SENTIMENT_CSV = r"C:\Users\adith\OneDrive\Desktop\Assignments\DS301\sentiment_scores.csv"   
EXTRACTED_CSV = r"C:\Users\adith\OneDrive\Desktop\Assignments\DS301\extracted.csv"          
OUTPUT_PATH   = r"C:\Users\adith\OneDrive\Desktop\Assignments\DS301\eda figures\unpleasant_words_figure_2.png"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def fit_and_sweep(data, feat_col, ctrl_cols, label_col, n_boot=500, panel_name=""):
    sub   = data[[feat_col] + ctrl_cols + [label_col]].dropna()
    X_raw = sub[[feat_col] + ctrl_cols].values
    y     = sub[label_col].values

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_scaled, y)

    feat_vals = sub[feat_col].values
    sweep = np.linspace(np.percentile(feat_vals, 5),
                                    np.percentile(feat_vals, 95), 200)
    X_sweep_raw = np.zeros((200, X_raw.shape[1]))
    X_sweep_raw[:, 0] = sweep
    X_sweep = scaler.transform(X_sweep_raw)
    probs = model.predict_proba(X_sweep)[:, 1]

    boot_probs = np.zeros((n_boot, 200))
    rng = np.random.default_rng(42)
    print(f"  Bootstrapping '{panel_name}' ({n_boot} resamples)...", flush=True)
    for b in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        m = LogisticRegression(max_iter=300, random_state=0)
        try:
            m.fit(X_scaled[idx], y[idx])
            boot_probs[b] = m.predict_proba(X_sweep)[:, 1]
        except Exception:
            boot_probs[b] = probs

    ci_low = np.percentile(boot_probs, 2.5,  axis=0)
    ci_high = np.percentile(boot_probs, 97.5, axis=0)
    return sweep, probs, ci_low, ci_high


def draw_panel(ax, sweep, probs, ci_low, ci_high, title):
    ax.plot(sweep, probs, color="black", linewidth=2)
    ax.fill_between(sweep, ci_low, ci_high, color="grey", alpha=0.35)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel(
        "Percent More Unpleasant Words Directed at Petitioner\n"
        "(VADER negativity: petitioner side minus respondent side)",
        fontsize=9,
    )
    ax.set_ylabel("Probability Petitioner Wins")
    ax.set_ylim(0.25, 0.85)
    ax.axvline(x=0, color="black", linestyle=":", linewidth=1, alpha=0.5)

    mid = len(sweep) // 2
    d_sweep = sweep[mid + 5]  - sweep[mid - 5]
    marginal = (probs[mid+5]   - probs[mid-5])   / d_sweep
    ci_m_low = (ci_low[mid+5]  - ci_low[mid-5])  / d_sweep
    ci_m_high = (ci_high[mid+5] - ci_high[mid-5]) / d_sweep

    ax.text(
        0.97, 0.97,
        f"Marginal Effect at Mean:\n{marginal:+.4f}  [{ci_m_low:+.4f}, {ci_m_high:+.4f}]",
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="grey"),
    )


def main():
    output_path = Path(OUTPUT_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading sentiment scores...")
    sent = pd.read_csv(SENTIMENT_CSV)
    print(f"Loading extracted features...")
    feat = pd.read_csv(EXTRACTED_CSV, usecols=[
        "case_id", "justice_id", "total_words",
        "word_ratio_0_to_1", "question_ratio_0_to_1"
    ])

    df = sent.merge(feat, on=["case_id", "justice_id"], how="left")
    print(f"Merged: {len(df):,} rows.")

    df = df[df["label"].isin([0, 1])].copy()
    df_both = df[(df["n_utt_to_petitioner"] > 0) &
                 (df["n_utt_to_respondent"] > 0)].copy()
    print(f"Rows where justice questioned both sides: {len(df_both):,}")

    # case-level (left panel)
    case_df = (
        df_both.groupby("case_id")
               .agg(
                   unpleasant_diff=("unpleasant_diff",   "mean"),
                   neg_to_pet     =("neg_to_petitioner", "mean"),
                   neg_to_res     =("neg_to_respondent", "mean"),
                   label          =("win_side",           "first"),
               )
               .dropna()
    )
    case_df["label"] = case_df["label"].astype(int)
    case_df = case_df[case_df["label"].isin([0, 1])]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    sweep, probs, ci_low, ci_high = fit_and_sweep(
        case_df, "unpleasant_diff", ["neg_to_pet", "neg_to_res"],
        "label", panel_name="Court Outcome")
    draw_panel(axes[0], sweep, probs, ci_low, ci_high, "Court Outcome")

    sweep, probs, ci_low, ci_high = fit_and_sweep(
        df_both, "unpleasant_diff", ["neg_to_petitioner", "neg_to_respondent"],
        "label", panel_name="Justice Votes")
    draw_panel(axes[1], sweep, probs, ci_low, ci_high, "Justice Votes")

    fig.suptitle(
        "Effect of Unpleasant Language Directed at Petitioner on Win Probability\n"
        "(shaded = 95% CI via bootstrap, 500 resamples)",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved to {output_path}")
    plt.show()


if __name__ == "__main__":
    main()
