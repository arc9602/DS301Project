"""
eda.py

Exploratory Data Analysis + visualizations for the Super-SCOTUS dataset.
Run AFTER extract_data.py has produced the extracted CSV.

Usage:
    python eda.py --input data/extracted.csv --output_dir figures/
"""

from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats

# ── style ─────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.15)
PALETTE = {"0": "#4C72B0", "1": "#DD8452"}
FIG_DIR = Path("figures")

ISSUE_AREA_MAP = {
    1: "Criminal Procedure", 2: "Civil Rights", 3: "First Amendment",
    4: "Due Process",        5: "Privacy",      6: "Attorneys",
    7: "Unions",             8: "Economic",     9: "Judicial Power",
    10: "Federalism",        11: "Interstate",  12: "Federal Taxation",
    13: "Miscellaneous",     14: "Private Law",
}

def save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  saved → {path}")
    plt.close(fig)


# ── individual plots ───────────────────────────────────────────────────────────

def plot_label_distribution(df):
    """How balanced is the vote label?"""
    counts = df["label"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(["Side 0 (Respondent)", "Side 1 (Petitioner)"],
                  counts.values, color=["#4C72B0", "#DD8452"], width=0.5)
    for bar, v in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 80,
                f"{v:,}\n({v/len(df)*100:.1f}%)", ha="center", va="bottom", fontsize=11)
    ax.set_title("Vote Label Distribution\n(per justice × case pair)", fontsize=13)
    ax.set_ylabel("Count")
    ax.set_ylim(0, counts.max() * 1.18)
    save(fig, "01_label_distribution.png")


def plot_cases_per_year(df):
    """How many cases (unique) per year?"""
    yearly = df.drop_duplicates("case_id").groupby("year").size()
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.fill_between(yearly.index, yearly.values, alpha=0.3, color="#4C72B0")
    ax.plot(yearly.index, yearly.values, color="#4C72B0", linewidth=1.8)
    ax.set_title("Number of Cases per Term Year", fontsize=13)
    ax.set_xlabel("Year")
    ax.set_ylabel("Cases")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
    save(fig, "02_cases_per_year.png")


def plot_words_per_justice(df):
    """Top 20 most talkative justices (median total words)."""
    med = (df.groupby("justice_id")["total_words"]
             .median()
             .sort_values(ascending=False)
             .head(20))
    labels = [j.replace("j__", "").replace("_", " ").title() for j in med.index]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(labels[::-1], med.values[::-1], color="#4C72B0")
    ax.set_xlabel("Median Words per Oral Argument")
    ax.set_title("Most Active Justices\n(median words spoken per case)", fontsize=13)
    save(fig, "03_words_per_justice.png")


def plot_word_asymmetry_vs_vote(df):
    """
    Key predictive feature: do justices speak more words to the side they vote against?
    word_ratio_0_to_1 > 1  → more words to side 0
    """
    df2 = df.copy()
    df2["log_ratio"] = np.log1p(df2["word_ratio_0_to_1"])
    df2["vote_label"] = df2["label"].map({0: "Voted Side 0", 1: "Voted Side 1"})

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # violin
    sns.violinplot(data=df2, x="vote_label", y="log_ratio",
                   palette=["#4C72B0", "#DD8452"], inner="quartile", ax=axes[0])
    axes[0].axhline(0, linestyle="--", color="grey", linewidth=1)
    axes[0].set_title("Word Ratio (side 0 / side 1)\nvs Justice Vote", fontsize=12)
    axes[0].set_xlabel("")
    axes[0].set_ylabel("log(word ratio + 1)")

    # box
    sns.boxplot(data=df2, x="vote_label", y="log_ratio",
                palette=["#4C72B0", "#DD8452"], width=0.4, ax=axes[1])
    axes[1].axhline(0, linestyle="--", color="grey", linewidth=1)
    axes[1].set_title("Same — Boxplot View", fontsize=12)
    axes[1].set_xlabel("")
    axes[1].set_ylabel("")

    fig.suptitle("Justices tend to direct more words at the side they vote AGAINST",
                 fontsize=13, y=1.02)
    save(fig, "04_word_asymmetry_vs_vote.png")


def plot_question_asymmetry_vs_vote(df):
    """Same analysis but for question counts."""
    df2 = df.copy()
    df2["log_q_ratio"] = np.log1p(df2["question_ratio_0_to_1"])
    df2["vote_label"] = df2["label"].map({0: "Voted Side 0", 1: "Voted Side 1"})

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.violinplot(data=df2, x="vote_label", y="log_q_ratio",
                   palette=["#4C72B0", "#DD8452"], inner="quartile", ax=ax)
    ax.axhline(0, linestyle="--", color="grey", linewidth=1)
    ax.set_title("Question Count Ratio (side 0 / side 1)\nvs Justice Vote", fontsize=13)
    ax.set_xlabel("")
    ax.set_ylabel("log(question ratio + 1)")
    save(fig, "05_question_asymmetry_vs_vote.png")


def plot_issue_area_distribution(df):
    """What types of cases are in the dataset?"""
    cases = df.drop_duplicates("case_id").copy()
    cases["issue_label"] = cases["issue_area"].map(ISSUE_AREA_MAP).fillna("Unknown")
    counts = cases["issue_label"].value_counts()

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = sns.color_palette("muted", len(counts))
    ax.barh(counts.index[::-1], counts.values[::-1], color=colors[::-1])
    ax.set_xlabel("Number of Cases")
    ax.set_title("Cases by Issue Area", fontsize=13)
    save(fig, "06_issue_area_distribution.png")


def plot_vote_rate_by_issue(df):
    """Does the petitioner win more in certain issue areas?"""
    df2 = df.drop_duplicates("case_id").copy()
    df2["issue_label"] = df2["issue_area"].map(ISSUE_AREA_MAP).fillna("Unknown")
    df2["petitioner_won"] = (df2["win_side"] == 1).astype(int)
    rate = (df2.groupby("issue_label")["petitioner_won"]
              .agg(["mean", "count"])
              .query("count >= 20")
              .sort_values("mean"))

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#DD8452" if v > 0.5 else "#4C72B0" for v in rate["mean"]]
    ax.barh(rate.index, rate["mean"], color=colors)
    ax.axvline(0.5, linestyle="--", color="grey")
    ax.set_xlabel("Petitioner Win Rate")
    ax.set_title("Petitioner Win Rate by Issue Area\n(cases with ≥20 observations)", fontsize=13)
    ax.set_xlim(0, 1)
    save(fig, "07_win_rate_by_issue.png")


def plot_utterance_length_distribution(df):
    """Distribution of words-per-argument across justices."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(df["total_words"].clip(upper=3000), bins=60,
            color="#4C72B0", edgecolor="white", linewidth=0.4)
    ax.set_xlabel("Total Words Spoken per Oral Argument")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Justice Word Count\nper Oral Argument", fontsize=13)
    median_val = df["total_words"].median()
    ax.axvline(median_val, color="#DD8452", linestyle="--",
               label=f"Median = {median_val:.0f}")
    ax.legend()
    save(fig, "08_utterance_length_distribution.png")


def plot_justice_vote_heatmap(df):
    """
    Heatmap: for each justice, what fraction of the time do they vote side 1?
    (Shows ideological leanings encoded in the data.)
    """
    top_justices = (df.groupby("justice_id").size()
                      .sort_values(ascending=False)
                      .head(25).index)
    sub = df[df["justice_id"].isin(top_justices)]
    vote_rate = sub.groupby("justice_id")["label"].mean().sort_values()
    labels = [j.replace("j__", "").replace("_", " ").title() for j in vote_rate.index]

    fig, ax = plt.subplots(figsize=(3.5, 9))
    hm_data = vote_rate.values.reshape(-1, 1)
    im = ax.imshow(hm_data, cmap="RdBu", vmin=0.3, vmax=0.7, aspect="auto")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xticks([])
    ax.set_title("Fraction of Votes\nfor Side 1 (Petitioner)\nby Justice", fontsize=11)
    plt.colorbar(im, ax=ax, fraction=0.08, pad=0.04)
    save(fig, "09_justice_vote_heatmap.png")


def plot_correlation_matrix(df):
    """Correlation between numeric features and the label."""
    feat_cols = [
        "total_words", "total_utterances",
        "words_to_side0", "words_to_side1", "word_ratio_0_to_1",
        "questions_to_side0", "questions_to_side1", "question_ratio_0_to_1",
        "interruptions", "label"
    ]
    sub = df[feat_cols].dropna()
    corr = sub.corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, linewidths=0.5, ax=ax, annot_kws={"size": 9})
    ax.set_title("Feature Correlation Matrix", fontsize=13)
    save(fig, "10_correlation_matrix.png")


def plot_words_over_time(df):
    """Has oral argument become more or less verbal over time?"""
    yearly = df.groupby("year")["total_words"].median()
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(yearly.index, yearly.values, color="#4C72B0", linewidth=2)
    ax.fill_between(yearly.index, yearly.values, alpha=0.15, color="#4C72B0")
    ax.set_title("Median Justice Word Count per Argument Over Time", fontsize=13)
    ax.set_xlabel("Year")
    ax.set_ylabel("Median Words")
    save(fig, "11_words_over_time.png")


def print_summary_stats(df):
    print("\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)
    print(f"Total (case × justice) rows : {len(df):,}")
    print(f"Unique cases                : {df['case_id'].nunique():,}")
    print(f"Unique justices             : {df['justice_id'].nunique():,}")
    print(f"Year range                  : {df['year'].min()} – {df['year'].max()}")
    print(f"Label balance (side 1 %)    : {df['label'].mean()*100:.1f}%")
    print(f"\nMedian words per argument   : {df['total_words'].median():.0f}")
    print(f"Median utterances           : {df['total_utterances'].median():.0f}")

    # point-biserial correlation of key features with label
    print("\nCorrelation with vote label (point-biserial r):")
    for col in ["word_ratio_0_to_1", "question_ratio_0_to_1",
                "words_to_side0", "words_to_side1", "interruptions"]:
        sub = df[[col, "label"]].dropna()
        r, p = stats.pointbiserialr(sub["label"], sub[col])
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
        print(f"  {col:<30} r={r:+.3f}  p={p:.3e} {sig}")
    print("="*60 + "\n")


# ── main ──────────────────────────────────────────────────────────────────────

def main(input_path: str, output_dir: str):
    global FIG_DIR
    FIG_DIR = Path(output_dir)

    print(f"Loading {input_path} ...")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df):,} rows.")

    # drop rows where justice did not vote (recused/absent, label == -1)
    before = len(df)
    df = df[df["label"] != -1].copy()
    dropped = before - len(df)
    if dropped:
        print(f"Dropped {dropped} rows with label=-1 (recused/absent justices).")

    print_summary_stats(df)

    print("Generating plots...")
    plot_label_distribution(df)
    plot_cases_per_year(df)
    plot_words_per_justice(df)
    plot_word_asymmetry_vs_vote(df)
    plot_question_asymmetry_vs_vote(df)
    plot_issue_area_distribution(df)
    plot_vote_rate_by_issue(df)
    plot_utterance_length_distribution(df)
    plot_justice_vote_heatmap(df)
    plot_correlation_matrix(df)
    plot_words_over_time(df)

    print(f"\nAll figures saved to {FIG_DIR}/")


# ── SET YOUR PATHS HERE ──────────────────────────────────────────────────────
INPUT_PATH  = r"C:\Users\adith\OneDrive\Desktop\Assignments\DS301\extracted.csv"   # output from extract_data.py
OUTPUT_DIR  = r"C:\Users\adith\OneDrive\Desktop\Assignments\DS301\eda figures"          # folder to save plots into
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main(INPUT_PATH, OUTPUT_DIR)