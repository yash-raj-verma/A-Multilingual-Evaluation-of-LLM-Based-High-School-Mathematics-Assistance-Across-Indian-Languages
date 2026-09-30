"""
bootstrap_ci.py

Computes bootstrap 95% confidence intervals over the robust equivalence
results produced by robust_equivalence_scoring.py.

Outputs a markdown-formatted table of CIs by model, language, and topic,
suitable for inclusion in a paper or rebuttal.

Usage:
    python bootstrap_ci.py \
        --results_csv /path/to/robust_equiv_results.csv \
        --n_boot 2000
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

LANG_ORDER = ["english", "hindi", "bengali", "urdu", "gujrati", "malayalam"]


def bootstrap_ci(arr, n_boot: int = 2000, seed: int = 42):
    """Return (lower, upper) bootstrap 95% CI as percentages."""
    rng  = np.random.default_rng(seed)
    arr  = np.array(arr, dtype=float)
    means = [
        rng.choice(arr, size=len(arr), replace=True).mean()
        for _ in range(n_boot)
    ]
    lo, hi = np.percentile(means, [2.5, 97.5])
    return lo * 100, hi * 100


def ci_table(df: pd.DataFrame, group_col: str, order=None) -> pd.DataFrame:
    rows = []
    groups = order if order else df[group_col].unique()
    for val in groups:
        sub = df[df[group_col] == val]
        if sub.empty:
            continue
        exact = sub['exact'].mean() * 100
        equiv = sub['robust_equiv'].mean() * 100
        lo_e, hi_e = bootstrap_ci(sub['exact'].values)
        lo_q, hi_q = bootstrap_ci(sub['robust_equiv'].values)
        rows.append({
            group_col:          val,
            'exact_acc':        f"{exact:.1f}%",
            'exact_95ci':       f"[{lo_e:.1f}, {hi_e:.1f}]",
            'robust_equiv_acc': f"{equiv:.1f}%",
            'robust_95ci':      f"[{lo_q:.1f}, {hi_q:.1f}]",
            'gain':             f"+{equiv - exact:.1f} pp",
        })
    return pd.DataFrame(rows)


def to_markdown(df: pd.DataFrame) -> str:
    cols = df.columns.tolist()
    header = "| " + " | ".join(cols) + " |"
    sep    = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows   = [
        "| " + " | ".join(str(v) for v in row) + " |"
        for row in df.itertuples(index=False)
    ]
    return "\n".join([header, sep] + rows)


def main():
    parser = argparse.ArgumentParser(
        description="Compute bootstrap CIs from robust equivalence results."
    )
    parser.add_argument(
        '--results_csv',
        type=Path,
        required=True,
        help="Path to robust_equiv_results.csv from robust_equivalence_scoring.py",
    )
    parser.add_argument(
        '--n_boot',
        type=int,
        default=2000,
        help="Number of bootstrap resamples (default: 2000).",
    )
    parser.add_argument(
        '--output_md',
        type=Path,
        default=None,
        help="Optional path to write markdown tables.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.results_csv)
    print(f"Loaded {len(df)} records from {args.results_csv}\n")

    out_lines = []

    # By model
    print("=== Bootstrap CIs by Model ===")
    model_table = ci_table(df, 'model')
    md = to_markdown(model_table)
    print(md); print()
    out_lines += ["## By Model", md, ""]

    # By language
    print("=== Bootstrap CIs by Language ===")
    lang_table = ci_table(df, 'language', order=LANG_ORDER)
    md = to_markdown(lang_table)
    print(md); print()
    out_lines += ["## By Language", md, ""]

    # By topic
    print("=== Bootstrap CIs by Topic ===")
    topic_table = ci_table(df, 'topic')
    topic_table = topic_table.sort_values('robust_equiv_acc', ascending=False)
    md = to_markdown(topic_table)
    print(md); print()
    out_lines += ["## By Topic", md, ""]

    # English vs Malayalam significance check
    en = df[df['language'] == 'english']['robust_equiv'].values
    ml = df[df['language'] == 'malayalam']['robust_equiv'].values
    en_lo, en_hi = bootstrap_ci(en, n_boot=args.n_boot)
    ml_lo, ml_hi = bootstrap_ci(ml, n_boot=args.n_boot)
    sig_msg = (
        f"English CI: [{en_lo:.1f}, {en_hi:.1f}]  |  "
        f"Malayalam CI: [{ml_lo:.1f}, {ml_hi:.1f}]  |  "
        f"Non-overlapping: {en_lo > ml_hi}"
    )
    print("=== English vs Malayalam Gap ===")
    print(sig_msg)
    out_lines += ["## English vs Malayalam Gap", sig_msg]

    if args.output_md:
        args.output_md.write_text("\n".join(out_lines))
        print(f"\nMarkdown tables written to: {args.output_md}")


if __name__ == '__main__':
    main()
