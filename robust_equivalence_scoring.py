"""
robust_equivalence_scoring.py

Computes three-layer robust equivalence-aware accuracy over all Trial-1 results.

Layers:
    1. Numeric extraction with comma normalisation and fraction handling
    2. Symbolic equation solving (e.g. x + 8 = 0 vs -8)
    3. SymPy algebraic simplification

Outputs:
    - robust_equiv_results.csv  : per-record results
    - summary_by_model.csv      : accuracy by model
    - summary_by_language.csv   : accuracy by language
    - summary_by_topic.csv      : accuracy by topic
    - format_failure_table.csv  : format failure rates by model x language

Usage:
    python robust_equivalence_scoring.py \
        --data_dir /path/to/Supplementary_Materials \
        --output_dir /path/to/output

Requirements:
    pip install sympy pandas numpy
"""

import argparse
import re
import pandas as pd
import numpy as np
from pathlib import Path
from sympy import symbols, solve, simplify
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
)

# ── Constants ──────────────────────────────────────────────────────────────────

MODEL_MAP = {
    "GPT-4o":    "GPT-4o mini",
    "LLaMA 3":   "LLaMA 3-8B",
    "DeepSeek-R1": "DeepSeek-R1",
}

TOPIC_MAP = {
    "Conic":      "Conic Sections",
    "Limits":     "Limits & Derivatives",
    "PnC":        "Permutations & Combinations",
    "SL":         "Straight Lines",
    "SeqnSeries": "Sequences & Series",
}

STRAT_MAP = {
    "One_Shot":        "Zero-Shot",
    "One_Shot_Subcat": "Zero-Shot+Topic",
    "CoT":             "Self-Ask",
}

LANG_ORDER = ["english", "hindi", "bengali", "urdu", "gujrati", "malayalam"]

SYMPY_TRANSFORMS = standard_transformations + (implicit_multiplication_application,)

# ── Helper functions ───────────────────────────────────────────────────────────

def extract_final_answer(text: str) -> str:
    """Extract the final answer portion from a verbose model response."""
    text = str(text)
    # LaTeX boxed answers
    boxes = re.findall(r'\\boxed\{([^}]+)\}', text)
    if boxes:
        return boxes[-1]
    # Explicit Final Answer label
    fa = re.findall(r'[Ff]inal [Aa]nswer[:\s"*`]*([^\n"}{`]{1,200})', text)
    if fa:
        return fa[-1].strip().strip('"').strip("'")
    # JSON-style answer field
    ja = re.findall(r'"[Ff]inal [Aa]nswer"\s*:\s*"([^"]+)"', text)
    if ja:
        return ja[-1]
    # Last short line containing a digit
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    for line in reversed(lines):
        if re.search(r'\d', line) and len(line) < 100:
            return line
    return text[-200:] if len(text) > 200 else text


def extract_number(text: str):
    """Extract a numeric value from text, handling commas and fractions."""
    text = re.sub(r',', '', str(text))
    nums = re.findall(r'-?\d+\.?\d*(?:/\d+)?', text)
    if nums:
        try:
            n = nums[0]
            if '/' in n:
                p, q = n.split('/')
                return float(p) / float(q)
            return float(n)
        except (ValueError, ZeroDivisionError):
            pass
    return None


def solve_equation(text: str):
    """
    If text contains an equation (has '='), attempt to solve for x
    and return the numeric solution (or None).
    """
    text = str(text).strip()
    if '=' not in text:
        return None
    try:
        cleaned = re.sub(r'\\[a-zA-Z]+', '', text)
        cleaned = cleaned.replace('^', '**').replace('x', 'x')
        lhs, rhs = cleaned.split('=', 1)
        x = symbols('x')
        expr = parse_expr(
            f"({lhs}) - ({rhs})",
            transformations=SYMPY_TRANSFORMS,
        )
        sols = solve(expr, x)
        if sols and len(sols) == 1:
            return float(sols[0])
    except Exception:
        pass
    return None


def clean_for_sympy(text: str) -> str:
    """Prepare an expression string for SymPy parsing."""
    text = re.sub(r'\\boxed\{([^}]+)\}', r'\1', str(text))
    text = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'(\1)/(\2)', text)
    text = re.sub(r'\\[a-zA-Z]+', '', text)
    text = text.replace('^', '**').replace('x', 'x').replace(',', '')
    if '=' in text:
        lhs, rhs = text.split('=', 1)
        text = f"({lhs}) - ({rhs})"
    return text.strip()


def robust_equiv_check(pred_raw: str, gt_raw: str):
    """
    Three-layer equivalence check.

    Returns:
        (is_equivalent: bool, method: str)
    """
    pred = str(pred_raw).strip()
    gt   = str(gt_raw).strip()

    # Layer 0: exact string containment (already captured in 'exact' column)
    if gt.lower() in pred.lower():
        return True, 'exact'

    # Layer 1: numeric extraction
    pred_num = extract_number(pred)
    gt_num   = extract_number(gt)
    if pred_num is not None and gt_num is not None:
        if gt_num != 0 and abs(pred_num - gt_num) / abs(gt_num) < 0.001:
            return True, 'numeric'
        if gt_num == 0 and abs(pred_num - gt_num) < 1e-4:
            return True, 'numeric'

    # Layer 2: equation solving
    pred_sol = solve_equation(pred)
    if pred_sol is not None and gt_num is not None:
        if abs(pred_sol - gt_num) < 0.001:
            return True, 'eq_solved'
    gt_sol = solve_equation(gt)
    if gt_sol is not None and pred_num is not None:
        if abs(gt_sol - pred_num) < 0.001:
            return True, 'eq_solved'

    # Layer 3: SymPy symbolic simplification
    try:
        pe = parse_expr(clean_for_sympy(pred), transformations=SYMPY_TRANSFORMS)
        ge = parse_expr(clean_for_sympy(gt),   transformations=SYMPY_TRANSFORMS)
        if simplify(pe - ge) == 0:
            return True, 'sympy'
    except Exception:
        pass

    return False, 'none'


def is_format_failure(text: str) -> bool:
    """Detect pure format / truncation failures."""
    if pd.isna(text):
        return True
    t = str(text).strip()
    if len(t) < 5:
        return True
    if re.match(r'^numeric answer$', t, re.IGNORECASE):
        return True
    return False


# ── Data loading ───────────────────────────────────────────────────────────────

def load_all_results(data_dir: Path) -> pd.DataFrame:
    """
    Walk the Supplementary_Materials folder structure and load all
    *_trials.csv files (excluding CoT_Choices and lock files).
    """
    records = []

    for model_folder, model_name in MODEL_MAP.items():
        model_path = data_dir / model_folder
        if not model_path.exists():
            continue

        for topic_folder, topic_name in TOPIC_MAP.items():
            topic_path = model_path / topic_folder
            if not topic_path.exists():
                continue

            for strat_folder, strat_name in STRAT_MAP.items():
                strat_path = topic_path / strat_folder
                if not strat_path.exists():
                    continue

                for lang in LANG_ORDER:
                    csvs = [
                        c for c in strat_path.glob(f"*{lang}*trials.csv")
                        if "Choices" not in c.name and "~lock" not in c.name
                    ]
                    if not csvs:
                        continue

                    try:
                        df = pd.read_csv(csvs[0])
                        sol_col = (
                            'Sol'      if 'Sol'      in df.columns else
                            'Solution' if 'Solution' in df.columns else
                            None
                        )
                        if sol_col is None:
                            continue

                        for _, row in df.iterrows():
                            gt  = str(row[sol_col]).strip()
                            t1  = str(row.get('trial1', '')).strip()
                            t1_ans = extract_final_answer(t1)

                            exact   = gt.lower() in t1.lower() or t1.lower() == gt.lower()
                            fmt     = is_format_failure(t1)
                            equiv, method = robust_equiv_check(t1_ans, gt)

                            records.append({
                                'model':         model_name,
                                'topic':         topic_name,
                                'strategy':      strat_name,
                                'language':      lang,
                                'ground_truth':  gt,
                                'extracted_ans': t1_ans[:200],
                                'exact':         int(exact),
                                'fmt_fail':      int(fmt),
                                'robust_equiv':  int(equiv or exact),
                                'equiv_method':  method if not exact else 'exact',
                            })

                    except Exception as e:
                        print(f"Warning: could not process {csvs[0]}: {e}")

    return pd.DataFrame(records)


# ── Bootstrap CI ───────────────────────────────────────────────────────────────

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


# ── Summary tables ─────────────────────────────────────────────────────────────

def summary_by_group(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []
    for val in df[group_col].unique():
        sub = df[df[group_col] == val]
        exact = sub['exact'].mean() * 100
        equiv = sub['robust_equiv'].mean() * 100
        fmt   = sub['fmt_fail'].mean() * 100
        lo_e, hi_e = bootstrap_ci(sub['exact'].values)
        lo_q, hi_q = bootstrap_ci(sub['robust_equiv'].values)
        rows.append({
            group_col:          val,
            'exact_acc':        round(exact, 1),
            'exact_ci_lo':      round(lo_e, 1),
            'exact_ci_hi':      round(hi_e, 1),
            'robust_equiv_acc': round(equiv, 1),
            'equiv_ci_lo':      round(lo_q, 1),
            'equiv_ci_hi':      round(hi_q, 1),
            'gain_pp':          round(equiv - exact, 1),
            'format_fail_pct':  round(fmt, 1),
        })
    return pd.DataFrame(rows)


def format_failure_table(df: pd.DataFrame) -> pd.DataFrame:
    pivot = (
        df.groupby(['model', 'language'])['fmt_fail']
        .mean()
        .unstack()
        * 100
    )
    return pivot[LANG_ORDER].round(1)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Robust equivalence-aware scoring for multilingual math evaluation."
    )
    parser.add_argument(
        '--data_dir',
        type=Path,
        required=True,
        help="Path to the Supplementary_Materials folder.",
    )
    parser.add_argument(
        '--output_dir',
        type=Path,
        default=Path('./output'),
        help="Directory to write output CSVs.",
    )
    parser.add_argument(
        '--n_boot',
        type=int,
        default=2000,
        help="Number of bootstrap resamples for CIs (default: 2000).",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading results...")
    df = load_all_results(args.data_dir)
    print(f"  Total records: {len(df)}")

    # Save full per-record results
    out_full = args.output_dir / 'robust_equiv_results.csv'
    df.to_csv(out_full, index=False)
    print(f"  Saved: {out_full}")

    # Equivalence method breakdown
    print("\nEquivalence method breakdown:")
    print(df[df['robust_equiv'] == 1]['equiv_method'].value_counts().to_string())

    # Summary by model
    print("\nSummary by model:")
    model_summary = summary_by_group(df, 'model')
    print(model_summary.to_string(index=False))
    model_summary.to_csv(args.output_dir / 'summary_by_model.csv', index=False)

    # Summary by language
    print("\nSummary by language:")
    lang_summary = summary_by_group(df, 'language')
    # Reorder by LANG_ORDER
    lang_summary['_order'] = lang_summary['language'].map(
        {l: i for i, l in enumerate(LANG_ORDER)}
    )
    lang_summary = lang_summary.sort_values('_order').drop(columns='_order')
    print(lang_summary.to_string(index=False))
    lang_summary.to_csv(args.output_dir / 'summary_by_language.csv', index=False)

    # Summary by topic
    print("\nSummary by topic:")
    topic_summary = summary_by_group(df, 'topic')
    topic_summary = topic_summary.sort_values('robust_equiv_acc', ascending=False)
    print(topic_summary.to_string(index=False))
    topic_summary.to_csv(args.output_dir / 'summary_by_topic.csv', index=False)

    # Format failure table
    print("\nFormat failure rate by model x language (%):")
    fmt_table = format_failure_table(df)
    print(fmt_table.to_string())
    fmt_table.to_csv(args.output_dir / 'format_failure_table.csv')

    # English vs Malayalam CI check
    en = df[df['language'] == 'english']['robust_equiv'].values
    ml = df[df['language'] == 'malayalam']['robust_equiv'].values
    en_lo, en_hi = bootstrap_ci(en, n_boot=args.n_boot)
    ml_lo, ml_hi = bootstrap_ci(ml, n_boot=args.n_boot)
    print(f"\nEnglish CI:   [{en_lo:.1f}, {en_hi:.1f}]")
    print(f"Malayalam CI: [{ml_lo:.1f}, {ml_hi:.1f}]")
    print(f"Non-overlapping (EN > ML): {en_lo > ml_hi}")

    print(f"\nDone. All outputs written to: {args.output_dir}")


if __name__ == '__main__':
    main()
