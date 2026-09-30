# A Multilingual Evaluation of LLM-Based High-School Mathematics Assistance Across Indian Languages

**Yash Kumar and Subhajit Roy**  
Indian Institute of Technology Kanpur, India  


---

## Overview

This repository contains all datasets, model inference scripts, experimental results, and analysis code for our paper. We evaluate GPT-4o mini, LLaMA 3-8B, and DeepSeek-R1 on high-school mathematics problems across six Indian languages (English, Hindi, Bengali, Urdu, Gujarati, Malayalam) under three lightweight student-inspired prompting strategies.

---

## Repository Structure

```
.
├── README.md                          ← this file
├── LICENSE                            ← MIT License (code)
├── LICENSE_DATA                       ← CC BY 4.0 (dataset)
├── robust_equivalence_scoring.py      ← three-layer equivalence checker
├── bootstrap_ci.py                    ← bootstrap 95% CI computation
│
├── Datasets/                          ← parallel problem datasets (6 languages each)
│   ├── PnC.csv
│   ├── StraightLines.csv
│   ├── SeqnSeries.csv
│   ├── LimitsnDerivatives.csv
│   └── conic_sections.csv
│
├── output/                            ← pre-computed analysis outputs
│   ├── robust_equiv_results.csv       ← per-record equivalence scoring results
│   ├── summary_by_model.csv
│   ├── summary_by_language.csv
│   ├── summary_by_topic.csv
│   ├── format_failure_table.csv
│   └── ci_tables.md                   ← bootstrap CI tables (markdown)
│
└── Supplementary_Materials/
    ├── Datasets/                      ← copy of datasets (same as above)
    ├── Survey_Results/                ← anonymized prompt elicitation responses
    │   └── AI in Education (Responses) - Form Responses.csv
    ├── GPT-4o/
    │   ├── test_openai.py             ← inference script
    │   └── [Topic]/
    │       ├── One_Shot/              ← Zero-Shot prompting results
    │       ├── One_Shot_Subcat/       ← Zero-Shot + Topic Info results
    │       └── CoT/                   ← Self-Ask prompting results
    ├── LLaMA 3/
    │   ├── llama_fast.py              ← inference script
    │   └── [Topic]/  (same structure as above)
    └── DeepSeek-R1/
        ├── deepseek_r1.py             ← inference script
        └── [Topic]/  (same structure as above)
```

**Topics:** `Conic`, `Limits`, `PnC`, `SL` (Straight Lines), `SeqnSeries`  
**Strategies:** `One_Shot` = Zero-Shot, `One_Shot_Subcat` = Zero-Shot + Topic Info, `CoT` = Self-Ask

---

## Datasets

Each CSV in `Datasets/` contains 50 parallel problems across six languages. Columns:

| Column | Description |
|---|---|
| `problem_id` | Unique identifier |
| `subcategory` | Textbook subcategory (e.g., "Slope of a line") |
| `english` / `hindi` / `bengali` / `urdu` / `gujarati` / `malayalam` | Problem text in each language |
| `solution` | Ground-truth numerical answer (manually derived from perturbed problem) |
| `original_solution` | Ground-truth for the unperturbed original problem |

Problems were extracted from NCERT/SCERT Class 11 textbooks using OCR and numerically perturbed to reduce memorization effects. All perturbed variants were manually solved and verified.

---

## Inference Scripts

### GPT-4o mini — `Supplementary_Materials/GPT-4o/test_openai.py`
Requires an OpenAI API key set as `OPENAI_API_KEY` environment variable.

```bash
export OPENAI_API_KEY=your_key_here
python Supplementary_Materials/GPT-4o/test_openai.py 
```

### LLaMA 3-8B — `Supplementary_Materials/LLaMA 3/llama_fast.py`
Requires a local LLaMA 3-8B model checkpoint.

```bash
python Supplementary_Materials/LLaMA 3/llama_fast.py 
```

### DeepSeek-R1 (7B) — `Supplementary_Materials/DeepSeek-R1/deepseek_r1.py`
Requires a local DeepSeek-R1 7B model checkpoint.

```bash
python Supplementary_Materials/DeepSeek-R1/deepseek_r1.py 
```

All scripts were run on an NVIDIA RTX A4000 GPU (16 GB). Each script queries each prompt over three independent trials from an empty context.

---

## Equivalence-Aware Scoring

Exact-match evaluation penalises mathematically correct answers that differ in symbolic form (e.g., `−8` vs `x + 8 = 0`). We provide a three-layer deterministic equivalence checker to recover these cases.

### `robust_equivalence_scoring.py`

**Three layers:**
1. Numeric extraction with comma normalisation (`60,480 = 60480`; `3/5 = 0.6`)
2. Symbolic equation solving (`x + 8 = 0` vs `−8`)
3. SymPy algebraic simplification for symbolic expressions

```bash
pip install sympy pandas numpy
python robust_equivalence_scoring.py \
    --data_dir Supplementary_Materials/ \
    --output_dir output/
```

**Outputs:**

| File | Description |
|---|---|
| `robust_equiv_results.csv` | Per-record results: exact, format-fail, robust-equiv flags |
| `summary_by_model.csv` | Accuracy aggregated by model |
| `summary_by_language.csv` | Accuracy aggregated by language |
| `summary_by_topic.csv` | Accuracy aggregated by topic |
| `format_failure_table.csv` | Format failure rates by model × language |

---

## Bootstrap Confidence Intervals

### `bootstrap_ci.py`

Computes bootstrap 95% CIs (2,000 resamples) from the equivalence scorer output.

```bash
python bootstrap_ci.py \
    --results_csv output/robust_equiv_results.csv \
    --n_boot 2000 \
    --output_md output/ci_tables.md
```

Pre-computed results are in `output/ci_tables.md`. Key findings:

**By model:**

| Model | Exact-match | Equiv.-aware | Gain |
|---|---|---|---|
| GPT-4o mini | 40.2% [37.0, 43.3] | 50.9% [47.7, 54.2] | +10.7 pp |
| DeepSeek-R1 | 21.9% [20.6, 23.3] | 39.0% [37.5, 40.6] | +17.1 pp |
| LLaMA 3-8B | 15.9% [14.3, 17.7] | 18.3% [16.5, 20.1] | +2.3 pp |

**By language (equiv.-aware):**

| Language | Accuracy | 95% CI |
|---|---|---|
| English | 44.9% | [41.9, 47.9] |
| Hindi | 35.2% | [32.3, 38.3] |
| Bengali | 34.0% | [31.1, 36.8] |
| Urdu | 32.4% | [29.7, 35.2] |
| Gujarati | 32.5% | [29.7, 35.3] |
| Malayalam | 29.8% | [27.0, 32.7] |

The English–Malayalam gap is **non-overlapping**, confirming the core multilingual disparity finding is robust to metric choice. 755 additional correct cases were identified across 6,326 responses that exact-match penalised incorrectly.

---

## Survey Results

`Supplementary_Materials/Survey_Results/` contains the anonymized responses from the prompt elicitation exercise conducted with 53 Grade 11–12 students from schools in Uttar Pradesh, India. The CSV includes student responses and prompt-style preferences. Email addresses and personally identifying fields have been removed prior to release.

**Note:** This sample reflects a predominantly Hindi/English-medium interaction context and does not represent Bengali-, Gujarati-, or Malayalam-medium students.

---

## Requirements

```bash
pip install sympy pandas numpy openai transformers torch
```

- Python 3.8+
- For LLaMA 3 and DeepSeek-R1: GPU with at least 16 GB VRAM recommended
- For GPT-4o mini: valid OpenAI API key

---

## Citation

If you use this dataset or code, please cite:

```bibtex
@inproceedings{kumar-roy-2026-multilingual,
  title     = {A Multilingual Evaluation of {LLM}-Based High-School Mathematics Assistance Across Indian Languages},
  author    = {Kumar, Yash and Roy, Subhajit},
  booktitle = {Proceedings of the 5rd Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics},
  year      = {2026}
}
```

---

## License

This repository contains two separately licensed components:

**Code** — all Python scripts (`robust_equivalence_scoring.py`, `bootstrap_ci.py`, and all scripts inside `Supplementary_Materials/`) are released under the **MIT License**. See `LICENSE`.

**Dataset** — all CSV files inside `Datasets/` and `Supplementary_Materials/Datasets/` are released under **CC BY 4.0** (Creative Commons Attribution 4.0 International). See `LICENSE_DATA`.

The dataset consists of numerically perturbed mathematical problem instances created by the authors from publicly available NCERT and SCERT Class 11 mathematics textbooks. The perturbations, manually derived ground-truth solutions, cross-lingual alignments, and subcategory annotations are original contributions of the authors. If you use the dataset, please cite the paper using the BibTeX entry above.
