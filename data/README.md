# Data Directory

This directory contains the datasets and results for reproducing the ML Sentinel experiments.

## Quick Setup

```bash
# Download all datasets automatically
python download.py
```

## Datasets

### 1. Folktables (ACS Income Prediction)
- **Source:** [Folktables](https://github.com/socialfoundations/folktables)
- **Task:** Binary classification (income >$50K)
- **Shift:** Geographic (California → 50 US states)
- **Size:** ~1.4M samples total

### 2. CivilComments (Toxicity Detection)
- **Source:** [WILDS Benchmark](https://wilds.stanford.edu/)
- **Task:** Binary classification (toxic/non-toxic)
- **Shift:** Demographic groups (LGBTQ, Muslim, Black, White, etc.)
- **Size:** 71,854 test samples

### 3. Amazon Reviews (Sentiment Analysis)
- **Source:** [WILDS Benchmark](https://wilds.stanford.edu/)
- **Task:** Multi-class sentiment (1-5 stars)
- **Shift:** Product categories (Electronics → Clothing, etc.)
- **Size:** 35,006 test samples

## File Structure

After running `download.py`:

```
data/
├── README.md                    # This file
├── download.py                  # Download script
├── folktables_results.csv       # Folktables experiment results
├── civilcomments_results.csv    # CivilComments results
└── amazon_results.csv           # Amazon results
```

## Pre-computed Results

If you don't want to run the full experiments, the key statistics are:

| Benchmark | Scenarios | Degraded | Drift TPR | Sentinel r | Test Samples |
|-----------|-----------|----------|-----------|------------|--------------|
| Folktables | 50 | 27 | 0% | 0.91 | 1,398,764 |
| CivilComments | 8 | 4 | 0% | 0.90 | 71,854 |
| Amazon | 10 | 5 | 0% | 0.82 | 35,006 |
| **Total** | **68** | **36** | **0%** | — | **1,576,624** |

## Column Descriptions

Each results CSV contains:

| Column | Description |
|--------|-------------|
| `scenario` | Scenario identifier (state, group, category) |
| `ks_drift` | Kolmogorov-Smirnov drift score |
| `sentinel_score` | Model confidence drop |
| `ref_acc` | Reference (training) accuracy |
| `test_acc` | Test accuracy |
| `acc_drop` | Accuracy degradation (ref - test) |
| `is_significant` | Statistical significance of degradation |
| `n_samples` | Number of test samples |

## License

- Folktables: MIT License
- WILDS (CivilComments, Amazon): MIT License

See individual dataset sources for detailed licensing.
