#!/usr/bin/env python3
"""
Folktables cross-state benchmark: does input drift tell you when a model got worse?

Setup (2018 ACS 1-Year person data, via the folktables task definitions):
- For each task (income, employment, public coverage), train a gradient-boosted
  classifier on 70% of California and keep the other 30% as an in-distribution holdout.
- Deploy the model unchanged to each of the other 49 states. A state counts as
  "degraded" when its accuracy is more than DEGRADED_PP points below the CA holdout.
- Compare label-free signals for spotting degraded states:
    * input drift: max per-feature KS and max per-feature PSI (CA train vs state).
      The state-code feature (ST, in the public-coverage task) is left out of drift:
      it is constant within a state, so it would trivially alarm on every deployment.
    * confidence: drop in mean max-probability (ml_sentinel.SentinelMonitor)
    * ATC: average thresholded confidence (Garg et al., ICLR 2022)

Per-scenario results go to results/folktables_2018.csv. The script appends and
skips (task, state) pairs already present, so it can be stopped and resumed.

Usage:
    python experiments/folktables_benchmark.py --data-dir /path/to/acs_cache
    python experiments/folktables_benchmark.py --summarize
"""

import argparse
import json
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ml_sentinel import (  # noqa: E402
    SentinelMonitor,
    atc_accuracy,
    atc_threshold,
    ks_scores,
    psi_scores,
)

try:
    from folktables import ACSEmployment, ACSIncome, ACSPublicCoverage
except ImportError:
    sys.exit("pip install folktables")

ROOT = Path(__file__).parent.parent
RESULTS_CSV = ROOT / "results" / "folktables_2018.csv"
SUMMARY_JSON = ROOT / "results" / "folktables_2018_summary.json"

YEAR = "2018"
SOURCE_STATE = "CA"
SEED = 42
DEGRADED_PP = 0.02  # accuracy drop (absolute) that counts as degradation
KS_THRESHOLD = 0.10
PSI_THRESHOLD = 0.25  # common "significant shift" rule of thumb
CONF_THRESHOLD = 0.02  # alert when mean confidence drops by > 2 points
ATC_THRESHOLD = 0.02  # alert when ATC-estimated accuracy drops by > 2 points

TASKS = {
    "income": ACSIncome,
    "employment": ACSEmployment,
    "public_coverage": ACSPublicCoverage,
}

# fmt: off
STATES = {
    "AL": "01", "AK": "02", "AZ": "04", "AR": "05", "CA": "06", "CO": "08", "CT": "09",
    "DE": "10", "FL": "12", "GA": "13", "HI": "15", "ID": "16", "IL": "17", "IN": "18",
    "IA": "19", "KS": "20", "KY": "21", "LA": "22", "ME": "23", "MD": "24", "MA": "25",
    "MI": "26", "MN": "27", "MS": "28", "MO": "29", "MT": "30", "NE": "31", "NV": "32",
    "NH": "33", "NJ": "34", "NM": "35", "NY": "36", "NC": "37", "ND": "38", "OH": "39",
    "OK": "40", "OR": "41", "PA": "42", "RI": "44", "SC": "45", "SD": "46", "TN": "47",
    "TX": "48", "UT": "49", "VT": "50", "VA": "51", "WA": "53", "WV": "54", "WI": "55",
    "WY": "56",
}
# fmt: on

COLUMNS = sorted(
    set().union(*[set(t.features) | {t.target} for t in TASKS.values()])
    | {"PWGTP", "ST", "AGEP", "PINCP", "WKHP"}
)


def load_state(state: str, data_dir: Path) -> pd.DataFrame:
    """Download (once) and read only the needed columns of one state's PUMS file."""
    data_dir.mkdir(parents=True, exist_ok=True)
    zpath = data_dir / f"{YEAR}_csv_p{state.lower()}.zip"
    if not zpath.exists():
        url = (
            "https://www2.census.gov/programs-surveys/acs/data/pums/"
            f"{YEAR}/1-Year/csv_p{state.lower()}.zip"
        )
        tmp = zpath.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(zpath)
    with zipfile.ZipFile(zpath) as zf:
        member = next(n for n in zf.namelist() if n.endswith(".csv"))
        with zf.open(member) as fh:
            return pd.read_csv(fh, usecols=lambda c: c in COLUMNS)


def run(data_dir: Path) -> None:
    done = set()
    if RESULTS_CSV.exists():
        prev = pd.read_csv(RESULTS_CSV)
        done = set(zip(prev["task"], prev["state"]))

    ca = load_state(SOURCE_STATE, data_dir)
    for task_name, task in TASKS.items():
        todo = [s for s in STATES if s != SOURCE_STATE and (task_name, s) not in done]
        if not todo:
            continue
        X, y, _ = task.df_to_numpy(ca)
        drift_cols = [i for i, f in enumerate(task.features) if f != "ST"]
        drift_names = [task.features[i] for i in drift_cols]
        X_tr, X_ho, y_tr, y_ho = train_test_split(
            X, y, test_size=0.3, random_state=SEED, stratify=y
        )
        model = HistGradientBoostingClassifier(random_state=SEED).fit(X_tr, y_tr)

        p_ho = model.predict_proba(X_ho)
        ho_acc = float((p_ho.argmax(1) == y_ho).mean())
        monitor = SentinelMonitor(method="max_prob")
        ho_conf = monitor.score_from_predictions(p_ho)
        t_atc = atc_threshold(p_ho, y_ho)
        print(f"[{task_name}] CA train={len(y_tr)} holdout acc={ho_acc:.4f}", flush=True)

        for state in todo:
            t0 = time.time()
            Xs, ys, _ = task.df_to_numpy(load_state(state, data_dir))
            ps = model.predict_proba(Xs)
            acc = float((ps.argmax(1) == ys).mean())
            conf = monitor.score_from_predictions(ps)
            ks = ks_scores(X_tr[:, drift_cols], Xs[:, drift_cols])
            psi = psi_scores(X_tr[:, drift_cols], Xs[:, drift_cols])
            row = {
                "task": task_name,
                "state": state,
                "n": len(ys),
                "ca_holdout_acc": ho_acc,
                "acc": acc,
                "acc_drop": ho_acc - acc,
                "ks_max": float(ks.max()),
                "psi_max": float(psi.max()),
                "conf_drop": ho_conf - conf,
                "atc_drop": ho_acc - atc_accuracy(ps, t_atc),
                "ks_feature": drift_names[int(ks.argmax())],
                "psi_feature": drift_names[int(psi.argmax())],
            }
            pd.DataFrame([row]).to_csv(
                RESULTS_CSV, mode="a", header=not RESULTS_CSV.exists(), index=False
            )
            print(
                f"  {state} n={row['n']:>6} drop={row['acc_drop']:+.3f} "
                f"ks={row['ks_max']:.3f} psi={row['psi_max']:.3f} "
                f"conf={row['conf_drop']:+.3f} atc={row['atc_drop']:+.3f} "
                f"({time.time() - t0:.1f}s)",
                flush=True,
            )


def bootstrap_ci(fn, df: pd.DataFrame, n: int = 2000) -> list:
    rng = np.random.default_rng(SEED)
    vals = [fn(df.iloc[rng.integers(0, len(df), len(df))]) for _ in range(n)]
    return [float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))]


def summarize() -> dict:
    df = pd.read_csv(RESULTS_CSV)
    degraded = df["acc_drop"] > DEGRADED_PP
    signals = {
        "ks_max": KS_THRESHOLD,
        "psi_max": PSI_THRESHOLD,
        "conf_drop": CONF_THRESHOLD,
        "atc_drop": ATC_THRESHOLD,
    }
    out = {
        "data": f"ACS {YEAR} 1-Year, train {SOURCE_STATE}, test 49 other states",
        "n_scenarios": int(len(df)),
        "n_test_samples": int(df["n"].sum()),
        "degraded_definition": f"accuracy drop vs CA holdout > {DEGRADED_PP:.0%} absolute",
        "n_degraded": int(degraded.sum()),
        "signals": {},
    }
    for col, thr in signals.items():
        alert = df[col] > thr
        sig = {
            "threshold": thr,
            "tpr": float((alert & degraded).sum() / max(degraded.sum(), 1)),
            "fpr": float((alert & ~degraded).sum() / max((~degraded).sum(), 1)),
            "auroc": float(roc_auc_score(degraded, df[col])),
            "auroc_ci95": bootstrap_ci(
                lambda d, c=col: (
                    roc_auc_score(d["acc_drop"] > DEGRADED_PP, d[c])
                    if (d["acc_drop"] > DEGRADED_PP).nunique() == 2
                    else np.nan
                ),
                df,
            ),
            "spearman_vs_acc_drop": float(stats.spearmanr(df[col], df["acc_drop"])[0]),
        }
        if col in ("conf_drop", "atc_drop"):
            sig["mae_vs_acc_drop"] = float((df[col] - df["acc_drop"]).abs().mean())
        out["signals"][col] = sig
    # Pooled AUROC mixes tasks (income both drifts and degrades most), so also
    # report rank correlation with the accuracy drop within each task.
    out["per_task"] = {
        t: {
            "n_degraded": int((g["acc_drop"] > DEGRADED_PP).sum()),
            "n": int(len(g)),
            "median_acc_drop": float(g["acc_drop"].median()),
            "max_acc_drop": float(g["acc_drop"].max()),
            "spearman_vs_acc_drop": {
                c: float(stats.spearmanr(g[c], g["acc_drop"])[0]) for c in signals
            },
        }
        for t, g in df.groupby("task")
    }
    for t, g in df.groupby("task"):
        out["per_task"][t]["atc_mae"] = float((g["atc_drop"] - g["acc_drop"]).abs().mean())
        if "ks_feature" in g:
            out["per_task"][t]["ks_feature_counts"] = g["ks_feature"].value_counts().to_dict()
            out["per_task"][t]["psi_feature_counts"] = g["psi_feature"].value_counts().to_dict()
    err = df["atc_drop"] - df["acc_drop"]
    out["signals"]["atc_drop"]["max_abs_error"] = float(err.abs().max())
    out["signals"]["atc_drop"]["mean_error"] = float(err.mean())
    # Sensitivity to the 2-point "degraded" cutoff: re-label deployments at other
    # cutoffs. Drift thresholds stay fixed; the confidence and ATC alarms use the
    # cutoff itself as their threshold (alarm when the estimated drop exceeds it).
    out["cutoff_sensitivity"] = {}
    for cut in (0.01, 0.02, 0.03, 0.05):
        deg = df["acc_drop"] > cut
        thr = {"ks_max": KS_THRESHOLD, "psi_max": PSI_THRESHOLD, "conf_drop": cut, "atc_drop": cut}
        out["cutoff_sensitivity"][f"{cut:.2f}"] = {
            "n_degraded": int(deg.sum()),
            **{
                c: {
                    "alarms_on_degraded": int(((df[c] > v) & deg).sum()),
                    "alarms_on_healthy": int(((df[c] > v) & ~deg).sum()),
                }
                for c, v in thr.items()
            },
        }
    SUMMARY_JSON.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data" / "acs")
    ap.add_argument("--summarize", action="store_true")
    args = ap.parse_args()
    RESULTS_CSV.parent.mkdir(exist_ok=True)
    if not args.summarize:
        run(args.data_dir)
    summarize()
