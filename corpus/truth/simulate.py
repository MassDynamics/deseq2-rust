# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2", "pandas>=2"]
# ///
"""Seeded negative-binomial simulator for the count truth corpus.

    uv run corpus/truth/simulate.py [--out ~/wd/md-count-truth-corpus] [scenario ...]

Each scenario directory gets the engine inputs (counts.csv, sample_info.csv, comparisons.csv,
params.json) in the same shape as the count golden corpus `input_*` files, plus the truth:
truth.csv (one row per gene), truth_samples.csv (true size factors) and truth.json (the
generating parameters). The generating model is

    y_gj ~ NB(mean = s_j * q_gj, dispersion = alpha_g),   var = mu + alpha_g * mu^2
    log2 q_gj = log2 mu_g + x_j' beta_g
    alpha_g = (a0 + a1 / mu_g) * exp(N(0, sigma_d^2))

mu_g is drawn from the quantile function of airway's base means (the 8000-gene airway subset in
the count golden corpus, normalised by median-of-ratios), so the count range matches real data.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# 41 quantiles (0, 2.5%, ..., 100%) of log(baseMean) over the airway subset's genes with
# baseMean > 0 (7390 genes), computed once from md-count-golden-corpus/shared/airway. The top
# quantile is capped at 11.5 (a mean of ~1e5) so a single draw cannot dominate a library.
AIRWAY_LOG_BASEMEAN_Q = np.array([
    -2.423, -1.63, -0.89, -0.139, 0.584, 1.251, 1.841, 2.415, 2.955, 3.548, 3.967, 4.308, 4.624,
    4.864, 5.1, 5.277, 5.455, 5.604, 5.732, 5.855, 5.961, 6.081, 6.193, 6.295, 6.384, 6.492, 6.583,
    6.686, 6.792, 6.926, 7.032, 7.126, 7.245, 7.364, 7.495, 7.665, 7.866, 8.083, 8.424, 8.985, 11.5])

# Dispersion trend and scatter. a0/a1 sit between airway's fit (0.009, 3.7) and noisier tissue
# data; sigma_d^2 = 0.49 is kept above DESeq2's 0.25 floor on the prior variance so the
# estimate is identifiable against the truth.
A0, A1, SIGMA_D = 0.04, 1.5, 0.7
SF_SD = 0.25  # log-scale sd of the true size factors


def draw_means(rng, n):
    u = rng.uniform(0, 1, n)
    grid = np.linspace(0, 1, len(AIRWAY_LOG_BASEMEAN_Q))
    return np.exp(np.interp(u, grid, AIRWAY_LOG_BASEMEAN_Q))


def draw_lfc(rng, n, pi0):
    """Null/DE mixture on log2: pi0 exact zeros; of the DE genes 60% small (|lfc| in
    [0.3, 1]) and 40% large (|lfc| in [1, 4]), random sign."""
    de = rng.uniform(0, 1, n) >= pi0
    big = rng.uniform(0, 1, n) < 0.4
    mag = np.where(big, rng.uniform(1, 4, n), rng.uniform(0.3, 1, n))
    sign = np.where(rng.uniform(0, 1, n) < 0.5, -1.0, 1.0)
    return np.where(de, sign * mag, 0.0), de


def rnbinom(rng, mean, alpha):
    # NB as a gamma-Poisson mixture: lambda ~ Gamma(shape 1/alpha, scale alpha*mean).
    shape = 1.0 / alpha
    lam = rng.gamma(shape, mean / shape)
    return rng.poisson(lam)


def size_factors(rng, m):
    s = np.exp(rng.normal(0, SF_SD, m))
    return s / np.exp(np.mean(np.log(s)))


def simulate(seed, n_genes, groups, n_rep, pi0=0.8, batch=False, covariate=False,
             outliers=0.0, anova=False):
    """Inputs and truth for one scenario. `groups` are condition labels, the first is the
    reference. Comparisons are every later level against the first, and with three levels
    also C - B, which exercises production's relevel refit."""
    rng = np.random.default_rng(seed)
    k = len(groups)
    cond = np.repeat(groups, n_rep)
    m = len(cond)
    samples = [f"S{j + 1:02d}" for j in range(m)]
    s = size_factors(rng, m)
    mu = draw_means(rng, n_genes)
    alpha = (A0 + A1 / mu) * np.exp(rng.normal(0, SIGMA_D, n_genes))

    # Group effects on log2 against the reference. With three groups each non-reference level
    # is DE with probability (1 - pi0) / 1.5, so roughly 1 - pi0 of genes are DE somewhere.
    beta = np.zeros((n_genes, k))
    de_any = np.zeros(n_genes, bool)
    for g in range(1, k):
        b, de = draw_lfc(rng, n_genes, pi0 if k == 2 else 1 - (1 - pi0) / 1.5)
        beta[:, g] = b
        de_any |= de
    log2q = np.log2(mu)[:, None] + beta[:, [groups.index(c) for c in cond]]

    si = pd.DataFrame({"replicate": samples, "condition": cond})
    truth = pd.DataFrame({"id": np.arange(1, n_genes + 1), "mu": mu, "disp": alpha,
                          "disp_trend": A0 + A1 / mu, "de": de_any})
    for g in range(1, k):
        truth[f"beta_{groups[g]}"] = beta[:, g]

    controls = []
    if batch:
        # Two batches alternating within each condition: full rank, m - p = m - (k + 1).
        bt = np.empty(m, dtype=object)
        for gi in range(k):
            idx = np.where(cond == groups[gi])[0]
            bt[idx] = ["b1" if (i + gi) % 2 == 0 else "b2" for i in range(len(idx))]
        gb = rng.normal(0, 0.5, n_genes)
        log2q = log2q + gb[:, None] * (bt == "b2")[None, :]
        si["batch"] = bt
        truth["beta_batch_b2"] = gb
        controls.append("batch")
    if covariate:
        x = rng.normal(0, 1, m)
        x = np.round(x - x.mean(), 3)
        has = rng.uniform(0, 1, n_genes) < 0.3
        gc = np.where(has, rng.normal(0, 0.5, n_genes), 0.0)
        log2q = log2q + gc[:, None] * x[None, :]
        si["covariate"] = x
        truth["beta_covariate"] = gc
        controls.append("covariate")

    mean = s[None, :] * np.exp2(log2q)
    y = rnbinom(rng, mean, alpha[:, None])

    outlier_sample = np.array([""] * n_genes, dtype=object)
    outlier_factor = np.zeros(n_genes)
    if outliers > 0:
        # One sample per chosen gene gets its count multiplied far beyond the NB tail. Only
        # genes with a reasonable mean are eligible so the outlier is detectable at all.
        elig = np.where(mu >= 20)[0]
        pick = rng.choice(elig, size=int(outliers * n_genes), replace=False)
        for g in pick:
            j = rng.integers(m)
            f = rng.uniform(10, 50)
            y[g, j] = round(mean[g, j] * f) + 50
            outlier_sample[g] = samples[j]
            outlier_factor[g] = f
    truth["outlier"] = outlier_sample != ""
    truth["outlier_sample"] = outlier_sample
    truth["outlier_factor"] = outlier_factor

    # True log2 fold change per production comparison label "left - right".
    pairs = [(groups[g], groups[0]) for g in range(1, k)]
    if k == 3:
        pairs.append((groups[2], groups[1]))
    for left, right in pairs:
        truth[f"lfc {left} - {right}"] = beta[:, groups.index(left)] - beta[:, groups.index(right)]

    counts = pd.DataFrame(y, columns=samples)
    counts.insert(0, "id", np.arange(1, n_genes + 1))
    comparisons = pd.DataFrame({"left": [p[0] for p in pairs], "right": [p[1] for p in pairs]})
    comparisons["encoded_left"] = comparisons["left"]
    comparisons["encoded_right"] = comparisons["right"]
    params = {
        "condition_col": "condition",
        "control_cols": controls,
        "mode": "anova" if anova else "discovery",
        "comparison_type": "custom",
        "edger_norm_method": "TMM",
        "deseq2_alpha": 0.05,
        "apeglm_seed": 1,
        "condition_separator": " - ",
    }
    meta = {"seed": seed, "n_genes": n_genes, "groups": groups, "n_rep": n_rep, "pi0": pi0,
            "a0": A0, "a1": A1, "sigma_d": SIGMA_D, "sf_sd": SF_SD, "batch": batch,
            "covariate": covariate, "outlier_fraction": outliers, "mode": params["mode"]}
    ts = pd.DataFrame({"replicate": samples, "size_factor": s})
    return counts, si, comparisons, params, truth, ts, meta


SCENARIOS = {
    # name: (seed, n_genes, groups, n_rep, kwargs)
    "s2x3": (101, 10000, ["A", "B"], 3, {}),
    "s2x4": (102, 10000, ["A", "B"], 4, {}),
    "s3x3": (103, 10000, ["A", "B", "C"], 3, {}),
    "s3x3_anova": (103, 10000, ["A", "B", "C"], 3, {"anova": True}),
    "s2x7_outliers": (104, 10000, ["A", "B"], 7, {"outliers": 0.05}),
    "s2x3_batch": (105, 10000, ["A", "B"], 3, {"batch": True}),
    "s2x4_covariate": (106, 10000, ["A", "B"], 4, {"covariate": True}),
}
N_NULL, N_MIX, N_SMALL = 200, 50, 1000


def all_scenarios():
    out = dict(SCENARIOS)
    for i in range(N_NULL):
        out[f"null_{i:03d}"] = (10_000 + i, N_SMALL, ["A", "B"], 3, {"pi0": 1.0})
    for i in range(N_MIX):
        out[f"mix_{i:03d}"] = (20_000 + i, N_SMALL, ["A", "B"], 3, {"pi0": 0.8})
    return out


def write(root: Path, name, spec):
    seed, n, groups, n_rep, kw = spec
    counts, si, cmp, params, truth, ts, meta = simulate(seed, n, groups, n_rep, **kw)
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    counts.to_csv(d / "counts.csv", index=False)
    si.to_csv(d / "sample_info.csv", index=False)
    cmp.to_csv(d / "comparisons.csv", index=False)
    (d / "params.json").write_text(json.dumps(params, indent=2) + "\n")
    truth.to_csv(d / "truth.csv", index=False, float_format="%.17g")
    ts.to_csv(d / "truth_samples.csv", index=False, float_format="%.17g")
    (d / "truth.json").write_text(json.dumps(meta, indent=2) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path.home() / "wd/md-count-truth-corpus"))
    ap.add_argument("only", nargs="*")
    a = ap.parse_args()
    root = Path(a.out).expanduser()
    specs = all_scenarios()
    names = a.only or list(specs)
    for name in names:
        write(root, name, specs[name])
    index = {n: {"seed": specs[n][0], "n_genes": specs[n][1], "groups": specs[n][2],
                 "n_rep": specs[n][3], **specs[n][4]} for n in specs}
    (root / "index.json").write_text(json.dumps(index, indent=1) + "\n")
    print(f"wrote {len(names)} scenarios to {root}")


if __name__ == "__main__":
    main()
