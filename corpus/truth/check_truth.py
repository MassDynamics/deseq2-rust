# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "scipy"]
# ///
"""Truth-corpus checks: is an engine's output mathematically right, independent of R?

    uv run corpus/truth/check_truth.py --engine r          # R reference outputs (r_*.csv)
    uv run corpus/truth/check_truth.py --engine rust       # runs run_rust.py first, then checks
    uv run corpus/truth/check_truth.py --engine pydeseq2   # third opinion (py_*.csv)

Every check compares an engine's output with the simulated truth (calibration, recovery,
shrinkage) or with its own defining equations (per-gene certificates). Thresholds and their
rationale live in THRESHOLDS below; status-truth.md carries the same table with R's measured
values. A certificate needs internals the production table lacks (size factors, dispersions,
coefficients); those come from `<prefix>_<engine>.diag/`. Without that dir the certificate is
SKIP (not exposed).

edge-rust can import this file by path:
    spec = importlib.util.spec_from_file_location("check_truth", ".../corpus/truth/check_truth.py")
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats

HERE = Path(__file__).resolve().parent
OUT_NOTE = ("documented: both trends are fitted before outlier handling (DESeq2 replaces after the "
            "trend; edgeR has none), so 5% outlier genes inflate the trend")
DEFAULT_CORPUS = Path.home() / "wd" / "md-count-truth-corpus"
PREFIX = {"r": "r", "rust": "rust", "pydeseq2": "py"}
LN2 = math.log(2.0)
Z975 = stats.norm.ppf(0.975)
MAIN = ["s2x3", "s2x4", "s3x3", "s2x7_outliers", "s2x3_batch", "s2x4_covariate"]
SHRINKS = ["deseq2_normal", "deseq2_apeglm", "deseq2_ashr"]

THRESHOLDS = {
    # Calibration
    "ks_p_min": 0.01,
    "ks_mu_min": 10.0,
    "ks_per_rep": 10,
    "type1_z": 3.0,
    "fdr_alpha": 0.05,
    "fdr_z": 2.326,
    # DESeq2's Wald test plugs in the MAP dispersion as known and uses a normal reference, so at
    # n = 3 per group it is anti-conservative on this truth model (edgeR QL is not). These bands
    # are the documented-property thresholds for DESeq2: about 1.2x / 2x the nominal level.
    "deseq2_type1": {0.05: 0.06, 0.01: 0.02},
    "deseq2_fdp_max": 0.10,
    # Recovery
    "sf_max_logdev": 0.05,
    "lfc_mu_min": 50.0,
    "lfc_bias_max": 0.05,
    "ci_cover_min": {"edger": 0.93, "deseq2": 0.93},
    "disp_trend_logratio_max": 0.15,
    # Shrinkage
    "mse_ratio_max": 1.0,
    "lfsr_cut": 0.05,
    "lfsr_z": 2.326,
    # Certificates
    "score_z_max": 5e-3,
    "score_tight": 1e-4,
    "boundary_mu": 0.5,
    "se_rel": 1e-6,
    "wald_rel": 1e-9,
    "p_rel": 1e-8,
    "cooks_rel": 1e-6,
    "apeglm_newton_dec_max": 5e-6,
    "prior_var_abs": 2e-4,
    "kkt_tol": 1e-4,
    "ashr_post_rel": 1e-6,
}


@dataclass
class Result:
    group: str
    check: str
    scope: str
    engine: str
    status: str  # PASS / FAIL / SKIP / INFO
    value: str
    threshold: str = ""
    note: str = ""


@dataclass
class Report:
    rows: list = field(default_factory=list)

    def add(self, *a, **k):
        self.rows.append(Result(*a, **k))

    def table(self) -> pd.DataFrame:
        return pd.DataFrame([r.__dict__ for r in self.rows])


def fmt(x, d=4):
    if x is None:
        return "NA"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    x = float(x)
    if not np.isfinite(x):
        return "NA"
    ax = abs(x)
    return f"{x:.{d}g}" if (0 < ax < 1e-3 or ax >= 1e4) else f"{x:.{d}f}"


def verdict(ok) -> str:
    return "PASS" if bool(ok) else "FAIL"


class Scenario:
    def __init__(self, root: Path, name: str, prefix: str):
        self.name, self.dir, self.prefix = name, root / name, prefix
        self.truth = pd.read_csv(self.dir / "truth.csv").set_index("id")
        self.tsamp = pd.read_csv(self.dir / "truth_samples.csv").set_index("replicate")
        self.meta = json.loads((self.dir / "truth.json").read_text())
        self.params = json.loads((self.dir / "params.json").read_text())
        cmp = pd.read_csv(self.dir / "comparisons.csv")
        sep = self.params.get("condition_separator", " - ")
        self.labels = [f"{a}{sep}{b}" for a, b in zip(cmp["left"], cmp["right"])]
        self.si = pd.read_csv(self.dir / "sample_info.csv")
        self._counts = None

    @property
    def counts(self) -> pd.DataFrame:
        if self._counts is None:
            self._counts = pd.read_csv(self.dir / "counts.csv").set_index("id")
        return self._counts

    def out(self, engine: str):
        f = self.dir / f"{self.prefix}_{engine}.csv"
        return pd.read_csv(f).set_index("GroupId") if f.exists() else None

    def diag(self, engine: str):
        d = self.dir / f"{self.prefix}_{engine}.diag"
        return d if d.is_dir() else None


def ids(df):
    return df.index.to_numpy()


def bh(p):
    p = np.asarray(p, float)
    n = len(p)
    o = np.argsort(p)[::-1]
    q = np.minimum.accumulate(p[o] * n / np.arange(n, 0, -1))
    out = np.empty(n)
    out[o] = np.minimum(q, 1.0)
    return out


def max_rel(a, b, floor=1e-12):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if not m.any():
        return float("nan")
    return float(np.max(np.abs(a[m] - b[m]) / np.maximum(np.abs(b[m]), floor)))


def p_err(a, b):
    """Relative error of p-values on the -log scale (exact for tiny p, rel for ordinary p)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    if not m.any():
        return float("nan")
    la, lb = -np.log(a[m]), -np.log(b[m])
    return float(np.max(np.abs(la - lb) / np.maximum(lb, 1.0)))


def r_trim_mean(x, trim):
    """Row-wise mean(x, trim) with R's lo/hi index rule."""
    n = x.shape[1]
    lo = math.floor(n * trim) + 1
    hi = n + 1 - lo
    s = np.sort(x, axis=1)
    return s[:, lo - 1:hi].mean(axis=1)


def design_from(path: Path):
    d = pd.read_csv(path)
    reps = d["replicate"].astype(str).tolist()
    return d.drop(columns="replicate").to_numpy(float), reps, list(d.columns[1:])


def nb_score(y, mu, alpha, X, beta_nat, ridge_nat=0.0):
    """Score of the NB log-likelihood (log link) and score_k / sqrt(I_kk)."""
    a = np.asarray(alpha, float)
    a = a.reshape(-1, 1) if a.ndim else a
    r = (y - mu) / (1.0 + a * mu)
    score = r @ X - ridge_nat * beta_nat
    w = mu / (1.0 + a * mu)
    info = w @ (X * X) + ridge_nat
    return score, score / np.sqrt(info)


def xtwx(w, X):
    return np.einsum("gm,mk,ml->gkl", w, X, X)


# --------------------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------------------
def null_pvalues(scns, engine, col):
    pooled, thinned, nrej_rep = [], [], []
    rng = np.random.default_rng(7)
    T = THRESHOLDS
    for s in scns:
        o = s.out(engine)
        if o is None:
            return None
        p = o[col].to_numpy(float)
        mu = s.truth.loc[ids(o), "mu"].to_numpy()
        ok = np.isfinite(p)
        pooled.append(p[ok])
        cand = np.flatnonzero(ok & (mu >= T["ks_mu_min"]))
        pick = rng.choice(cand, size=min(T["ks_per_rep"], len(cand)), replace=False)
        thinned.append(p[pick])
        adj = o[col.replace("PValue", "AdjPValue")]
        nrej_rep.append(int((adj <= T["fdr_alpha"]).sum()))
    return np.concatenate(pooled), np.concatenate(thinned), np.array(nrej_rep)


def check_calibration(rep, root, prefix, engines):
    T = THRESHOLDS
    nulls = [Scenario(root, f"null_{i:03d}", prefix) for i in range(200)]
    mixes = [Scenario(root, f"mix_{i:03d}", prefix) for i in range(50)]
    a = T["fdr_alpha"]
    for eng in ("edger", "deseq2"):
        if eng not in engines:
            continue
        col = "PValue B - A"
        got = null_pvalues(nulls, eng, col)
        if got is None:
            rep.add("calibration", "null p-values", "null x200", eng, "SKIP", "no output")
            continue
        pooled, thinned, nrej = got
        ds = eng == "deseq2"
        DNOTE = "documented: DESeq2 Wald plugs in the MAP dispersion and a normal reference"
        ks = stats.kstest(thinned, "uniform")
        rep.add("calibration", "KS uniform (thinned)", "null x200", eng,
                "INFO" if ds else verdict(ks.pvalue > T["ks_p_min"]), f"p={fmt(ks.pvalue)} n={len(thinned)}",
                f"> {T['ks_p_min']}", DNOTE if ds else "10 genes/rep with true mean >= 10")
        ksall = stats.kstest(pooled, "uniform")
        rep.add("calibration", "KS uniform (all genes, info)", "null x200", eng, "INFO",
                f"D={fmt(ksall.statistic)} p={fmt(ksall.pvalue)} n={len(pooled)}")
        for lev in (0.05, 0.01):
            rate = float(np.mean(pooled <= lev))
            ub = lev + T["type1_z"] * math.sqrt(lev * (1 - lev) / len(pooled))
            if ds:
                ub = T["deseq2_type1"][lev]
            rep.add("calibration", f"type I error at {lev}", "null x200", eng,
                    verdict(rate <= ub), fmt(rate), f"<= {fmt(ub)}", DNOTE if ds else "nominal + 3 binomial sd")
        fwer = float(np.mean(nrej > 0))
        ub = a + T["fdr_z"] * math.sqrt(a * (1 - a) / len(nrej))
        rep.add("calibration", "BH FDR (= FWER under global null)", "null x200", eng,
                "INFO" if ds else verdict(fwer <= ub), fmt(fwer), f"<= {fmt(ub)}",
                DNOTE + "; one extreme gene per rep is enough to reject" if ds else "")
        fdp = []
        for s in mixes:
            o = s.out(eng)
            if o is None:
                break
            rej = (o["AdjPValue B - A"] <= a).to_numpy()
            null = ~s.truth.loc[ids(o), "de"].to_numpy().astype(bool)
            fdp.append((rej & null).sum() / max(rej.sum(), 1))
        if len(fdp) == len(mixes):
            fdp = np.array(fdp)
            ub = a + T["fdr_z"] * fdp.std(ddof=1) / math.sqrt(len(fdp))
            if ds:
                ub = T["deseq2_fdp_max"]
            rep.add("calibration", "mean BH FDP (pi0=0.8)", "mix x50", eng,
                    verdict(fdp.mean() <= ub), fmt(fdp.mean()), f"<= {fmt(ub)}",
                    DNOTE if ds else "BH controls at pi0*alpha = 0.04; bound alpha + 2.326 MC sd")


# --------------------------------------------------------------------------------------
# Recovery
# --------------------------------------------------------------------------------------
def true_lfc(s, lab, gids):
    return s.truth.loc[gids, f"lfc {lab}"].to_numpy(float)


def check_recovery(rep, root, prefix, engines):
    T = THRESHOLDS
    for name in MAIN + ["s3x3_anova"]:
        s = Scenario(root, name, prefix)
        sig2 = s.meta["sigma_d"] ** 2
        tsf = s.tsamp["size_factor"]
        deng = "deseq2_anova" if s.params.get("mode") == "anova" else "deseq2"
        out_all = s.truth["outlier"].astype(str).str.upper().eq("TRUE")
        has_out = bool(out_all.any())
        for eng in (deng, "edger"):
            if s.out(eng) is None:
                continue
            d = s.diag(eng)
            if d is None:
                rep.add("recovery", "size factors", name, eng, "SKIP", "not exposed")
                continue
            sm = pd.read_csv(d / "samples.csv").set_index("replicate")
            est = sm["size_factor"] if eng.startswith("deseq2") else sm["lib_size"] * sm["norm_factor"]
            dev = np.log(est.to_numpy()) - np.log(tsf.loc[sm.index].to_numpy())
            dev -= dev.mean()
            what = "size factors" if eng.startswith("deseq2") else "effective library (lib x TMM)"
            rep.add("recovery", f"{what} vs truth (up to scale)", name, eng,
                    verdict(np.max(np.abs(dev)) <= T["sf_max_logdev"]),
                    f"max|dlog|={fmt(np.max(np.abs(dev)))}", f"<= {T['sf_max_logdev']}")
        d = s.diag(deng)
        if d is not None:
            g = pd.read_csv(d / "genes.csv").set_index("id")
            tr = s.truth.loc[g.index]
            out_g = out_all.loc[g.index].to_numpy()
            m = tr["mu"].to_numpy() >= T["lfc_mu_min"]
            ref = tr["disp_trend"].to_numpy()[m] * math.exp(sig2 / 2)
            med = float(np.median(np.log(g["dispFit"].to_numpy()[m] / ref)))
            st_ = "INFO" if has_out else verdict(abs(med) <= T["disp_trend_logratio_max"])
            rep.add("recovery", "DESeq2 dispersion trend vs E[true disp | mean]", name, deng, st_,
                    f"median log ratio={fmt(med)}", f"|.| <= {T['disp_trend_logratio_max']}",
                    OUT_NOTE if has_out else "genes with true mean >= 50")
            gw = np.median(g["dispGeneEst"].to_numpy()[m & ~out_g] / tr["disp"].to_numpy()[m & ~out_g])
            rep.add("recovery", "DESeq2 gene-wise dispersion / true, median (info)", name, deng, "INFO", fmt(gw),
                    note="below 1 at small n: the MLE of a variance is biased down")
            lr2 = np.log(g["dispersion"].to_numpy()[m] / tr["disp"].to_numpy()[m])
            rho = stats.spearmanr(g["dispersion"].to_numpy()[m], tr["disp"].to_numpy()[m]).statistic
            sj = json.loads((d / "scalars.json").read_text()) if (d / "scalars.json").exists() else {}
            rep.add("recovery", "DESeq2 MAP dispersion vs true (info)", name, deng, "INFO",
                    f"median log ratio={fmt(float(np.median(lr2)))} spearman={fmt(rho)} "
                    f"dispPriorVar={fmt(sj.get('dispPriorVar'))} (true sigma^2={sig2:.2f})")
        d = s.diag("edger")
        if d is not None and (d / "disp.csv").exists():
            g = pd.read_csv(d / "disp.csv").set_index("id")
            tr = s.truth.loc[g.index]
            m = tr["mu"].to_numpy() >= T["lfc_mu_min"]
            ref = tr["disp_trend"].to_numpy()[m] * math.exp(sig2 / 2)
            med = float(np.median(np.log(g["trended_disp"].to_numpy()[m] / ref)))
            st_ = "INFO" if has_out else verdict(abs(med) <= T["disp_trend_logratio_max"])
            rep.add("recovery", "edgeR dispersion trend vs E[true disp | mean]", name, "edger", st_,
                    f"median log ratio={fmt(med)}", f"|.| <= {T['disp_trend_logratio_max']}",
                    OUT_NOTE if has_out else "genes with true mean >= 50")
        if s.params.get("mode") == "anova":
            continue
        for eng in ("edger", "deseq2"):
            o = s.out(eng)
            if o is None:
                continue
            for lab in s.labels:
                gid = ids(o)
                est = o[f"Log2FC {lab}"].to_numpy(float)
                tl = true_lfc(s, lab, gid)
                mu = s.truth.loc[gid, "mu"].to_numpy()
                hm = np.isfinite(est) & (mu >= T["lfc_mu_min"])
                for sub, msk in (("null", hm & (tl == 0)), ("DE", hm & (tl != 0))):
                    dlt = est[msk] - tl[msk]
                    b = float(np.mean(dlt))
                    se = float(np.std(dlt, ddof=1) / math.sqrt(msk.sum()))
                    rep.add("recovery", f"LFC bias ({sub} genes, mu>=50)", f"{name} {lab}", eng,
                            verdict(abs(b) <= T["lfc_bias_max"]), f"{fmt(b)} (se {fmt(se)}, n={msk.sum()})",
                            f"|.| <= {T['lfc_bias_max']}")
                lo, hi = o[f"CILeft {lab}"].to_numpy(float), o[f"CIRight {lab}"].to_numpy(float)
                inside = (lo <= tl) & (tl <= hi)
                okall = np.isfinite(lo) & np.isfinite(hi)
                okc = okall & ~out_all.loc[gid].to_numpy()
                cov = float(np.mean(inside[okc]))
                if has_out:
                    rep.add("recovery", "95% CI coverage incl. planted outlier genes (info)", f"{name} {lab}", eng,
                            "INFO", fmt(float(np.mean(inside[okall]))))
                covh = float(np.mean(inside[okc & (mu >= T["lfc_mu_min"])]))
                covd = float(np.mean(inside[okc & (tl != 0)]))
                thr = T["ci_cover_min"][eng]
                rep.add("recovery", "95% CI coverage", f"{name} {lab}", eng, verdict(cov >= thr),
                        f"{fmt(cov)} (mu>=50: {fmt(covh)}, DE: {fmt(covd)}, n={okc.sum()})", f">= {thr}")


# --------------------------------------------------------------------------------------
# Shrinkage
# --------------------------------------------------------------------------------------
def shrink_sets(root, prefix):
    for name in MAIN:
        yield name, [Scenario(root, name, prefix)]
    yield "mix x50", [Scenario(root, f"mix_{i:03d}", prefix) for i in range(50)]


def check_shrinkage(rep, root, prefix, engines):
    T = THRESHOLDS
    sets = list(shrink_sets(root, prefix))
    for scope, scns in sets:
        base = [s.out("deseq2") for s in scns]
        if any(b is None for b in base):
            continue
        for eng in SHRINKS:
            if eng not in engines:
                continue
            outs = [s.out(eng) for s in scns]
            if any(o is None for o in outs):
                rep.add("shrinkage", "MSE vs MLE", scope, eng, "SKIP", "no output")
                continue
            e_mle, e_shr, cover, o_mle, o_shr = [], [], [], [], []
            for s, b, o in zip(scns, base, outs):
                for lab in s.labels:
                    gid = ids(o)
                    tl = true_lfc(s, lab, gid)
                    og = s.truth.loc[gid, "outlier"].astype(str).str.upper().eq("TRUE").to_numpy()
                    m = b[f"Log2FC {lab}"].reindex(gid).to_numpy(float)
                    h = o[f"Log2FC {lab}"].to_numpy(float)
                    ok = np.isfinite(m) & np.isfinite(h)
                    e_mle.append((m - tl)[ok & ~og])
                    e_shr.append((h - tl)[ok & ~og])
                    o_mle.append((m - tl)[ok & og])
                    o_shr.append((h - tl)[ok & og])
                    lo = o[f"CrILeft {lab}"].to_numpy(float)
                    hi = o[f"CrIRight {lab}"].to_numpy(float)
                    okc = np.isfinite(lo) & np.isfinite(hi)
                    cover.append(((lo <= tl) & (tl <= hi))[okc])
            em, es = np.concatenate(e_mle), np.concatenate(e_shr)
            ratio = float(np.mean(es**2) / np.mean(em**2))
            rep.add("shrinkage", "MSE(shrunk)/MSE(MLE) vs true LFC", scope, eng,
                    verdict(ratio < T["mse_ratio_max"]), f"{fmt(ratio)} (n={len(em)})", f"< {T['mse_ratio_max']}",
                    "planted outlier genes excluded")
            om, osh = np.concatenate(o_mle), np.concatenate(o_shr)
            if len(om):
                rep.add("shrinkage", "MSE ratio on planted outlier genes (info)", scope, eng, "INFO",
                        f"{fmt(float(np.mean(osh**2) / np.mean(om**2)))} (n={len(om)})",
                        note="documented: lfcShrink refits on counts(dds), the original counts, while the "
                             "MLE uses Cook's-replaced counts")
            cv = np.concatenate(cover)
            rep.add("shrinkage", "95% CrI coverage (info)", scope, eng, "INFO", f"{fmt(float(cv.mean()))} (n={len(cv)})")
    if "deseq2_ashr" not in engines:
        return
    L_all, E_all, Z_all, n_sc = [], [], [], 0
    for scope, scns in sets:
        for s in scns:
            d = s.diag("deseq2_ashr")
            if d is None:
                continue
            n_sc += 1
            for k, lab in enumerate(s.labels, 1):
                a = pd.read_csv(d / f"cmp{k}.csv").set_index("id").dropna(subset=["lfsr"])
                tl = true_lfc(s, lab, a.index)
                og = s.truth.loc[a.index, "outlier"].astype(str).str.upper().eq("TRUE").to_numpy()
                wrong = np.sign(a["PosteriorMean"].to_numpy()) != np.sign(tl)
                L_all.append(a["lfsr"].to_numpy()[~og])
                E_all.append(wrong[~og])
                Z_all.append((tl == 0)[~og])
    if n_sc == 0:
        rep.add("shrinkage", "ashr lfsr calibration", "all", "deseq2_ashr", "SKIP", "not exposed")
        return
    L, E, Zr = np.concatenate(L_all), np.concatenate(E_all), np.concatenate(Z_all)
    rep.add("shrinkage", f"true-zero fraction among lfsr<={T['lfsr_cut']} (info)", "all scenarios", "deseq2_ashr",
            "INFO", fmt(float(Zr[L <= T["lfsr_cut"]].mean())),
            note="documented: method='shrink' has no point mass at 0, so its lfsr is a sign error rate "
                 "only and says nothing about true zeros (pi0 = 0.8 here)")
    L, E = L[~Zr], E[~Zr]
    S = L <= T["lfsr_cut"]
    est, real = float(L[S].mean()), float(E[S].mean())
    ub = est + T["lfsr_z"] * math.sqrt(max(est * (1 - est), 1e-12) / S.sum())
    rep.add("shrinkage", f"ashr realised false sign rate among lfsr<={T['lfsr_cut']}", "all scenarios",
            "deseq2_ashr", verdict(real <= ub), f"realised {fmt(real)} vs mean lfsr {fmt(est)} (n={S.sum()})",
            f"<= {fmt(ub)}", "genes with true LFC != 0; planted outliers excluded")
    bins = [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5001]
    parts = []
    for lo, hi in itertools.pairwise(bins):
        b = (L > lo) & (L <= hi) if lo > 0 else (L <= hi)
        if b.sum():
            parts.append(f"({lo},{hi:.2f}] {fmt(float(L[b].mean()), 2)}->{fmt(float(E[b].mean()), 2)} n={b.sum()}")
    rep.add("shrinkage", "ashr lfsr calibration curve: mean lfsr -> realised (info)", "all scenarios",
            "deseq2_ashr", "INFO", "; ".join(parts))


# --------------------------------------------------------------------------------------
# Certificates
# --------------------------------------------------------------------------------------
def robust_mom_disp(nc, X):
    """DESeq2 robustMethodOfMomentsDisp on normalised counts (Cook's distance variance)."""
    cells = pd.Series(["|".join(f"{v:g}" for v in row) for row in X])
    cnt = cells.value_counts()
    big = set(cnt[cnt >= 3].index)
    if big:
        trimr = {1: 1 / 3, 2: 1 / 4, 3: 1 / 8}
        scl = {1: 2.04, 2: 1.86, 3: 1.51}
        vs = []
        for lv in [c for c in pd.unique(cells) if c in big]:
            idx = np.flatnonzero((cells == lv).to_numpy())
            n = len(idx)
            b = 1 if n <= 3.5 else (2 if n <= 23.5 else 3)
            cm = r_trim_mean(nc[:, idx], trimr[b])
            sq = (nc[:, idx] - cm[:, None]) ** 2
            vs.append(scl[b] * r_trim_mean(sq, trimr[b]))
        v = np.max(np.vstack(vs), axis=0)
    else:
        rmean = r_trim_mean(nc, 1 / 8)
        v = 1.51 * r_trim_mean((nc - rmean[:, None]) ** 2, 1 / 8)
    mbar = nc.mean(axis=1)
    return np.maximum((v - mbar) / mbar**2, 0.04), cells, big


def cert_deseq2(rep, s, eng="deseq2"):
    T = THRESHOLDS
    d = s.diag(eng)
    if d is None:
        rep.add("certificate", "DESeq2 certificates", s.name, eng, "SKIP", "not exposed")
        return
    g = pd.read_csv(d / "genes.csv").set_index("id")
    sc = json.loads((d / "scalars.json").read_text())
    X, reps, _ = design_from(d / "design.csv")
    sf = pd.read_csv(d / "samples.csv").set_index("replicate").loc[reps, "size_factor"].to_numpy()
    rn = sc["results_names"]
    y = s.counts.loc[g.index, reps].to_numpy(float)
    yfit = y.copy()
    if (d / "replaced_counts.csv").exists():
        rc = pd.read_csv(d / "replaced_counts.csv").set_index("id")[reps]
        yfit[g.index.get_indexer(rc.index)] = rc.to_numpy(float)
    beta2 = g[[f"coef_{r}" for r in rn]].to_numpy(float)
    beta = beta2 * LN2
    alpha = g["dispersion"].to_numpy(float)
    mu = sf * np.exp(beta @ X.T)
    ridge = 1e-6 / LN2**2
    bc = g["betaConv"] if "betaConv" in g else pd.Series(True, index=g.index)
    conv = bc.astype(str).str.upper().eq("TRUE").to_numpy() & (np.max(np.abs(beta2), axis=1) < 20)
    _, z = nb_score(yfit, mu, alpha, X, beta, ridge)
    az = np.abs(z).max(axis=1)
    edge = mu.min(axis=1) < T["boundary_mu"]
    inner = conv & ~edge
    zmax = float(np.max(az[inner]))
    rep.add("certificate", "NB score ~ 0 at IRLS MLE", s.name, eng, verdict(zmax <= T["score_z_max"]),
            f"max|score/sqrt(I)|={fmt(zmax)} (q99.9 {fmt(float(np.quantile(az[inner], 0.999)))}, "
            f"{inner.sum()} genes; {edge.sum()} near boundary max {fmt(float(az[edge].max()) if edge.any() else 0.0)})",
            f"<= {T['score_z_max']}", "ridge 1e-6 (log2) included; near boundary = a fitted mean < 0.5")
    conv = inner & (az <= T["score_tight"])
    w = mu / (1 + alpha[:, None] * mu)
    A = xtwx(w, X)
    cov = np.linalg.inv(A + np.eye(X.shape[1]) * ridge)
    se = np.sqrt(np.einsum("gkk->gk", cov @ A @ cov)) / LN2
    se_rep = g[[f"SE_{r}" for r in rn]].to_numpy(float)
    e = max_rel(se[conv], se_rep[conv])
    rep.add("certificate", "SE = sqrt(diag((X'WX)^-1)) / ln2 at the MLE", s.name, eng, verdict(e <= T["se_rel"]),
            f"max rel={fmt(e)} ({conv.sum()} genes)", f"<= {T['se_rel']}", "genes whose score z <= 1e-4")
    if s.params.get("mode") == "anova":
        return
    repl = g["replace"].astype(str).str.upper().eq("TRUE").to_numpy()
    cooks_rep = pd.read_csv(d / "cooks.csv").set_index("id").loc[g.index, reps].to_numpy(float)
    p = X.shape[1]
    arob, cells, big = robust_mom_disp(y / sf, X)
    H = np.einsum("gm,mk,gkl,ml->gm", w, X, cov, X)
    V = mu + arob[:, None] * mu**2
    cooks = (y - mu) ** 2 / V / p * H / (1 - H) ** 2
    # Rows that hit the IRLS iteration cap go to DESeq2's optim fallback, which updates beta, SE
    # and mu but keeps the IRLS hat diagonals (fitNbinomGLMs.R), so their Cook's uses a stale H.
    it = pd.to_numeric(g["betaIter"], errors="coerce").to_numpy() if "betaIter" in g else np.zeros(len(g))
    optim_rows = np.nan_to_num(it, nan=0) >= 100
    if optim_rows.any():
        rep.add("certificate", "Cook's skipped on optim-fallback rows (info)", s.name, eng, "INFO",
                f"{int(optim_rows.sum())} genes", note="documented: DESeq2 keeps the IRLS hat diagonals after optim")
    keep = ~repl & conv & ~optim_rows
    e = max_rel(cooks[keep], cooks_rep[keep], floor=1e-8)
    rep.add("certificate", "Cook's distance recomputed (robust MoM disp, hat diag)", s.name, eng,
            verdict(e <= T["cooks_rel"]), f"max rel={fmt(e)} ({keep.sum()} genes; refit genes skipped)",
            f"<= {T['cooks_rel']}")
    m, cutoff = sc["m"], sc["cooks_cutoff"]
    o = s.out(eng)
    if not (m > p and cutoff is not None and o is not None):
        rep.add("certificate", "Cook's flagging", s.name, eng, "INFO", "not applied (m <= p or no cell with >= 3)")
        return
    elig = cells.isin(big).to_numpy()
    if not elig.any():
        rep.add("certificate", "Cook's flagging", s.name, eng, "INFO", "no cell with >= 3 samples: maxCooks NA")
        return
    if cells.value_counts().min() >= 7:
        flag = np.any(cooks_rep[:, elig] > cutoff, axis=1)
        agree = float(np.mean(flag == repl))
        rep.add("certificate", "replaced genes <=> some Cook's > qf(.99,p,m-p)", s.name, eng,
                verdict(agree == 1.0), f"agreement={fmt(agree)} ({repl.sum()} replaced)", "== 1")
        out_g = s.truth.loc[g.index, "outlier"].astype(str).str.upper().eq("TRUE").to_numpy()
        if out_g.any():
            rep.add("recovery", "planted outliers replaced (info)", s.name, eng, "INFO",
                    f"recall={fmt(float(np.mean(repl[out_g])))} ({out_g.sum()} planted), "
                    f"false flag rate={fmt(float(np.mean(repl[~out_g])))}")
    else:
        maxc = np.max(cooks_rep[:, elig], axis=1)
        pna = o.loc[g.index, f"PValue {s.labels[0]}"].isna().to_numpy()
        flag = maxc > cutoff
        agree = float(np.mean(flag == pna))
        rep.add("certificate", "PValue NA <=> maxCooks > qf(.99,p,m-p)", s.name, eng,
                verdict(agree == 1.0), f"agreement={fmt(agree)} ({flag.sum()} flagged)", "== 1")


def cert_wald(rep, s, eng="deseq2"):
    T = THRESHOLDS
    o = s.out(eng)
    if o is None:
        return
    for lab in s.labels:
        lfc, se, st = (o[f"{c} {lab}"].to_numpy(float) for c in ("Log2FC", "SE", "stat"))
        p = o[f"PValue {lab}"].to_numpy(float)
        ok = np.isfinite(st)
        e1 = max_rel(lfc[ok] / se[ok], st[ok])
        okp = ok & np.isfinite(p)
        e2 = p_err(2 * stats.norm.sf(np.abs(st[okp])), p[okp])
        lo, hi = o[f"CILeft {lab}"].to_numpy(float), o[f"CIRight {lab}"].to_numpy(float)
        e3 = max(max_rel(lfc - Z975 * se, lo, 1e-8), max_rel(lfc + Z975 * se, hi, 1e-8))
        good = e1 <= T["wald_rel"] and e2 <= T["p_rel"] and e3 <= T["p_rel"]
        rep.add("certificate", "Wald: stat=LFC/SE, p=2*Phi(-|stat|), CI=LFC+-z*SE", f"{s.name} {lab}", eng,
                verdict(good), f"rel {fmt(e1)} / {fmt(e2)} / {fmt(e3)}",
                f"<= {T['wald_rel']} / {T['p_rel']} / {T['p_rel']}")


def cert_lfc_reparam(rep, s, eng="deseq2"):
    """A relevelled comparison's LFC equals the contrast of the base fit (same MLE, other basis)."""
    d, o = s.diag(eng), s.out(eng)
    if d is None or o is None or len(s.labels) < 3:
        return
    g = pd.read_csv(d / "genes.csv").set_index("id")
    repl = g["replace"].astype(str).str.upper().eq("TRUE").to_numpy()
    sc = json.loads((d / "scalars.json").read_text())
    rn = sc["results_names"]
    base = [r for r in rn if r.startswith("condition_")]
    lv = {r.split("_")[1]: r for r in base}
    for lab in s.labels:
        left, right = lab.split(" - ")
        if right != base[0].split("_vs_")[1]:
            c = g[f"coef_{lv[left]}"] - g[f"coef_{lv[right]}"]
            t = o.loc[g.index, f"Log2FC {lab}"]
            ok = ~repl & np.isfinite(t.to_numpy()) & (np.abs(c.to_numpy()) < 20)
            e = float(np.max(np.abs(c.to_numpy()[ok] - t.to_numpy()[ok])))
            rep.add("certificate", "relevelled LFC = contrast of base fit (log2)", f"{s.name} {lab}", eng,
                    verdict(e <= 1e-4), f"max abs diff={fmt(e)}", "<= 1e-4",
                    "two IRLS solutions to the same likelihood; tolerance = IRLS convergence")


def cert_shrink_tables(rep, s, eng):
    """Shrinkage keeps deseq2's stat/p-values; normal and ashr CrI = shrunk LFC +- z*SE."""
    T = THRESHOLDS
    o, b = s.out(eng), s.out("deseq2")
    if o is None or b is None:
        return
    for lab in s.labels:
        bs = b[f"stat {lab}"].reindex(o.index)
        bp = b[f"PValue {lab}"].reindex(o.index)
        e1 = max_rel(o[f"stat {lab}"], bs)
        e2 = max_rel(o[f"PValue {lab}"], bp, 1e-300)
        na_same = bool((o[f"PValue {lab}"].isna() == bp.isna()).all())
        good = e1 <= T["wald_rel"] and e2 <= T["wald_rel"] and na_same
        rep.add("certificate", "unshrunk stat/p kept under shrinkage", f"{s.name} {lab}", eng, verdict(good),
                f"rel {fmt(e1)} / {fmt(e2)}, NA pattern same={na_same}")
        if eng != "deseq2_apeglm":
            lfc, se = o[f"Log2FC {lab}"].to_numpy(float), o[f"SE {lab}"].to_numpy(float)
            lo, hi = o[f"CrILeft {lab}"].to_numpy(float), o[f"CrIRight {lab}"].to_numpy(float)
            e3 = max(max_rel(lfc - Z975 * se, lo, 1e-8), max_rel(lfc + Z975 * se, hi, 1e-8))
            rep.add("certificate", "CrI = shrunk LFC +- z*SE", f"{s.name} {lab}", eng, verdict(e3 <= T["p_rel"]),
                    f"max rel={fmt(e3)}", f"<= {T['p_rel']}")


def apeglm_prior_var(lfc_ln, se_ln, min_var=0.001**2, max_var=20.0**2):
    keep = np.isfinite(lfc_ln)
    X, D = lfc_ln[keep], se_ln[keep] ** 2
    S = X**2

    def obj(A):
        I = 1 / (2 * (A + D) ** 2)
        return np.sum((S - D) * I) / np.sum(I) - A

    if obj(min_var) < 0:
        return min_var
    return optimize.brentq(obj, min_var, max_var, xtol=1e-14, rtol=1e-14)


def cert_apeglm(rep, s):
    T = THRESHOLDS
    d, base = s.diag("deseq2_apeglm"), s.diag("deseq2")
    o, b = s.out("deseq2_apeglm"), s.out("deseq2")
    if d is None or base is None or o is None:
        rep.add("certificate", "apeglm certificates", s.name, "deseq2_apeglm", "SKIP", "not exposed")
        return
    g = pd.read_csv(base / "genes.csv").set_index("id")
    for k, lab in enumerate(s.labels, 1):
        meta = json.loads((d / f"cmp{k}.json").read_text())
        a = pd.read_csv(d / f"cmp{k}.csv").set_index("id")
        X, reps, _ = design_from(d / f"cmp{k}_design.csv")
        sf = pd.read_csv(base / "samples.csv").set_index("replicate").loc[reps, "size_factor"].to_numpy()
        y = s.counts.loc[a.index, reps].to_numpy(float)
        size = (1 / g.loc[a.index, "dispersion"].to_numpy(float))[:, None]
        mp = a[[c for c in a.columns if c.startswith("map_")]].to_numpy(float)
        ci = meta["coef_index"] - 1
        S2 = meta["prior_scale"] ** 2
        sig2 = meta["no_shrink_scale"] ** 2
        e = np.exp(mp @ X.T) * sf
        grad = (y - (y + size) * e / (size + e)) @ X
        prior = -mp / sig2
        prior[:, ci] = -2 * mp[:, ci] / (S2 + mp[:, ci] ** 2)
        grad = grad + prior
        w = (y + size) * e * size / (size + e) ** 2
        hd = w @ (X * X)
        hd[:, ci] += 2 * (S2 - mp[:, ci] ** 2) / (S2 + mp[:, ci] ** 2) ** 2
        hd[:, np.arange(X.shape[1]) != ci] += 1 / sig2
        # Full Hessian of the negative log posterior, Newton step and decrement.
        p_ = X.shape[1]
        H = np.einsum("gi,ij,ik->gjk", w, X, X)
        for j in range(p_):
            H[:, j, j] += (1 / sig2) if j != ci else 2 * (S2 - mp[:, ci] ** 2) / (S2 + mp[:, ci] ** 2) ** 2
        okH = np.isfinite(H).all(axis=(1, 2)) & np.isfinite(grad).all(axis=1)
        step = np.full_like(mp, np.nan)
        step[okH] = np.linalg.solve(H[okH], -grad[okH][:, :, None])[:, :, 0]
        dec = -0.5 * (grad * step).sum(axis=1)
        # apeglm's C++ L-BFGS minimises the posterior divided by cnst = max(f(0), 1) + 10 with
        # eps_f = 1e-8, so the decrement is only small relative to cnst.
        xb0 = np.zeros_like(mp) @ X.T
        f0 = -(y * xb0 - (y + size) * np.log(size + np.exp(xb0) * sf)).sum(axis=1)
        cn = np.maximum(f0, 1.0)
        conv = (a["conv"].to_numpy() == 0) & okH
        rd = dec[conv] / cn[conv]
        sdz = np.abs(step[conv, ci]) / a["sd"].to_numpy(float)[conv]
        rep.add("certificate", "apeglm MAP stationary (Newton decrement / cnst)", f"{s.name} {lab}",
                "deseq2_apeglm", verdict(float(np.max(rd)) <= T["apeglm_newton_dec_max"]),
                f"max={fmt(float(np.max(rd)))} q99={fmt(float(np.quantile(rd, 0.99)))} "
                f"({conv.sum()} genes, {(~conv).sum()} conv!=0)",
                f"<= {T['apeglm_newton_dec_max']}", "Y = original counts, offset log(sf), Cauchy(0,S) on coef")
        rep.add("certificate", "apeglm MAP distance to exact mode, posterior sd units (info)",
                f"{s.name} {lab}", "deseq2_apeglm", "INFO",
                f"median {fmt(float(np.median(sdz)))} q99 {fmt(float(np.quantile(sdz, 0.99)))} "
                f"max {fmt(float(np.max(sdz)))}", note="documented: L-BFGS stops on the scaled objective")
        if b is not None:
            mle = b[f"Log2FC {lab}"].reindex(a.index).to_numpy(float) * LN2
            sem = b[f"SE {lab}"].reindex(a.index).to_numpy(float) * LN2
            pv = apeglm_prior_var(mle, sem)
            err = abs(pv - meta["prior_var"])
            scale_ok = abs(min(math.sqrt(pv), 1.0) - meta["prior_scale"]) <= 1e-3
            rep.add("certificate", "apeglm prior variance (Efron-Morris) recomputed", f"{s.name} {lab}",
                    "deseq2_apeglm", verdict(err <= T["prior_var_abs"] and scale_ok),
                    f"|dA|={fmt(err)} (A={fmt(pv)}, S={fmt(meta['prior_scale'])})", f"<= {T['prior_var_abs']}")
        lfc = o.loc[a.index, f"Log2FC {lab}"].to_numpy(float)
        se = o.loc[a.index, f"SE {lab}"].to_numpy(float)
        sd = a["sd"].to_numpy(float)
        e1 = max_rel(mp[:, ci] / LN2, lfc, 1e-8)
        e2 = max_rel(sd / LN2, se, 1e-8)
        lo = o.loc[a.index, f"CrILeft {lab}"].to_numpy(float)
        hi = o.loc[a.index, f"CrIRight {lab}"].to_numpy(float)
        e3 = max(max_rel((mp[:, ci] - Z975 * sd) / LN2, lo, 1e-8), max_rel((mp[:, ci] + Z975 * sd) / LN2, hi, 1e-8))
        rep.add("certificate", "apeglm table = MAP, sd, MAP +- z*sd (log2)", f"{s.name} {lab}", "deseq2_apeglm",
                verdict(max(e1, e2, e3) <= T["p_rel"]), f"rel {fmt(e1)} / {fmt(e2)} / {fmt(e3)}", f"<= {T['p_rel']}")


def cert_ashr(rep, s):
    T = THRESHOLDS
    d, o = s.diag("deseq2_ashr"), s.out("deseq2_ashr")
    if d is None or o is None:
        rep.add("certificate", "ashr certificates", s.name, "deseq2_ashr", "SKIP", "not exposed")
        return
    for k, lab in enumerate(s.labels, 1):
        a = pd.read_csv(d / f"cmp{k}.csv").set_index("id")
        g = pd.read_csv(d / f"cmp{k}_g.csv")
        a = a[np.isfinite(a["betahat"]) & np.isfinite(a["sebetahat"])]
        bhat, se = a["betahat"].to_numpy(), a["sebetahat"].to_numpy()
        pi, mk, sk = g["pi"].to_numpy(), g["mean"].to_numpy(), g["sd"].to_numpy()
        tot = np.sqrt(sk[None, :] ** 2 + se[:, None] ** 2)
        L = stats.norm.pdf(bhat[:, None], mk[None, :], tot)
        Lp = L @ pi
        gradk = (L / Lp[:, None]).mean(axis=0)
        viol = float(np.max(gradk) - 1)
        supp = pi > 1e-6 * pi.max()
        eq = float(np.max(np.abs(gradk[supp] - 1)))
        rep.add("certificate", "mixture MLE KKT (max_k grad<=1, =1 on support)", f"{s.name} {lab}", "deseq2_ashr",
                verdict(viol <= T["kkt_tol"] and eq <= T["kkt_tol"]),
                f"max excess={fmt(viol)}, support dev={fmt(eq)} ({supp.sum()}/{len(pi)} comps)",
                f"<= {T['kkt_tol']}", f"sum(pi)={float(pi.sum()):.10f}")
        post_w = pi[None, :] * L / Lp[:, None]
        v = sk**2
        s2 = se[:, None] ** 2
        pm_k = (mk[None, :] * s2 + bhat[:, None] * v[None, :]) / (v[None, :] + s2)
        pv_k = v[None, :] * s2 / (v[None, :] + s2)
        pm = (post_w * pm_k).sum(1)
        psd = np.sqrt(np.maximum((post_w * (pv_k + pm_k**2)).sum(1) - pm**2, 0))
        pos = pv_k > 0
        negk = np.where(pos, stats.norm.cdf(0, pm_k, np.sqrt(np.where(pos, pv_k, 1.0))), (pm_k < 0) * 1.0)
        zerok = np.where(pos, 0.0, (pm_k == 0) * 1.0)
        neg, zero = (post_w * negk).sum(1), (post_w * zerok).sum(1)
        lfsr = np.where(neg > 0.5 * (1 - zero), 1 - neg, neg + zero)
        e1 = max_rel(pm, a["PosteriorMean"].to_numpy(), 1e-8)
        e2 = max_rel(psd, a["PosteriorSD"].to_numpy(), 1e-8)
        e3 = max_rel(lfsr, a["lfsr"].to_numpy(), 1e-8)
        e4 = max_rel(a["PosteriorMean"].to_numpy(), o.loc[a.index, f"Log2FC {lab}"].to_numpy(float), 1e-8)
        rep.add("certificate", "ashr PosteriorMean/SD/lfsr from g; table LFC = PosteriorMean", f"{s.name} {lab}",
                "deseq2_ashr", verdict(max(e1, e2, e3, e4) <= T["ashr_post_rel"]),
                f"rel {fmt(e1)} / {fmt(e2)} / {fmt(e3)} / {fmt(e4)}", f"<= {T['ashr_post_rel']}")


def cert_edger(rep, s):
    T = THRESHOLDS
    o, d = s.out("edger"), s.diag("edger")
    if o is None:
        return
    for lab in s.labels:
        F, st = o[f"F {lab}"].to_numpy(float), o[f"stat {lab}"].to_numpy(float)
        lfc, se = o[f"Log2FC {lab}"].to_numpy(float), o[f"SE {lab}"].to_numpy(float)
        p, q = o[f"PValue {lab}"].to_numpy(float), o[f"AdjPValue {lab}"].to_numpy(float)
        ok = np.isfinite(p)
        fs = ok & np.isfinite(st)
        e1 = max_rel(st[fs] ** 2, F[fs], 1e-12)
        e2 = max_rel(bh(p[ok]), q[ok], 1e-300)
        okse = np.isfinite(se) & np.isfinite(st) & (st != 0)
        e3 = max_rel(lfc[okse] / se[okse], st[okse])
        rep.add("certificate", "edgeR stat^2=F, SE=LFC/stat, AdjP=BH(P)", f"{s.name} {lab}", "edger",
                verdict(max(e1, e2, e3) <= T["p_rel"]), f"rel {fmt(e1)} / {fmt(e2)} / {fmt(e3)}",
                f"<= {T['p_rel']}", f"{int(np.sum(ok & ~np.isfinite(st)))} genes with stat NaN (F<0)")
    if d is None:
        rep.add("certificate", "edgeR certificates", s.name, "edger", "SKIP", "not exposed")
        return
    g = pd.read_csv(d / "genes.csv").set_index("id")
    sc = json.loads((d / "scalars.json").read_text())
    X, reps, cols = design_from(d / "design.csv")
    sm = pd.read_csv(d / "samples.csv").set_index("replicate").loc[reps]
    off = np.log(sm["lib_size"].to_numpy() * sm["norm_factor"].to_numpy())
    y = s.counts.loc[g.index, reps].to_numpy(float)
    beta = g[[f"coef_{c}" for c in cols]].to_numpy(float)
    mu = np.exp(beta @ X.T + off)
    disp = sc["fit_dispersion"] / sc["ave_ql_dispersion"]
    nz = (np.abs(beta) < 20).all(axis=1) & (beta > -1e7).all(axis=1)
    _, z = nb_score(y, mu, disp, X, beta)
    zmax = float(np.max(np.abs(z[nz])))
    rep.add("certificate", "NB score ~ 0 at glmFit MLE (unshrunk coefs)", s.name, "edger",
            verdict(zmax <= T["score_z_max"]), f"max|score/sqrt(I)|={fmt(zmax)} ({nz.sum()} genes)",
            f"<= {T['score_z_max']}", "dispersion = fit$dispersion / average.ql.dispersion")
    dft = np.minimum(g["df_prior"].to_numpy() + g["df_residual_adj"].to_numpy(), sc["df_residual_total"])
    for lab in s.labels:
        F = o.loc[g.index, f"F {lab}"].to_numpy(float)
        p = o.loc[g.index, f"PValue {lab}"].to_numpy(float)
        e = p_err(stats.f.sf(F, 1, dft), p)
        rep.add("certificate", "edgeR QL p = pf(F, 1, min(df.prior+df.res.adj, sum df.res))", f"{s.name} {lab}",
                "edger", verdict(e <= T["p_rel"]), f"max rel={fmt(e)}", f"<= {T['p_rel']}")
    k = s.si["condition"].nunique()
    F = o.loc[g.index, "F"].to_numpy(float)
    p = o.loc[g.index, "PValue"].to_numpy(float)
    e = p_err(stats.f.sf(F, k - 1, dft), p)
    rep.add("certificate", "edgeR omnibus p = pf(F, k-1, df.total)", s.name, "edger", verdict(e <= T["p_rel"]),
            f"max rel={fmt(e)}", f"<= {T['p_rel']}")


def check_certificates(rep, root, prefix, engines, n_mix=5):
    scns = [Scenario(root, n, prefix) for n in MAIN + ["s3x3_anova"]]
    scns += [Scenario(root, f"mix_{i:03d}", prefix) for i in range(n_mix)]
    for s in scns:
        if s.params.get("mode") == "anova":
            if "deseq2_anova" in engines:
                cert_deseq2(rep, s, "deseq2_anova")
            if "edger" in engines:
                cert_edger(rep, s)
            continue
        if "deseq2" in engines:
            cert_deseq2(rep, s)
            cert_wald(rep, s)
            cert_lfc_reparam(rep, s)
        if "edger" in engines:
            cert_edger(rep, s)
        for eng in SHRINKS:
            if eng in engines:
                cert_shrink_tables(rep, s, eng)
        if "deseq2_apeglm" in engines:
            cert_apeglm(rep, s)
        if "deseq2_ashr" in engines:
            cert_ashr(rep, s)


# --------------------------------------------------------------------------------------
# Third opinion: agreement with R DESeq2 (directional only)
# --------------------------------------------------------------------------------------
def check_agreement(rep, root, prefix, ref_prefix="r"):
    for name in MAIN + ["mix x50"]:
        if name == "mix x50":
            scns = [Scenario(root, f"mix_{i:03d}", prefix) for i in range(50)]
        else:
            scns = [Scenario(root, name, prefix)]
        xl, yl, ql, rl = [], [], [], []
        for s in scns:
            o = s.out("deseq2")
            ref = Scenario(root, s.name, ref_prefix).out("deseq2")
            if o is None or ref is None:
                xl = []
                break
            for lab in s.labels:
                xl.append(o[f"Log2FC {lab}"].reindex(ref.index).to_numpy(float))
                yl.append(ref[f"Log2FC {lab}"].to_numpy(float))
                ql.append(o[f"AdjPValue {lab}"].reindex(ref.index).to_numpy(float))
                rl.append(ref[f"AdjPValue {lab}"].to_numpy(float))
        if not xl:
            continue
        x, y = np.concatenate(xl), np.concatenate(yl)
        ok = np.isfinite(x) & np.isfinite(y)
        rho = stats.pearsonr(x[ok], y[ok]).statistic
        qa, qb = np.concatenate(ql), np.concatenate(rl)
        ra, rb = qa <= 0.05, qb <= 0.05
        jac = (ra & rb).sum() / max((ra | rb).sum(), 1)
        sgn = float(np.mean(np.sign(x[ok & rb]) == np.sign(y[ok & rb]))) if (ok & rb).any() else float("nan")
        rep.add("agreement", "pydeseq2 vs R DESeq2 (directional)", name, "deseq2", "INFO",
                f"LFC pearson={fmt(rho)} median|dLFC|={fmt(float(np.median(np.abs(x[ok] - y[ok]))))} "
                f"padj<=.05 jaccard={fmt(jac)} (py {ra.sum()}, R {rb.sum()}) sign agree on R hits={fmt(sgn)}")


def run(engine="r", corpus=DEFAULT_CORPUS, groups=("calibration", "recovery", "shrinkage", "certificate"),
        rerun=True, out=None, quiet=False):
    prefix = PREFIX[engine]
    if engine == "rust" and rerun:
        # TRUTH_RUST_PYTHON: an interpreter with edge_rust and deseq2_rust installed.
        py = os.environ.get("TRUTH_RUST_PYTHON", sys.executable)
        subprocess.run([py, str(HERE / "run_rust.py"), "--corpus", str(corpus)], check=True)
    engines = ["deseq2"] if engine == "pydeseq2" else ["edger", "deseq2", "deseq2_anova"] + SHRINKS
    rep = Report()
    if "calibration" in groups:
        check_calibration(rep, corpus, prefix, engines)
    if "recovery" in groups:
        check_recovery(rep, corpus, prefix, engines)
    if "shrinkage" in groups and engine != "pydeseq2":
        check_shrinkage(rep, corpus, prefix, engines)
    if "certificate" in groups and engine != "pydeseq2":
        check_certificates(rep, corpus, prefix, engines)
    if engine == "pydeseq2":
        check_agreement(rep, corpus, prefix)
    t = rep.table()
    out = out or corpus / f"checks_{engine}.csv"
    t.to_csv(out, index=False)
    if not quiet:
        pd.set_option("display.width", 250, "display.max_colwidth", 110, "display.max_rows", 5000)
        for grp, sub in t.groupby("group", sort=False):
            print(f"\n## {grp}")
            print(sub.drop(columns="group").to_string(index=False))
        cnt = t["status"].value_counts().to_dict()
        print(f"\n{engine}: " + ", ".join(f"{k} {v}" for k, v in sorted(cnt.items())) + f"  -> {out}")
    return t


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=list(PREFIX), default="r")
    ap.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    ap.add_argument("--no-run", action="store_true", help="rust: check existing rust_*.csv without rerunning")
    ap.add_argument("--groups", default="calibration,recovery,shrinkage,certificate")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    t = run(a.engine, a.corpus, a.groups.split(","), not a.no_run, a.out)
    return 1 if (t["status"] == "FAIL").any() else 0


if __name__ == "__main__":
    sys.exit(main())
