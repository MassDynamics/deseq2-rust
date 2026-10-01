# /// script
# requires-python = ">=3.10"
# dependencies = ["pydeseq2", "pandas", "numpy"]
# ///
"""Third opinion: pydeseq2 on the truth corpus, same gene set as R's DESeq2 table.

Writes <scenario>/py_deseq2.csv (production column names) and py_deseq2.diag/{samples,genes}.csv.
Directional comparison only; pydeseq2 is a re-implementation, not a port.
"""
import json
import os
import sys
from multiprocessing import Pool
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from pydeseq2.dds import DeseqDataSet
from pydeseq2.default_inference import DefaultInference
from pydeseq2.ds import DeseqStats
from scipy import stats

warnings.filterwarnings("ignore")
CORPUS = Path.home() / "wd/md-count-truth-corpus"
Z = stats.norm.ppf(0.975)


def run_one(d: Path):
    p = json.loads((d / "params.json").read_text())
    if p.get("mode") == "anova":
        return "skip (anova)"
    ref = pd.read_csv(d / "r_deseq2.csv")["GroupId"].astype(str)
    counts = pd.read_csv(d / "counts.csv", index_col=0)
    counts.index = counts.index.astype(str)
    counts = counts.loc[ref]
    si = pd.read_csv(d / "sample_info.csv").set_index("replicate")
    cond = p["condition_col"]
    meta = si.loc[counts.columns].copy()
    terms = list(p.get("control_cols", [])) + [cond]
    dds = DeseqDataSet(counts=counts.T, metadata=meta, design="~" + " + ".join(terms),
                       inference=DefaultInference(n_cpus=1), quiet=True)
    dds.deseq2()
    out = pd.DataFrame({"GroupId": counts.index.astype(int)})
    for _, c in pd.read_csv(d / "comparisons.csv").iterrows():
        lab = f"{c['left']}{p['condition_separator']}{c['right']}"
        ds = DeseqStats(dds, contrast=[cond, c["left"], c["right"]], alpha=p["deseq2_alpha"],
                        inference=DefaultInference(n_cpus=1), quiet=True)
        ds.summary()
        r = ds.results_df.loc[counts.index]
        out[f"Log2FC {lab}"] = r["log2FoldChange"].to_numpy()
        out[f"stat {lab}"] = r["stat"].to_numpy()
        out[f"SE {lab}"] = r["lfcSE"].to_numpy()
        out[f"CILeft {lab}"] = (r["log2FoldChange"] - Z * r["lfcSE"]).to_numpy()
        out[f"CIRight {lab}"] = (r["log2FoldChange"] + Z * r["lfcSE"]).to_numpy()
        out[f"PValue {lab}"] = r["pvalue"].to_numpy()
        out[f"AdjPValue {lab}"] = r["padj"].to_numpy()
        out["AveExpr"] = r["baseMean"].to_numpy()
    out.to_csv(d / "py_deseq2.csv", index=False)
    dg = d / "py_deseq2.diag"
    dg.mkdir(exist_ok=True)
    pd.DataFrame({"replicate": counts.columns, "size_factor": dds.obs["size_factors"].to_numpy()
                  if "size_factors" in dds.obs else dds.obsm["size_factors"]}).to_csv(dg / "samples.csv", index=False)
    v = dds.var
    pd.DataFrame({"id": counts.index.astype(int), "baseMean": out["AveExpr"].to_numpy(),
                  "dispGeneEst": v["genewise_dispersions"].to_numpy(), "dispFit": v["fitted_dispersions"].to_numpy(),
                  "dispersion": v["dispersions"].to_numpy()}).to_csv(dg / "genes.csv", index=False)
    return "ok"


def safe(n):
    try:
        return n, run_one(CORPUS / n)
    except Exception as e:  # keep going, report at the end
        return n, f"FAIL {e!r}"


def main(argv):
    names = argv or sorted(x.name for x in CORPUS.iterdir() if (x / "r_deseq2.csv").exists())
    bad = 0
    with Pool(max(1, min(8, (os.cpu_count() or 2) - 2))) as pool:
        for n, msg in pool.imap_unordered(safe, names):
            bad += msg.startswith("FAIL")
            print(n, msg, flush=True)
    print(f"pydeseq2: {len(names) - bad} / {len(names)} ok")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
