"""Run the Rust engines (edge_rust, deseq2_rust) on the truth corpus.

Writes <scenario>/rust_<engine>.csv with the production table, the same shape as r_<engine>.csv,
and <scenario>/rust_edger.diag/ in the layout of r_edger.diag/ (see write_edger_diag).
rust_edger (discovery table on every scenario, as R), rust_deseq2 (or rust_deseq2_anova), rust_deseq2_{normal,apeglm,ashr} on the main
scenarios. Null reps get edger + deseq2 only, as for R. Run with the Python that has both packages
installed (e.g. `uv run --with <wheel> ...`); check_truth.py --engine rust calls this file with
its own interpreter.

Inputs follow the engine contract run(counts, sample_info, comparisons, params) -> DataFrame.
`to_inputs` is the only place that knows the shapes, so adapt it there if the contract moves.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

MAIN = {"s2x3", "s2x4", "s3x3", "s2x7_outliers", "s2x3_batch", "s2x4_covariate"}
SHRINKS = ["normal", "apeglm", "ashr"]


def to_inputs(d: Path, de_method: str, shrink: str = "none"):
    counts = pd.read_csv(d / "counts.csv", index_col=0)
    counts.index = counts.index.rename("GroupId")
    si = pd.read_csv(d / "sample_info.csv")
    comps = pd.read_csv(d / "comparisons.csv")
    p = json.loads((d / "params.json").read_text())
    params = {
        "condition_col": p["condition_col"],
        "control_cols": [{"Column": c, "Type": "numerical" if c == "covariate" else "categorical"}
                         for c in p.get("control_cols", [])],
        "mode": p["mode"],
        "comparison_type": "custom",
        "custom_comparisons": comps[["left", "right"]].to_dict("records"),
        "de_method": de_method,
        "edger_norm_method": p["edger_norm_method"],
        "deseq2_alpha": p["deseq2_alpha"],
        "deseq2_lfc_shrinkage": shrink,
        "apeglm_seed": p["apeglm_seed"],
    }
    return counts, si, comps, params


def write(df: pd.DataFrame, path: Path):
    df = df.copy()
    if "GroupId" not in df.columns:
        df = df.reset_index().rename(columns={"index": "GroupId"})
    df["GroupId"] = df["GroupId"].astype(int)
    df.sort_values("GroupId").to_csv(path, index=False)


def write_edger_diag(diag: dict, d: Path):
    """r_edger.diag/ as run_r.R writes it: samples, genes, design, disp (.csv), scalars.json."""
    d.mkdir(exist_ok=True)
    for k in ("samples", "genes", "design", "disp"):
        diag[k].to_csv(d / f"{k}.csv", index=False)
    sc = {k: ({} if v is None else v) for k, v in diag["scalars"].items()}  # NULL as R writes it
    (d / "scalars.json").write_text(json.dumps(sc, indent=2))


def run_scenario(d: Path, edge_rust, deseq2_rust):
    p = json.loads((d / "params.json").read_text())
    anova = p["mode"] == "anova"
    if edge_rust is not None:
        # r_edger.csv is the discovery table on every scenario (anova included), so match it.
        c, si, comps, params = to_inputs(d, "edgeR")
        table, diag = edge_rust.run(c, si, comps, {**params, "mode": "discovery"}, diagnostics=True)
        write(table, d / "rust_edger.csv")
        write_edger_diag(diag, d / "rust_edger.diag")
    if deseq2_rust is None:
        return
    write(deseq2_rust.run(*to_inputs(d, "DESeq2")), d / ("rust_deseq2_anova.csv" if anova else "rust_deseq2.csv"))
    if d.name in MAIN or d.name.startswith("mix_"):
        for s in SHRINKS:
            write(deseq2_rust.run(*to_inputs(d, "DESeq2", s)), d / f"rust_deseq2_{s}.csv")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=Path.home() / "wd/md-count-truth-corpus")
    ap.add_argument("--engines", default="edger,deseq2", help="comma list of edger, deseq2")
    ap.add_argument("names", nargs="*")
    a = ap.parse_args(argv)
    eng = set(a.engines.split(","))
    edge_rust = __import__("edge_rust") if "edger" in eng else None
    deseq2_rust = __import__("deseq2_rust") if "deseq2" in eng else None

    names = a.names or list(json.loads((a.corpus / "index.json").read_text()))
    bad = 0
    for n in names:
        try:
            run_scenario(a.corpus / n, edge_rust, deseq2_rust)
            print(n, "ok", flush=True)
        except Exception as e:  # report and continue; the exit code carries the failure
            bad += 1
            print(n, "FAIL", repr(e), flush=True)
    print(f"rust: {len(names) - bad} / {len(names)} scenarios ok")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
