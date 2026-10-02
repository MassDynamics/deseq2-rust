"""The production DESeq2 table around the Rust engine.

Mirrors what MDFlexiComparisons does after the engine call: the left join of the engine table to
every gene id (``merge(..., all.x = TRUE)``, which orders rows by GroupId as a string), the
integer GroupId, and for ANOVA runs ``.packageANOVAOutput`` (``R/runANOVA.R``): the omnibus
columns plus ``MaxLog2FCPair`` / ``MaxLog2FC``, every column as a string with NA written as "".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from deseq2_rust import _core

PAIR_STATS = [
    "Log2FC",
    "stat",
    "SE",
    "CILeft",
    "CIRight",
    "CrILeft",
    "CrIRight",
    "PValue",
    "AdjPValue",
]
ANOVA_COLUMNS = ["GroupId", "AveExpr", "PValue", "AdjPValue", "LRT", "MaxLog2FCPair", "MaxLog2FC"]


def _control_specs(control_cols) -> list[tuple[str, str]]:
    """``control_cols`` as ``{Column, Type}`` (scalars or lists, as the job params carry it),
    a list of such dicts, or None."""
    if control_cols is None:
        return []
    if isinstance(control_cols, dict):
        cols, types = control_cols["Column"], control_cols["Type"]
        if isinstance(cols, str):
            cols, types = [cols], [types]
        return list(zip(cols, types))
    return [(c["Column"], c["Type"]) for c in control_cols]


def _r_character(x: float) -> str:
    """``as.character`` of a double: 15 significant digits, NA as ""."""
    return "" if np.isnan(x) else f"{x:.15g}"


def run(
    counts: pd.DataFrame,
    sample_info: pd.DataFrame,
    comparisons: pd.DataFrame,
    params: dict,
    diagnostics: bool = False,
):
    """Run the DESeq2 engine and return the production output table.

    counts: genes x samples, index = gene ids, columns = sample ids; non-negative integers.
    sample_info: one row per sample, sample ids in a ``replicate`` column or the index, the
        condition column and the control columns.
    comparisons: ``left``, ``right`` (output labels) and optionally ``encoded_left``,
        ``encoded_right`` (the condition values in ``sample_info``; default: left / right).
    params: ``condition_col`` (default "condition"), ``control_cols`` (``{Column, Type}``),
        ``mode`` ("discovery" or "anova"), ``deseq2_alpha`` (default 0.05),
        ``deseq2_lfc_shrinkage`` ("none", "normal", "apeglm" or "ashr"), ``entity_type``
        (default "gene"). ``apeglm_seed`` is accepted and ignored: the apeglm path used here
        draws no random numbers.

    diagnostics: also return the fit's intermediates, as ``(table, diag)``; see ``_diag``.

    Raises ValueError with the production message when the engine refuses the input.
    """
    cc = params.get("condition_col", "condition")
    si = sample_info.set_index("replicate") if "replicate" in sample_info.columns else sample_info
    si.index = si.index.astype(str)
    sample_ids = [str(s) for s in counts.columns]
    missing = set(sample_ids) - set(si.index)
    if missing:
        raise ValueError(f"samples missing from sample_info: {sorted(missing)}")
    # sampleInfo[colnames(countMatrix), ]: the count matrix fixes the sample order.
    si = si.loc[sample_ids]
    controls = [
        (c, t, [str(v) for v in si[c]]) for c, t in _control_specs(params.get("control_cols"))
    ]
    enc_l = comparisons["encoded_left"] if "encoded_left" in comparisons else comparisons["left"]
    enc_r = comparisons["encoded_right"] if "encoded_right" in comparisons else comparisons["right"]
    cmps = [
        (str(a), str(b), str(c), str(d))
        for a, b, c, d in zip(comparisons["left"], comparisons["right"], enc_l, enc_r)
    ]
    anova = params.get("mode") == "anova"
    alpha = params.get("deseq2_alpha")
    shrink = params.get("deseq2_lfc_shrinkage")
    gene_ids = [str(g) for g in counts.index]
    res = _core.deseq2_pipeline(
        np.ascontiguousarray(counts.to_numpy(dtype=np.float64)),
        gene_ids,
        sample_ids,
        cc,
        [str(v) for v in si[cc]],
        controls,
        cmps,
        anova=anova,
        alpha=0.05 if alpha is None else float(alpha),
        shrink="none" if shrink is None else str(shrink),
        entity_type=params.get("entity_type", "gene"),
        diagnostics=diagnostics,
    )

    # merge(allDT, stats, by = "GroupId"): rows ordered by the character key (C collation).
    order = sorted(range(len(gene_ids)), key=lambda i: gene_ids[i].encode())
    if anova:
        labels = res["anova_labels"]
        table = pd.DataFrame(
            {
                "GroupId": gene_ids,
                "AveExpr": res["ave_expr"],
                "PValue": res["PValue"],
                "AdjPValue": res["AdjPValue"],
                "LRT": res["LRT"],
                "MaxLog2FCPair": ["" if k is None else labels[k] for k in res["max_pair"]],
                "MaxLog2FC": res["max_log2fc"],
            }
        )
        table = table.iloc[order].reset_index(drop=True)
        for c in ["AveExpr", "PValue", "AdjPValue", "LRT", "MaxLog2FC"]:
            table[c] = [_r_character(v) for v in table[c]]
        table["GroupId"] = table["GroupId"].astype(str)
    else:
        out = {"GroupId": gene_ids}
        for p in res["pairs"]:
            for s in PAIR_STATS:
                out[f"{s} {p['label']}"] = p[s]
        out["AveExpr"] = res["ave_expr"]
        table = pd.DataFrame(out).iloc[order].reset_index(drop=True)
        # type_convert(out, "integer", "GroupId"), when every id is an integer.
        if all(g.lstrip("-").isdigit() for g in table["GroupId"]):
            table["GroupId"] = table["GroupId"].astype(np.int64)
    if diagnostics:
        return table, _diag(res, gene_ids, sample_ids)
    return table


def _diag(res: dict, gene_ids: list[str], sample_ids: list[str]) -> dict:
    """The fit's intermediates over the genes filterByExpr kept (input order).

    ``samples``: replicate, size_factor. ``genes``: id, baseMean, baseVar, dispGeneEst, dispFit,
    dispersion, maxCooks, then the unshrunk coefficients (log2) and their SEs as ``<coef>`` and
    ``SE_<coef>`` with DESeq2's ``resultsNames``. ``scalars``: coef_names.
    """
    d = res["diag"]
    ids = [gene_ids[i] for i in d["kept_idx"]]
    if all(g.lstrip("-").isdigit() for g in ids):
        ids = [int(g) for g in ids]
    samples = pd.DataFrame({"replicate": sample_ids, "size_factor": d["size_factors"]})
    genes = pd.DataFrame(
        {
            "id": ids,
            "baseMean": d["base_mean"],
            "baseVar": d["base_var"],
            "dispGeneEst": d["disp_gene_est"],
            "dispFit": d["disp_fit"],
            "dispersion": d["dispersion"],
            "maxCooks": d["max_cooks"],
        }
    )
    for k, c in enumerate(d["coef_names"]):
        genes[c] = d["beta"][:, k]
    for k, c in enumerate(d["coef_names"]):
        genes[f"SE_{c}"] = d["se"][:, k]
    return {"samples": samples, "genes": genes, "scalars": {"coef_names": list(d["coef_names"])}}
