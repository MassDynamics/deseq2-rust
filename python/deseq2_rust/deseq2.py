"""The production DESeq2 table around the Rust engine.

Mirrors what MDFlexiComparisons does after the engine call: the left join of the engine table to
every gene id, the integer GroupId, the row order, and for ANOVA runs
``.packageANOVAOutput`` (``R/runANOVA.R``): the omnibus columns plus ``MaxLog2FCPair`` /
``MaxLog2FC``, every column as a string with NA written as "".

Row order (review deseq2 r2, m-2, checked in the image): production's pairwise table is
``featuresMetadata %>% left_join(stats)``, so rows follow the features metadata, which the caller
passes as the counts row order. DESeq2 ANOVA merges on the integer GroupId and returns numeric
order; with a non-integer id (which production cannot run) the port falls back to bytewise order.
"""

from __future__ import annotations

import logging

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

log = logging.getLogger(__name__)


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


# _is_int_id and _group_id_order have a twin in edge-rust's edger.py; change both together.
def _is_int_id(g: str) -> bool:
    """Whether ``type_convert`` would read the GroupId as an integer."""
    return g.isascii() and g.removeprefix("-").isdigit()


def _group_id_order(ids: list[str]) -> list[int]:
    """Positions of ``ids`` in GroupId order: numeric when every id is an integer, else bytewise."""
    if all(_is_int_id(g) for g in ids):
        return sorted(range(len(ids)), key=lambda i: int(ids[i]))
    return sorted(range(len(ids)), key=lambda i: ids[i].encode())


def _r_character(x) -> list[str]:
    """R's ``as.character`` of each double (``1e5`` is ``"1e+05"``), NA as ""."""
    x = np.asarray(x, dtype=np.float64)
    out = _core.r_as_character(x.tolist())
    return ["" if np.isnan(v) else s for v, s in zip(x, out)]


def run(
    counts: pd.DataFrame,
    sample_info: pd.DataFrame,
    comparisons: pd.DataFrame,
    params: dict,
    diagnostics: bool = False,
):
    """Run the DESeq2 engine and return the production output table.

    counts: genes x samples, index = gene ids, columns = sample ids; non-negative integers. The
        row order is the features metadata order; pairwise output rows follow it.
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
    si = si.set_axis(si.index.astype(str))  # a copy: the caller's frame is left alone
    # On the string ids: 1001 and "1001" are the same GroupId.
    if pd.Index([str(g) for g in counts.index]).duplicated().any():
        raise ValueError("counts has duplicate gene ids")
    if pd.Index([str(s) for s in counts.columns]).duplicated().any():
        raise ValueError("counts has duplicate sample ids")
    # dcast orders the sample columns by id (C collation); the fit depends on that order.
    counts = counts[sorted(counts.columns, key=lambda s: str(s).encode())]
    sample_ids = [str(s) for s in counts.columns]
    missing = set(sample_ids) - set(si.index)
    if missing:
        raise ValueError(f"samples missing from sample_info: {sorted(missing)}")
    # sampleInfo[colnames(countMatrix), ]: the count matrix fixes the sample order.
    si = si.loc[sample_ids]
    if si[cc].isna().any():
        raise ValueError(
            f"Condition column '{cc}' contains missing values. "
            "Fix the sample metadata before running DE."
        )
    specs = _control_specs(params.get("control_cols"))
    # model.matrix refuses a single-level factor before DESeqDataSetFromMatrix sees the NA.
    for c, t in specs:
        if t == "categorical" and si[c].dropna().astype(str).nunique() < 2:
            raise ValueError("contrasts can be applied only to factors with 2 or more levels")
    na_cols = [c for c, _ in specs if si[c].isna().any()]
    if na_cols:
        raise ValueError("variables in design formula cannot contain NA: " + ", ".join(na_cols))
    controls = [(c, t, [str(v) for v in si[c]]) for c, t in specs]
    mat = counts.to_numpy(dtype=np.float64, na_value=np.nan, copy=True)
    na = np.isnan(mat)
    if na.any():
        # Production fills plain NA with 0 and stops on NaN and Inf (edgeRStatsFun.R:56-72).
        # pandas cannot tell NaN from NA, and a cell missing after a pivot arrives as NaN, so
        # every NaN is filled with 0 here: parity holds for NA only, and a literal NaN count
        # runs where production stops. Inf still stops in the engine.
        log.info("DESeq2: coercing %d NA cell(s) in the count matrix to 0", int(na.sum()))
        mat[na] = 0.0
    enc_l = comparisons["encoded_left"] if "encoded_left" in comparisons else comparisons["left"]
    enc_r = comparisons["encoded_right"] if "encoded_right" in comparisons else comparisons["right"]
    cmps = [
        (str(a), str(b), str(c), str(d))
        for a, b, c, d in zip(comparisons["left"], comparisons["right"], enc_l, enc_r)
    ]
    anova = params.get("mode") == "anova"
    alpha = params.get("deseq2_alpha")
    shrink = params.get("deseq2_lfc_shrinkage")
    # dcast orders the rows by GroupId and production fits in that order whatever the metadata
    # order; the fit depends on it at about 1e-6 (review deseq2 r3, R3-m2).
    input_ids = [str(g) for g in counts.index]
    fit_order = _group_id_order(input_ids)
    gene_ids = [input_ids[i] for i in fit_order]
    res = _core.deseq2_pipeline(
        np.ascontiguousarray(mat[fit_order]),
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

    if anova:
        # runDESeq2ANOVAImpl merges on the integer GroupId: numeric order, the fit's order.
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
        for c in ["AveExpr", "PValue", "AdjPValue", "LRT", "MaxLog2FC"]:
            table[c] = _r_character(table[c])
        table["GroupId"] = table["GroupId"].astype(str)
    else:
        out = {"GroupId": gene_ids}
        for p in res["pairs"]:
            for s in PAIR_STATS:
                out[f"{s} {p['label']}"] = p[s]
        out["AveExpr"] = res["ave_expr"]
        # left_join onto the features metadata: the input order.
        table = pd.DataFrame(out).iloc[np.argsort(fit_order)].reset_index(drop=True)
        # type_convert(out, "integer", "GroupId"), when every id is an integer.
        if all(_is_int_id(g) for g in table["GroupId"]):
            table["GroupId"] = table["GroupId"].astype(np.int64)
    if diagnostics:
        return table, _diag(res, gene_ids, sample_ids, fit_order)
    return table


def _diag(res: dict, gene_ids: list[str], sample_ids: list[str], fit_order: list[int]) -> dict:
    """The fit's intermediates over the genes filterByExpr kept (input order).

    ``samples``: replicate, size_factor. ``genes``: id, baseMean, baseVar, dispGeneEst, dispFit,
    dispersion, maxCooks, then the unshrunk coefficients (log2) and their SEs as ``<coef>`` and
    ``SE_<coef>`` with DESeq2's ``resultsNames``. ``scalars``: coef_names, and
    shrink_nonconverged (per comparison, the apeglm rows whose MAP fit did not converge; None
    for other shrinkage types).
    """
    d = res["diag"]
    ids = [gene_ids[i] for i in d["kept_idx"]]
    if all(_is_int_id(g) for g in ids):
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
    # The fit runs in GroupId order; report the kept genes in input order.
    genes = genes.iloc[np.argsort([fit_order[i] for i in d["kept_idx"]])].reset_index(drop=True)
    scalars = {
        "coef_names": list(d["coef_names"]),
        "shrink_nonconverged": list(d["shrink_nonconverged"]),
    }
    return {"samples": samples, "genes": genes, "scalars": scalars}
