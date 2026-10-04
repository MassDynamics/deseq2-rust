"""deseq2_rust.run against every DESeq2 run in the count golden corpus.

The strict numeric gate is the Rust test ``crates/deseq2-core/tests/results.rs`` (every column
within 1.2e-12 of the reference); the numbers pass through the Python layer unchanged. This test
checks what the Python layer owns: row order, GroupId type, column names, NA pattern and the
ANOVA string shaping, with the numbers at 1e-8 relative.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import deseq2_rust
import numpy as np
import pandas as pd
import pytest

CORPUS = Path(os.environ.get("MD_COUNT_CORPUS_DIR", Path.home() / "wd/md-count-golden-corpus"))
SMALL_TIER = CORPUS.resolve() == (Path(__file__).parent / "corpus-small").resolve()
RUNS = sorted(
    p.name
    for p in (CORPUS / "reference").glob("*deseq2*")
    if p.name.startswith(("count_", "edge_"))
)
if not RUNS:
    # A missing corpus fails the suite; skipping it silently turned the gate green with no parity
    # check (review r1, D-11). Set DESEQ2_RUST_ALLOW_NO_CORPUS=1 to run the rest without it.
    if os.environ.get("DESEQ2_RUST_ALLOW_NO_CORPUS") == "1":
        pytest.skip(f"no DESeq2 runs under {CORPUS}", allow_module_level=True)
    pytest.fail(f"no DESeq2 runs under {CORPUS} (set MD_COUNT_CORPUS_DIR)", pytrace=False)

TOL = 1e-8


def manifest(run: str) -> dict:
    return json.loads((CORPUS / "runs" / run / "manifest.json").read_text())


def inputs(run: str):
    ref = CORPUS / "reference" / run
    counts = pd.read_csv(ref / "input_counts.csv", dtype={"id": str}).set_index("id")
    si = pd.read_csv(ref / "input_sample_info.csv", dtype=str)
    cmp = pd.read_csv(ref / "input_comparisons.csv", dtype=str)
    m = manifest(run)
    params = dict(m["params"], entity_type=m["entity_type"], mode=m["mode"])
    return counts, si, cmp, params


def airway_ctlnone_inputs():
    """count_deseq2_airway_all_ctlnone's inputs: the ctlfactor run's without the cell column (the
    counts and comparisons are identical), so the small tier needs one airway run."""
    counts, si, cmp, params = inputs("count_deseq2_airway_all_ctlfactor")
    return counts, si.drop(columns="cell"), cmp, dict(params, control_cols=None)


def num(s: pd.Series) -> np.ndarray:
    return pd.to_numeric(s.replace("", np.nan)).to_numpy(dtype=float)


def check_numeric(name: str, got: np.ndarray, want: np.ndarray):
    na_g, na_w = np.isnan(got), np.isnan(want)
    assert (na_g == na_w).all(), f"{name}: NA pattern differs at {np.flatnonzero(na_g != na_w)[:5]}"
    a, b = got[~na_w], want[~na_w]
    gap = np.abs(a - b)
    # data.table::fwrite prints subnormals wrongly; two subnormals compare equal.
    tiny = (np.abs(a) < np.finfo(float).tiny) & (np.abs(b) < np.finfo(float).tiny)
    ok = (gap <= TOL * np.abs(b)) | tiny
    assert ok.all(), f"{name}: worst rel {np.max(gap[~ok] / np.abs(b[~ok])):.2e}"


TABLE_RUNS = [
    r
    for r in RUNS
    if manifest(r)["status"] != "error"
    and (CORPUS / "reference" / r / "reference_output.csv").exists()
]


@pytest.mark.parametrize("run", TABLE_RUNS)
def test_table_matches_reference_output(run):
    counts, si, cmp, params = inputs(run)
    got = deseq2_rust.run(counts, si, cmp, params)
    anova = params["mode"] == "anova"
    want = pd.read_csv(
        CORPUS / "reference" / run / "reference_output.csv",
        dtype=str if anova else {"GroupId": np.int64},
        keep_default_na=not anova,
    )
    assert list(got.columns) == list(want.columns)
    assert len(got) == len(want)
    # The reference CSVs are in C-collation order. Pairwise rows follow the input (features
    # metadata) order; DESeq2 ANOVA rows are in numeric GroupId order.
    if anova:
        gid = got["GroupId"].astype(int).to_numpy()
        assert (np.diff(gid) > 0).all(), "rows not in numeric GroupId order"
        key = want["GroupId"].astype(int).to_numpy()
    else:
        assert [str(g) for g in got["GroupId"]] == [str(g) for g in counts.index]
        pos = {str(g): i for i, g in enumerate(counts.index)}
        key = [pos[str(g)] for g in want["GroupId"]]
    want = want.iloc[np.argsort(key, kind="stable")].reset_index(drop=True)
    if anova:
        want = want.fillna("")
        assert list(got["GroupId"]) == list(want["GroupId"])
        assert list(got["MaxLog2FCPair"]) == list(want["MaxLog2FCPair"])
        for c in ["AveExpr", "PValue", "AdjPValue", "LRT", "MaxLog2FC"]:
            assert got[c].map(type).eq(str).all(), f"{c}: not strings"
    else:
        assert got["GroupId"].dtype == np.int64
        assert (got["GroupId"].to_numpy() == want["GroupId"].to_numpy()).all()
    for c in want.columns:
        if c in ("GroupId", "MaxLog2FCPair"):
            continue
        check_numeric(f"{run} {c}", num(got[c].astype(object)), num(want[c].astype(object)))


ERROR_RUNS = [r for r in RUNS if manifest(r)["status"] == "error"]


def test_corpus_is_complete():
    """A shrunken corpus must not pass quietly (review r1, D-11)."""
    want = 8 if SMALL_TIER else 35  # scripts/build_small_corpus.py for the small tier
    assert len(TABLE_RUNS) == want, f"{len(TABLE_RUNS)} table runs, expected {want}"
    assert len(ERROR_RUNS) == 4, f"{len(ERROR_RUNS)} error runs, expected 4"


@pytest.mark.parametrize("run", ERROR_RUNS)
def test_expected_error(run):
    expected = manifest(run)["expected_error"]
    if run == "edge_deseq2_non_integer":
        # Fails in prepare_inputs before inputs are dumped: rebuild it from an ordinary run.
        counts, si, cmp, params = airway_ctlnone_inputs()
        counts = counts.astype(float)
        counts.iloc[0, 0] += 0.5
    else:
        counts, si, cmp, params = inputs(run)
    with pytest.raises(ValueError) as e:
        deseq2_rust.run(counts, si, cmp, params)
    assert expected in str(e.value)


def test_diagnostics_are_consistent_with_the_table():
    run = "count_deseq2_airway_all_ctlfactor"
    counts, si, cmp, params = inputs(run)
    plain = deseq2_rust.run(counts, si, cmp, params)
    table, diag = deseq2_rust.run(counts, si, cmp, params, diagnostics=True)
    pd.testing.assert_frame_equal(table, plain)
    genes = diag["genes"]
    kept = table.loc[table["AveExpr"].notna(), "GroupId"].astype(str)
    assert sorted(genes["id"].astype(str)) == sorted(kept)
    ave = table.set_index(table["GroupId"].astype(str))["AveExpr"]
    np.testing.assert_array_equal(
        genes["baseMean"].to_numpy(), ave[genes["id"].astype(str)].to_numpy()
    )
    assert len(diag["samples"]) == counts.shape[1]


@pytest.mark.parametrize(
    "gene2, expected",
    [
        # All-zero sample: edgeR::filterByExpr stops before DESeq2 runs.
        (0, "library sizes should be greater than zero"),
        # Sample's only counts sit in a gene filterByExpr drops: DESeq2's size factors stop.
        (5, "every gene contains at least one zero, cannot compute log geometric means"),
    ],
)
def test_empty_sample_errors_like_production(gene2, expected):
    # Messages from runDiscovery(de_method = "DESeq2") in md-flexi-r45-local on these inputs.
    counts, si, cmp, params = airway_ctlnone_inputs()
    s1 = counts.columns[0]
    counts[s1] = 0
    counts.loc["2", s1] = gene2
    with pytest.raises(ValueError, match=expected):
        deseq2_rust.run(counts, si, cmp, params)


def test_invalid_shrinkage_is_refused():
    counts, si, cmp, params = airway_ctlnone_inputs()
    with pytest.raises(ValueError, match="Invalid deseq2_lfc_shrinkage value: 'bogus'"):
        deseq2_rust.run(counts, si, cmp, dict(params, deseq2_lfc_shrinkage="bogus"))
