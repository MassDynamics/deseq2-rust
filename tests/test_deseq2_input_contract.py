"""deseq2_rust.run against production's input handling (review r1, D-2 to D-10).

Each test feeds ``run`` one input that production refuses or repairs, or one invariance it holds.
Every expected message was taken from production R in md-flexi-r45-local (MDFlexiComparisons
``runDiscovery`` / ``runANOVA`` with ``de_method = "DESeq2"``, DESeq2 1.50.2) on the same kind of
input. Corpus-free.
"""

from __future__ import annotations

import deseq2_rust
import numpy as np
import pandas as pd
import pytest
from deseq2_rust import _core


def _base(ng=400, reps=3, seed=1):
    rng = np.random.default_rng(seed)
    samples = [f"S{i:02d}" for i in range(2 * reps)]
    cond = ["A"] * reps + ["B"] * reps
    mu = rng.gamma(1.0, 200.0, size=ng)
    fc = np.where(rng.random(ng) < 0.1, 4.0, 1.0)
    counts = np.empty((ng, len(samples)))
    for j, c in enumerate(cond):
        m = mu * (fc if c == "B" else 1.0)
        counts[:, j] = rng.negative_binomial(5, 5 / (5 + m))
    counts = pd.DataFrame(counts, index=[str(1000 + g) for g in range(ng)], columns=samples)
    si = pd.DataFrame(
        {"replicate": samples, "condition": cond, "batch": (["x", "y", "z"] * 2 * reps)[: 2 * reps]}
    )
    cmp = pd.DataFrame({"left": ["B"], "right": ["A"]})
    return counts, si, cmp, {"condition_col": "condition", "entity_type": "gene"}


def _three_groups():
    c, si, _, p = _base()
    si["condition"] = ["A", "A", "B", "B", "C", "C"]
    return c, si, p


# --- refusals production makes ---------------------------------------------------------------


def test_missing_condition_is_refused():
    """D-3: production stops instead of fitting a level called "nan"."""
    c, si, cmp, p = _base()
    si.loc[0, "condition"] = np.nan
    with pytest.raises(ValueError, match="Condition column 'condition' contains missing values"):
        deseq2_rust.run(c, si, cmp, p)


@pytest.mark.parametrize(
    "col,typ,val", [("batch", "categorical", None), ("dose", "numerical", np.nan)]
)
def test_missing_control_is_refused(col, typ, val):
    """D-3: DESeqDataSetFromMatrix refuses NA in a design variable."""
    c, si, cmp, p = _base()
    si["dose"] = [1.0, 2.0, 1.3, 1.5, 2.5, 3.0]
    si[col] = si[col].astype(object)
    si.loc[0, col] = val
    p["control_cols"] = {"Column": col, "Type": typ}
    with pytest.raises(ValueError, match=f"variables in design formula cannot contain NA: {col}"):
        deseq2_rust.run(c, si, cmp, p)


def test_empty_string_control_is_a_level():
    """Production DESeq2 accepts "" as a control level (production edgeR does not)."""
    c, si, cmp, p = _base()
    si["batch"] = ["", "y", "z", "", "y", "z"]
    p["control_cols"] = {"Column": "batch", "Type": "categorical"}
    t = deseq2_rust.run(c, si, cmp, p)
    assert t["PValue B - A"].notna().any()


@pytest.mark.parametrize("values", [["x"] * 6, [None, "x", "x", "x", "x", "x"]])
def test_single_level_control_is_refused(values):
    """D-6: model.matrix refuses a one-level factor, before the NA check sees a missing value."""
    c, si, cmp, p = _base()
    si["batch"] = values
    p["control_cols"] = {"Column": "batch", "Type": "categorical"}
    with pytest.raises(ValueError, match="contrasts can be applied only to factors with 2 or more"):
        deseq2_rust.run(c, si, cmp, p)


def test_count_above_int32_is_refused():
    """D-7: DESeqDataSetFromMatrix stores integers; 3e9 becomes NA and DESeq2 stops."""
    c, si, cmp, p = _base()
    c.iloc[0, :] = 3_000_000_000.0
    with pytest.raises(ValueError, match="NA counts not allowed"):
        deseq2_rust.run(c, si, cmp, p)


def test_count_at_int32_max_is_accepted():
    c, si, cmp, p = _base()
    c.iloc[0, :] = 2_147_483_647.0
    deseq2_rust.run(c, si, cmp, p)


@pytest.mark.parametrize("mode", ["discovery", "anova"])
@pytest.mark.parametrize("alpha", [0.0, 1.0, 1.5, float("nan")])
def test_alpha_outside_unit_interval_is_refused(alpha, mode):
    """D-10: results() checks stopifnot(alpha > 0 & alpha < 1)."""
    c, si, cmp, p = _base()
    with pytest.raises(ValueError, match=r"alpha > 0 & alpha < 1 is not TRUE"):
        deseq2_rust.run(c, si, cmp, dict(p, deseq2_alpha=alpha, mode=mode))


def test_pairwise_level_against_itself_is_refused():
    """D-5: checkContrast stops on the dispatch contrast."""
    c, si, _, p = _base()
    cmp = pd.DataFrame({"left": ["B"], "right": ["B"]})
    with pytest.raises(ValueError, match="B and B should be different level names"):
        deseq2_rust.run(c, si, cmp, p)


@pytest.mark.parametrize("level", ["A", "B"])
def test_anova_level_against_itself_is_refused(level):
    """D-5: an ANOVA "B - B" used to return B vs A under the B - B label."""
    c, si, p = _three_groups()
    cmp = pd.DataFrame({"left": ["C", level], "right": ["A", level]})
    with pytest.raises(ValueError, match=f"{level} and {level} should be different level names"):
        deseq2_rust.run(c, si, cmp, dict(p, mode="anova"))


@pytest.mark.parametrize("scale,rcond", [(1e15, "1.16364e-16"), (1e20, "1.16364e-21")])
def test_near_singular_numeric_control_is_refused(scale, rcond):
    """D-8: solve(qr.R) in linearModelMu stops below machine eps. Probe data and rcond from R."""
    c, si, cmp, p = _base(ng=300)
    si["dose"] = np.array([1, 2, 1.3, 1.5, 2.5, 3]) * scale
    p["control_cols"] = {"Column": "dose", "Type": "numerical"}
    with pytest.raises(
        ValueError, match=f"computationally singular: reciprocal condition number = {rcond}$"
    ):
        deseq2_rust.run(c, si, cmp, p)


def test_large_numeric_control_below_the_threshold_runs():
    c, si, cmp, p = _base(ng=300)
    si["dose"] = np.array([1, 2, 1.3, 1.5, 2.5, 3]) * 1e12
    p["control_cols"] = {"Column": "dose", "Type": "numerical"}
    deseq2_rust.run(c, si, cmp, p)


def test_zero_library_sample_message():
    c, si, cmp, p = _base()
    c.iloc[:, 0] = 0.0
    with pytest.raises(ValueError, match="library sizes should be greater than zero"):
        deseq2_rust.run(c, si, cmp, p)


def test_infinite_count_is_refused():
    """Production stops on Inf; only NA is filled (D-4)."""
    c, si, cmp, p = _base()
    c.iloc[5, 2] = np.inf
    with pytest.raises(ValueError, match="1 non-finite"):
        deseq2_rust.run(c, si, cmp, p)


@pytest.mark.parametrize("axis", ["genes", "samples"])
def test_duplicate_ids_are_refused(axis):
    """D-10."""
    c, si, cmp, p = _base()
    if axis == "genes":
        c.index = list(c.index[:-1]) + [c.index[0]]
    else:
        c = pd.concat([c, c[["S00"]]], axis=1)
    with pytest.raises(ValueError, match="duplicate"):
        deseq2_rust.run(c, si, cmp, p)


def test_internal_panic_is_a_runtime_error():
    """D-10: a Rust panic reaches Python as RuntimeError, not pyo3's BaseException."""
    with pytest.raises(RuntimeError, match="internal error in the DESeq2 engine: selftest panic"):
        _core._selftest_panic()


# --- repairs production makes ----------------------------------------------------------------


def test_missing_count_cell_is_zero_filled():
    """D-4: production fills plain NA with 0 (.buildCountMatrixFromLongDT); a missing cell after
    a pandas pivot is NaN. Same decision as edge_rust."""
    c, si, cmp, p = _base()
    filled = c.copy()
    filled.iloc[5, 2] = 0.0
    c.iloc[5, 2] = np.nan
    pd.testing.assert_frame_equal(
        deseq2_rust.run(c, si, cmp, p), deseq2_rust.run(filled, si, cmp, p)
    )


# --- invariances -----------------------------------------------------------------------------


def test_sample_column_order_does_not_change_results():
    """D-2: dcast sorts the run columns, so production never sees the caller's column order."""
    c, si, cmp, p = _base()
    a = deseq2_rust.run(c, si, cmp, p)
    b = deseq2_rust.run(c.iloc[:, [3, 0, 4, 1, 5, 2]], si, cmp, p)
    pd.testing.assert_frame_equal(a, b, check_exact=True)


def test_callers_sample_info_is_not_mutated():
    """D-10."""
    c, si, cmp, p = _base()
    si = si.drop(columns="replicate")
    c.columns = range(6)
    before = si.copy(deep=True)
    deseq2_rust.run(c, si, cmp, p)
    pd.testing.assert_frame_equal(si, before)


def test_non_ascii_digit_ids_do_not_become_integers():
    """D-10: "١٠٠٠" passes str.isdigit() and became 1000, colliding with the real gene."""
    c, si, cmp, p = _base()
    c.index = [c.index[0], "١٠٠٠"] + list(c.index[2:])
    t = deseq2_rust.run(c, si, cmp, p)
    assert not pd.api.types.is_integer_dtype(t["GroupId"])
    assert t["GroupId"].tolist().count("1000") == 1


@pytest.mark.parametrize("mode", ["discovery", "anova"])
def test_rows_are_in_numeric_group_id_order(mode):
    """Production's final table is in numeric GroupId order: 999 before 1000."""
    c, si, cmp, p = _base()
    c.index = [str(g) for g in range(990, 990 + len(c))]
    t = deseq2_rust.run(c, si, cmp, dict(p, mode=mode))
    ids = [int(g) for g in t["GroupId"]]
    assert ids == sorted(ids)
