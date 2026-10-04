"""Copy the small count-corpus tier into tests/corpus-small/ for CI.

The full corpus (~780 MB) lives outside git and stays the local default. This copies whole runs,
never trimmed: dispersion and the shrinkage priors pool across every gene, so a trimmed input
would no longer match the R results. Only the files the tests read are copied: reference/<run>/
without .rds, runs/<run>/manifest.json, and the shrink_* files plus reference.json of
reference-shrink/<run> (its deseq2_* and input_* files repeat reference/<run>). The tests' run
counts for this tier are in crates/*/tests/common/mod.rs and tests/test_deseq2_golden.py.

Usage: uv run python scripts/build_small_corpus.py ~/wd/md-count-golden-corpus
"""

import json
import shutil
import sys
from pathlib import Path

FULL = Path(sys.argv[1]).expanduser()
SMALL = Path(__file__).resolve().parent.parent / "tests" / "corpus-small"

RUNS = [
    # m - p <= 3 prior variance simulation (no count_synth run has it), Wald, factor control;
    # also the input of the Python diagnostics and input-error tests
    "count_deseq2_airway_all_ctlfactor",
    # custom comparisons with relevel refits, factor + numeric controls
    "count_deseq2_count_synth_custom_ctlfactor_numeric",
    # LRT (ANOVA) with factor + numeric controls
    "count_deseq2_count_synth_anova_ctlfactor_numeric",
    # alpha 0.1 on the ANOVA omnibus and contrast tables
    "count_deseq2_anova_alpha0.1",
    # Cook's: replacement and refit, outliers flagged in the results
    "edge_deseq2_cooks",
    # expected errors
    "edge_deseq2_filter_drops_all",
    "edge_deseq2_non_integer",
    "edge_deseq2_one_rep_per_condition",
    "edge_deseq2_rank_deficient",
    # lfcShrink end to end: normal, apeglm, ashr
    "count_deseq2_count_synth_shrink_normal",
    "count_deseq2_count_synth_shrink_apeglm",
    "count_deseq2_count_synth_shrink_ashr",
]

# apeglm and ashr intermediates and the mix-SQP traces
SHRINK_RUNS = [
    "count_deseq2_count_synth_shrink_apeglm",
    "count_deseq2_count_synth_shrink_ashr",
]


def copy_files(src: Path, dst: Path, keep) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.is_file() and keep(f.name):
            shutil.copy2(f, dst / f.name)


if SMALL.exists():
    shutil.rmtree(SMALL)
for run in RUNS:
    copy_files(FULL / "reference" / run, SMALL / "reference" / run, lambda n: not n.endswith(".rds"))
    copy_files(FULL / "runs" / run, SMALL / "runs" / run, lambda n: n == "manifest.json")
for run in SHRINK_RUNS:
    copy_files(
        FULL / "reference-shrink" / run,
        SMALL / "reference-shrink" / run,
        lambda n: n.startswith("shrink_") or n == "reference.json",
    )
index = json.loads((FULL / "reference-shrink" / "index.json").read_text())
index["runs"] = [r for r in index["runs"] if r["run_id"] in SHRINK_RUNS]
index["n_runs"] = len(index["runs"])
(SMALL / "reference-shrink" / "index.json").write_text(json.dumps(index, indent=2) + "\n")
print(f"{len(RUNS)} runs, {len(SHRINK_RUNS)} shrink runs into {SMALL}")
