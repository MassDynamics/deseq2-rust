# Golden-corpus harness (copy)

These scripts generate the golden corpus. They run from the MDFlexiComparisons repo root, where
they live untracked under `data-raw/golden-corpus/` because they source that repo's `R/` directly.
This directory is a byte-for-byte copy of them, so the corpus can be reproduced from committed
code. Every `manifest.json` records the harness hash under `provenance.harness_sha256`, and
`tests/golden/test_harness_copy.py` fails if this copy drifts from it:

    cd corpus-harness && shasum -a 256 *.R *.sh | shasum -a 256

To regenerate: copy these files into `MDFlexiComparisons/data-raw/golden-corpus/` at the commit a
manifest names (`provenance.md_flexi_comparisons_sha`), run `fetch_public.sh` and then
`build_public_datasets.R` in the image for the two public datasets, then `regen_all.sh` (every
run plus the CSV export) or `gen.sh <run id> ...`. The R image is `md-flexi-r45-limma368`: the
MDFlexiComparisons R 4.5.0 image plus limma 3.68.5 installed from git.bioconductor.org
RELEASE_3_23 (commit 825d1c8). `scalar_goldens.R` writes `scalar/` and `matrix/`; `run_matrix.R` declares
every run.

The same harness also writes the edgeR / DESeq2 count corpus (phase 2, dev-aios
`projects/2026-10-01-edger-deseq2-rust-port/plan.md`): `run_matrix.R` declares it as `count_runs`,
`gen_count.sh` runs it with `MD_CORPUS_KIND=count` into `~/wd/md-count-golden-corpus/` under
`MD_R_IMAGE` (default `md-flexi-r45-local`, the MDFlexiComparisons image), and
`audit_count_corpus.R` checks it. Its manifests carry their own `tolerance_policy` and
`provenance.versions_status`. The versions are `confirmed` (plan gate G0): production pins
nothing, so they are the newest BioC 3.22 / CRAN set, which `md-flexi-r45-local` matched on
2026-10-01. A newer patch release means rebuilding the image and regenerating.
