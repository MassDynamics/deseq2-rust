# count-reference

A standalone R reference for the gene entity's edgeR and DESeq2 paths in MDFlexiComparisons
(PR-E2 of the edgeR + DESeq2 Rust port). It recomputes each engine stage by stage and writes
every intermediate as its own golden. The Rust cores (PR-E3 onward) are checked against these
intermediates, not just the final table.

```
sh count-reference/gen_reference.sh                 # every count_* / edge_* run
sh count-reference/gen_reference.sh <run_id> ...    # selected runs
```

The script runs inside the `md-flexi-r45-local` image (R 4.5.0, BioC 3.22, edgeR 4.8.2,
DESeq2 1.50.2). It reads `$MD_COUNT_CORPUS_DIR/runs/<run_id>/` and writes
`$MD_COUNT_CORPUS_DIR/reference/<run_id>/`, with one CSV per intermediate plus `reference.json`.

## Checks

`reference.json` carries two verdicts.

- **`self_check`.** The reference's final table is compared with the end-to-end corpus output
  (`results.rds`). Continuous columns must agree to rel 1e-8, and the NA pattern must match
  exactly. Expected-error runs must fail with the recorded message.
- **`internal_checks`.** Each closed-form step recomputed in plain R is compared with the
  package value. For DESeq2 this also includes every mcol and assay of the stage-by-stage
  rebuild against `DESeq()` itself, and the independent-filtering threshold, theta, numRej and
  lowess fit against `metadata(results())`.

The self-check also fails when either table has a column the other lacks, except the columns
production adds outside the engine (`GeneNames`, `Description`, `GroupLabel`, `GroupLabelType`,
`NImputed: *`, `NReplicates: *`), which the router owns in the port (PR-E7).

Every `reference.json` carries a `provenance` block (md-limma commit, the hash of this directory
computed the way `gen_count.sh` hashes the harness, the corpus run's harness hash, the image
digest), and `reference/index.json` lists every run with the sha256 of each file it wrote, so a
Rust test can tell a stale golden from a current one.

The iterative fits are called, not rebuilt: DESeq2's C++ dispersion line search, the NB GLM
IRLS and the beta-prior refit. Their outputs are dumped. The rest is plain R. That covers size
factors, base means, the parametric trend, the dispersion prior variance, the outlier rule,
Wald and LRT p-values, the `m - p <= 3` simulated prior variance (`set.seed(2)`, histogram, KL
grid, loess, argmin, each dumped), Cook's distances, outlier replacement, the Cook's filter and its rescue,
independent filtering and BH. The edgeR reference rebuilds TMM, estimateDisp, glmQLFit and the
QL F test in full.

Status at PR-E2 round 2: 69 of 69 runs pass, and every internal check is bit-identical.

## Production behaviour reproduced as shipped

- The DESeq2 pairwise relevel refit runs `nbinomWaldTest` on the original counts with the
  post-replacement dispersions. This undoes Cook's replacement for any comparison whose right
  side is not the base level.
- The edgeR CI uses df = df.prior + df.residual, which does not match the QL F test's df.
- apeglm and ashr shrinkage call `lfcShrink` exactly as production does. Rust v1 does not port
  them (G1).
