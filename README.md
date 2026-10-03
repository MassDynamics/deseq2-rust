# deseq2-rust

DESeq2 differential expression (normal, apeglm and ashr LFC shrinkage) for the Mass Dynamics
platform, ported from MDFlexiComparisons' `runDiscovery` / `runANOVA` with `de_method = "DESeq2"`:
a Rust core with a Python entry point (`deseq2_rust.run`) and no R at run time.

## Build and test

```sh
uv sync
uv run pytest
cargo test --release --workspace
```

The golden tests read the count corpus at `~/wd/md-count-golden-corpus` (or
`MD_COUNT_CORPUS_DIR`) and fail without it.

## The edge-rust pin

The `rnum` and `edger-core` crates come from edge-rust through a git dependency in `Cargo.toml`,
pinned to a full commit SHA with a `file:///Users/...` URL. That URL resolves on the development
machine only; it is deliberate while both repos are local, and must become a hosted git URL (same
`rev`) before this repo is built anywhere else. To take an edge-rust change, commit it there, bump
the `rev` here and re-run the gate.

## Known limits

- **Ill-conditioned numeric controls.** Production refuses a design whose `qr.R` has a reciprocal
  condition number below machine epsilon, and the port refuses at the same point with the same
  message. Just below that boundary (rcond about 2e-16 to 4e-16, for example a dose column scaled
  to 3e14 to 5.24e14) both run, but the numbers differ: the fit there is noise-dominated and R's
  result depends on its exact LAPACK rounding. A tighter guard would refuse runs production makes,
  so the band is documented and its boundary pinned by a test.
- **Float-noise covariates.** A numeric control holding values that differ only in the last bits
  (for example `1.000000000000001` next to `1.0`) can move a gene's statistic by about 1e-7
  relative, because the cell grouping and the fit see those values exactly.
