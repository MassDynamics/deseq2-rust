# deseq2-rust

DESeq2 differential expression (normal, apeglm and ashr LFC shrinkage) for the Mass Dynamics
platform, ported from MDFlexiComparisons' `runDiscovery` / `runANOVA` with `de_method = "DESeq2"`:
a Rust core with a Python entry point (`deseq2_rust.run`) and no R at run time.

## Build and test

```sh
uv sync
uv run pytest
cargo test --release --workspace
cargo test --workspace   # debug: overflow checks and debug_assert! only run here
```

The golden tests read the count corpus at `~/wd/md-count-golden-corpus` (or
`MD_COUNT_CORPUS_DIR`) and fail without it. CI runs them on a small tier of whole runs committed
in `tests/corpus-small` (`MD_COUNT_CORPUS_DIR=$PWD/tests/corpus-small`), built by
`scripts/build_small_corpus.py`. The R references come from the production image
`md-flexi-r45-local:latest` run as `linux/amd64` under emulation on an Apple silicon Mac, and are
assumed, not checked, to match native x86-64 production.

## The edge-rust pin

The `rnum` and `edger-core` crates come from edge-rust through a git dependency in `Cargo.toml`,
pinned to a full commit SHA with a `file:///Users/...` URL. That URL resolves on the development
machine only; it is deliberate while both repos are local, and must become a hosted git URL (same
`rev`) before this repo is built anywhere else. To take an edge-rust change, commit it there, bump
the `rev` here and re-run the gate.

## Known limits

- **Ill-conditioned numeric controls.** Production refuses a design whose `qr.R` has a reciprocal
  condition number below machine epsilon, and the port refuses at the same point with the same
  message. Just above that boundary (for example a dose column scaled to 3e14 to 5.24e14) both run
  and match, since the port takes Armadillo's approximate solve where production does (review
  deseq2 r6). The exception: when the IRLS step's rcond sits within about 5% of eps on every
  iteration, the port's last-bit rounding can make R and the port stop on different iterations, so
  one returns a table and the other refuses. No probe has hit it.
- **Near-singular steps above 25 coefficients.** Production approximates such an IRLS step and
  carries on; the port refuses the run, because only the small `dgelsd` branch is ported.
- **Band dispatch at 32 or more coefficients.** Armadillo's `solve()` checks for a band matrix
  before it checks for a triangular one once the system has 32 or more columns, so an IRLS `R`
  factor with exact zeros at the top of its last columns would take Armadillo's band solver in
  production. The port always takes the triangular path. A QR `R` factor of a real design is
  unlikely to have that structure, and designs this wide are not expected in production.
- **Float-noise covariates.** A numeric control holding values that differ only in the last bits
  (for example `1.000000000000001` next to `1.0`) can move a gene's statistic by about 1e-7
  relative, because the cell grouping and the fit see those values exactly.
