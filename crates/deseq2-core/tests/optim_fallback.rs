//! The `fitNbinomGLMsOptim` fallback end to end: `DESeq(dds)` with `~ batch + cond` on a
//! constructed input (`tests/data/optim_fallback/make_case.R`, R 4.5.0 + DESeq2 1.50.2) where
//! six rows diverge in IRLS (|beta| > 30, so `betaIter = maxit`) and are refit by L-BFGS-B.

mod common;
use common::*;
use deseq2_core::design::{Design, Factor, Var};
use deseq2_core::fit;
use std::path::PathBuf;

const TOL: f64 = 1e-8;

fn data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/optim_fallback")
}

#[test]
fn optim_fallback_matches_r_deseq() {
    let dir = data_dir();
    let cd = Table::read(&dir.join("coldata.csv"));
    let m = cd.nrow;
    let design = Design {
        n: m,
        vars: vec![
            Var::Factor(Factor::new("batch", cd.str("batch"))),
            Var::Factor(Factor::new("cond", cd.str("cond"))),
        ],
    };
    let ct = Table::read(&dir.join("counts.csv"));
    let n = ct.nrow;
    let mut counts = vec![0.0; n * m];
    for (j, name) in ct.names.iter().enumerate() {
        for (g, v) in ct.f64(name).iter().enumerate() {
            counts[g * m + j] = *v;
        }
    }
    let f = fit::deseq(&counts, &design, false).unwrap();
    let t = &f.test;
    let want = Table::read(&dir.join("expected.csv"));
    let p = t.coef_names.len();
    assert_eq!(p, 5);

    // The rows R sent to optim are the ones the port sent to optim.
    let want_iter = want.f64("betaIter");
    assert_close("betaIter", &t.iter, &want_iter, 0.0);
    let optim_rows: Vec<usize> = (0..n).filter(|g| want_iter[*g] == 100.0).collect();
    assert_eq!(optim_rows, vec![0, 1, 2, 5, 6, 8]);
    assert_eq!(
        t.conv,
        want.bool("betaConv"),
        "betaConv (TRUE after optim converges)"
    );

    let coefs = [
        "Intercept",
        "batch_b2_vs_b1",
        "batch_b3_vs_b1",
        "cond_B_vs_A",
        "cond_C_vs_A",
    ];
    let mut worst = 0.0f64;
    let mut worst_optim = 0.0f64;
    for (k, cf) in coefs.iter().enumerate() {
        for (field, got_all) in [
            ("beta", &t.beta),
            ("SE", &t.se),
            ("stat", &t.stat),
            ("p", &t.pvalue),
        ] {
            let got: Vec<f64> = (0..n).map(|g| got_all[g * p + k]).collect();
            let w = want.f64(&format!("{field}_{cf}"));
            worst = worst.max(assert_close(&format!("{field}_{cf}"), &got, &w, TOL));
            for &g in &optim_rows {
                worst_optim = worst_optim.max(rel_diff(got[g], w[g]));
            }
        }
    }
    eprintln!(
        "optim fallback: {n} rows, {} refit by optim; max rel diff {worst:e} (optim rows {worst_optim:e})",
        optim_rows.len()
    );
}
