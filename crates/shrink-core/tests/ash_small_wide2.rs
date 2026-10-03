//! ashr on 40 wide-spread effects with one outlier (review r1, stats item 7). The QP route
//! crosses the 25-component Jacobi branch here, where Rust drifts from R by about 5e-13; this
//! pins the result at 1e-10 so an active-set flip (a large discrete jump) cannot pass quietly.

use std::fs;

fn nums(file: &str) -> Vec<Vec<f64>> {
    let path = format!(
        "{}/tests/data/ash_small_wide2/{file}",
        env!("CARGO_MANIFEST_DIR")
    );
    fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("{path}: {e}"))
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.split_whitespace().map(|t| t.parse().unwrap()).collect())
        .collect()
}

fn close(what: &str, got: f64, want: f64) {
    assert!(
        (got - want).abs() <= 1e-10 * want.abs().max(1.0),
        "{what}: {got:e} vs R {want:e}"
    );
}

#[test]
fn ash_small_wide2_matches_r() {
    let input = nums("ashr_in.txt");
    let x: Vec<f64> = input.iter().map(|r| r[0]).collect();
    let s: Vec<f64> = input.iter().map(|r| r[1]).collect();
    let fit = shrink_core::shrink_ashr(&x, &s).unwrap();
    let want = nums("ashr_R.txt");
    assert_eq!(want.len(), 40);
    let t = &fit.table;
    for (i, w) in want.iter().enumerate() {
        close(&format!("PosteriorMean[{i}]"), t.posterior_mean[i], w[0]);
        close(&format!("PosteriorSD[{i}]"), t.posterior_sd[i], w[1]);
        close(&format!("NegativeProb[{i}]"), t.negative_prob[i], w[2]);
        close(&format!("lfsr[{i}]"), t.lfsr[i], w[3]);
        close(&format!("svalue[{i}]"), t.svalue[i], w[4]);
    }
    let pi: Vec<f64> = nums("ashr_R_pi.txt").into_iter().map(|r| r[0]).collect();
    assert_eq!(fit.pi.len(), pi.len());
    for (k, (g, w)) in fit.pi.iter().zip(&pi).enumerate() {
        close(&format!("pi[{k}]"), *g, *w);
    }
}
