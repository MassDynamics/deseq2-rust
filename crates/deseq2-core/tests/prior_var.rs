//! Step 1 gate: the m - p <= 3 prior variance simulation (R RNG, hist, loess) reproduces
//! every `deseq2_prior_var_*` golden on the six airway ctlfactor runs (the first only, on the
//! small tier). Residuals come from the `deseq2_disp.csv` stage golden so this test isolates the
//! simulation.

mod common;
use common::*;

const RUNS: [&str; 6] = [
    "count_deseq2_airway_all_ctlfactor",
    "count_deseq2_airway_all_ctlfactor_numeric",
    "count_deseq2_airway_anova_ctlfactor",
    "count_deseq2_airway_anova_ctlfactor_numeric",
    "count_deseq2_airway_custom_ctlfactor",
    "count_deseq2_airway_custom_ctlfactor_numeric",
];

fn check_run(run: &str, df: f64) {
    let dir = reference_dir(run);
    let disp = Table::read(&dir.join("deseq2_disp.csv"));
    let all_zero = disp.bool("allZero");
    let ge = disp.f64("dispGeneEst");
    let fit = disp.f64("dispFit");
    let mut resid = Vec::new();
    for i in 0..disp.nrow {
        if all_zero[i] == Some(false) && ge[i] >= 1e-8 * 100.0 {
            resid.push(ge[i].ln() - fit[i].ln());
        }
    }
    let sim = deseq2_core::prior_var::prior_var_simulation(&resid, df).unwrap();
    let sc = &reference_json(run)["scalars"];
    assert_eq!(
        sim.obs.len() as u64,
        sc["deseq2_prior_var_sim_n_obs"].as_u64().unwrap(),
        "{run} n_obs"
    );

    let oh = Table::read(&dir.join("deseq2_prior_var_obs_hist.csv"));
    let want_counts: Vec<i64> = oh.f64("count").iter().map(|&v| v as i64).collect();
    assert_eq!(sim.obs_counts, want_counts, "{run} obs counts");
    assert_close(
        &format!("{run} obs density"),
        &sim.obs_density,
        &oh.f64("density"),
        1e-14,
    );

    let simc = Table::read(&dir.join("deseq2_prior_var_sim_counts.csv"));
    assert_close(&format!("{run} grid"), &sim.grid, &simc.f64("grid"), 1e-14);
    for b in 1..=40 {
        let want: Vec<i64> = simc
            .f64(&format!("bin{b}"))
            .iter()
            .map(|&v| v as i64)
            .collect();
        let got: Vec<i64> = sim.sim_counts.iter().map(|r| r[b - 1]).collect();
        assert_eq!(got, want, "{run} sim counts bin{b}");
    }

    let kl = Table::read(&dir.join("deseq2_prior_var_kl.csv"));
    let a = assert_close(&format!("{run} kl"), &sim.kl, &kl.f64("kl"), 1e-8);
    let b = assert_close(
        &format!("{run} loess fitted"),
        &sim.loess_fitted,
        &kl.f64("loess_fitted"),
        1e-8,
    );
    let fine = Table::read(&dir.join("deseq2_prior_var_fine.csv"));
    assert_close(
        &format!("{run} fine grid"),
        &sim.fine_grid,
        &fine.f64("fine_grid"),
        1e-14,
    );
    let c = assert_close(
        &format!("{run} loess predicted"),
        &sim.fine_predicted,
        &fine.f64("loess_predicted"),
        1e-8,
    );
    assert_eq!(
        sim.argmin_index as u64,
        sc["deseq2_prior_var_sim_argmin_index"].as_u64().unwrap(),
        "{run} argmin index"
    );
    // fwrite/jsonlite keep 15 significant digits, so "exact" means within that rounding.
    let pv = sc["deseq2_dispPriorVar"].as_f64().unwrap();
    assert!(
        rel_diff(sim.prior_var, pv) < 1e-14,
        "{run} prior var {} vs {pv}",
        sim.prior_var
    );
    eprintln!("{run}: kl {a:e} fitted {b:e} predicted {c:e}");
}

#[test]
fn prior_var_simulation_matches_goldens() {
    let runs = if is_small_tier() {
        &RUNS[..1]
    } else {
        &RUNS[..]
    };
    for &run in runs {
        // df = m - p: samples minus coefficients (one SE_ column per coefficient).
        let dir = reference_dir(run);
        let m = Table::read(&dir.join("input_sample_info.csv")).nrow;
        let fit = Table::read(&dir.join("deseq2_fit_initial.csv"));
        let p = fit.names.iter().filter(|n| n.starts_with("SE_")).count();
        check_run(run, (m - p) as f64);
    }
}

fn df1_file(f: &str) -> Vec<f64> {
    let path = format!(
        "{}/tests/data/prior_var_df1/{f}",
        env!("CARGO_MANIFEST_DIR")
    );
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("{path}: {e}"))
        .split_whitespace()
        .map(|t| t.parse().unwrap())
        .collect()
}

/// m - p = 1 (review r1, stats item 8a): no corpus run has df = 1, where `rchisq` takes the
/// gamma GS branch. The 200 x 40 simulated counts must match R exactly, and the synthetic
/// residuals put the argmin in the interior so the loess and argmin are exercised too.
#[test]
fn deseq2_prior_var_df1_draws() {
    let sim = deseq2_core::prior_var::prior_var_simulation(&df1_file("obs.txt"), 1.0).unwrap();
    let want = df1_file("sim_counts.txt");
    assert_eq!(want.len(), 200 * 40);
    for (i, row) in sim.sim_counts.iter().enumerate() {
        let w: Vec<i64> = want[i * 40..(i + 1) * 40]
            .iter()
            .map(|&v| v as i64)
            .collect();
        assert_eq!(row, &w, "df1 sim counts row {}", i + 1);
    }
    let a = assert_close("df1 kl", &sim.kl, &df1_file("kl.txt"), 1e-12);
    let b = assert_close(
        "df1 loess fitted",
        &sim.loess_fitted,
        &df1_file("loess_fitted.txt"),
        1e-12,
    );
    let c = assert_close(
        "df1 loess predicted",
        &sim.fine_predicted,
        &df1_file("fine_predicted.txt"),
        1e-12,
    );
    let am = df1_file("argmin.txt");
    assert_eq!(sim.argmin_index, am[0] as usize, "df1 argmin index");
    assert_eq!(sim.prior_var, am[1], "df1 prior var");
    eprintln!("df1: kl {a:e} fitted {b:e} predicted {c:e}");
}
