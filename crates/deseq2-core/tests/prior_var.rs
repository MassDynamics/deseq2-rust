//! Step 1 gate: the m - p <= 3 prior variance simulation (R RNG, hist, loess) reproduces
//! every `deseq2_prior_var_*` golden on the six airway ctlfactor runs. Residuals come from
//! the `deseq2_disp.csv` stage golden so this test isolates the simulation.

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
    for run in RUNS {
        // df = m - p: samples minus coefficients (one SE_ column per coefficient).
        let dir = reference_dir(run);
        let m = Table::read(&dir.join("input_sample_info.csv")).nrow;
        let fit = Table::read(&dir.join("deseq2_fit_initial.csv"));
        let p = fit.names.iter().filter(|n| n.starts_with("SE_")).count();
        check_run(run, (m - p) as f64);
    }
}
