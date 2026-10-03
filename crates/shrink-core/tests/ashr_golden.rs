//! ashr / mixsqp against the reference-shrink goldens.
//!
//! Gates (see status-shrink.md for the reasoning):
//! - deterministic intermediates (grid, L, lnorm, the 20 EM iterates, and the posterior table
//!   computed from R's own pi) at rel 1e-8;
//! - end to end, pi, PosteriorMean, PosteriorSD and lfsr at rel 1e-8. This needs the
//!   active-set path to match R's step for step, including the solves below rcond = eps that
//!   go to Armadillo's dgelsd fallback (tests/mixsqp_solve.rs checks that bit for bit);
//! - the mix-SQP certificate on top: Rust's x passes mixsqp's convergence test, its
//!   Frank-Wolfe gap is no worse than twice R's, and |f_rust - f_R| is within the sum of the
//!   two FW gaps.

mod common;
use common::*;
use shrink_core::ashr::{ash_shrink, posterior_table};
use shrink_core::dense::Mat;
use shrink_core::linalg::SolveRoute;
use shrink_core::mixsqp::{kkt, objective};

const TOL: f64 = 1e-8;

#[test]
fn ashr_golden() {
    let runs = runs("_shrink_ashr");
    // No corpus fails the test rather than passing it empty (review r1, D-11).
    assert_eq!(
        runs.len(),
        5,
        "_shrink_ashr runs in the corpus (set MD_COUNT_CORPUS_DIR)"
    );
    let mut n_cmp = 0;
    let mut worst = std::collections::BTreeMap::<&str, f64>::new();
    let bump = |k: &'static str, v: f64, w: &mut std::collections::BTreeMap<&str, f64>| {
        let e = w.entry(k).or_insert(0.0);
        if v > *e {
            *e = v;
        }
    };
    println!("run cmp | grid L lnorm em | route_mismatch/qp | sqp_it R/Rust | obj_rel gmin_rust | pi_rel | post(Rpi) | e2e pm psd lfsr");
    for run in &runs {
        for cmp in cmps(run, "ashr") {
            n_cmp += 1;
            let p = |s: &str| run.join(format!("{cmp}_ashr_{s}.csv"));
            let data = Table::read(&p("data"));
            let x = data.f("x");
            let s = data.f("s");
            let fit = ash_shrink(&x, &s).expect("ash_shrink");

            // grid
            let grid = Table::read(&p("grid"));
            let g_grid = max_rel(&fit.mixsd, &grid.f("mixsd"));
            assert_eq!(fit.nonzero_cols, grid.b("nonzero_col"));
            assert_eq!(fit.excluded, data.b("excluded"));

            // L and lnorm
            let lt = Table::read(&p("L"));
            let lcols = lt.numbered("V");
            assert_eq!(lcols.len(), fit.lik.ncol);
            let mut g_l: f64 = 0.0;
            for (c, col) in lcols.iter().enumerate() {
                g_l = g_l.max(max_rel(fit.lik.col(c), col));
            }
            let g_lnorm = max_rel(&fit.lnorm, &Table::read(&p("lnorm")).f("lnorm"));

            // EM iterates
            let ms = fit.mixsqp.as_ref().expect("mixsqp ran");
            let em = Table::read(&p("trace_em"));
            let emc = em.numbered("x");
            let mut g_em: f64 = 0.0;
            for (it, xr) in ms.em_iterates.iter().enumerate() {
                let rrow: Vec<f64> = emc.iter().map(|c| c[it]).collect();
                g_em = g_em.max(max_rel(xr, &rrow));
            }
            let mx = Table::read(&p("mixsqp_x"));
            g_em = g_em.max(max_rel(ms.em_iterates.last().unwrap(), &mx.f("em_x")));

            // route of each active-set solve vs the R trace, where both took the same path
            let qp = Table::read(&p("trace_qp"));
            let r_ok = qp.f("noapprox_ok");
            let r_it = qp.f("sqp_iter");
            let r_sqp0 = r_it.iter().filter(|v| **v == 0.0).count();
            let mut route_mis = 0;
            for (q, step) in ms
                .qp
                .iter()
                .enumerate()
                .take(r_sqp0.min(ms.qp.iter().filter(|s| s.sqp_iter == 0).count()))
            {
                let r_approx = r_ok[q] == 0.0;
                if r_approx != (step.route == SolveRoute::Approx) {
                    route_mis += 1;
                }
            }
            let sqp_r = Table::read(&p("trace_sqp")).nrow;

            // KKT certificate
            let nz: Vec<usize> = (0..fit.nonzero_cols.len())
                .filter(|&c| fit.nonzero_cols[c])
                .collect();
            let mut sub = Mat::zeros(fit.lik.nrow, nz.len());
            for (cc, &c) in nz.iter().enumerate() {
                sub.col_mut(cc).copy_from_slice(fit.lik.col(c));
            }
            let x_r = mx.f("x");
            let f_r = objective(&sub, &x_r);
            let f_rust = objective(&sub, &ms.x);
            let obj_rel = ((f_rust - f_r) / f_r).abs();
            let (g_rust, gmin_rust, stat_rust) = kkt(&sub, &ms.x);
            let (g_r, gmin_r, _) = kkt(&sub, &x_r);
            // Frank-Wolfe gap g.x - min_j g_j bounds f(x) - f* on the simplex (convex f)
            let fw = |g: &[f64], x: &[f64]| {
                g.iter().zip(x).map(|(a, b)| a * b).sum::<f64>()
                    - g.iter().cloned().fold(f64::INFINITY, f64::min)
            };
            let fw_rust = fw(&g_rust, &ms.x);
            let fw_r = fw(&g_r, &x_r);
            assert!(ms.converged, "{cmp}: mixsqp did not converge");
            assert!(ms.gmin >= -1e-8, "{cmp}: convergence statistic {}", ms.gmin);
            // both optima lie within their FW gaps of f*, so the objectives can differ by at most the sum
            assert!(
                (f_rust - f_r).abs() <= fw_rust + fw_r,
                "{cmp}: objective gap {:e} exceeds the FW bound {:e}",
                (f_rust - f_r).abs(),
                fw_rust + fw_r
            );

            let pi_r = Table::read(&p("pi")).f("pi");
            let g_pi = max_rel(&fit.pi, &pi_r);

            // posterior table from R's pi: deterministic, rel 1e-8
            let fin = Table::read(&p("final"));
            let t_r = posterior_table(&x, &s, &fit.excluded, &fit.mixsd, &pi_r);
            let cols = [
                ("PosteriorMean", &t_r.posterior_mean),
                ("PosteriorSD", &t_r.posterior_sd),
                ("NegativeProb", &t_r.negative_prob),
                ("ZeroProb", &t_r.zero_prob),
                ("lfsr", &t_r.lfsr),
                ("svalue", &t_r.svalue),
                ("CrILeft", &t_r.cri_left),
                ("CrIRight", &t_r.cri_right),
            ];
            let mut g_post: f64 = 0.0;
            for (name, v) in cols {
                let g = max_rel(v, &fin.f(name));
                bump(name, g, &mut worst);
                g_post = g_post.max(g);
            }
            assert_eq!(fin.f("log2FoldChange"), fin.f("PosteriorMean"));

            // end to end
            let t = &fit.table;
            let e_pm = max_rel(&t.posterior_mean, &fin.f("PosteriorMean"));
            let e_psd = max_rel(&t.posterior_sd, &fin.f("PosteriorSD"));
            let e_lfsr = max_rel(&t.lfsr, &fin.f("lfsr"));
            println!(
                "{} {cmp} | {g_grid:.1e} {g_l:.1e} {g_lnorm:.1e} {g_em:.1e} | {route_mis}/{} | {sqp_r}/{} | {obj_rel:.1e} fw {fw_rust:.1e}/{fw_r:.1e} gmin {gmin_rust:.1e}/{gmin_r:.1e} stat {stat_rust:.1e} | {g_pi:.1e} | {g_post:.1e} | {e_pm:.1e} {e_psd:.1e} {e_lfsr:.1e}",
                run.file_name().unwrap().to_string_lossy(),
                ms.qp.len(),
                ms.sqp.len()
            );
            bump("grid", g_grid, &mut worst);
            bump("L", g_l, &mut worst);
            bump("lnorm", g_lnorm, &mut worst);
            bump("em", g_em, &mut worst);
            bump("obj_rel", obj_rel, &mut worst);
            bump("fw_rust", fw_rust, &mut worst);
            bump("fw_r", fw_r, &mut worst);
            bump("pi_e2e", g_pi, &mut worst);
            bump("pm_e2e", e_pm, &mut worst);
            bump("psd_e2e", e_psd, &mut worst);
            bump("lfsr_e2e", e_lfsr, &mut worst);
            for (k, v) in [
                ("grid", g_grid),
                ("L", g_l),
                ("lnorm", g_lnorm),
                ("em", g_em),
                ("posterior(R pi)", g_post),
                ("pi", g_pi),
                ("PosteriorMean", e_pm),
                ("PosteriorSD", e_psd),
                ("lfsr", e_lfsr),
            ] {
                assert!(v <= TOL, "{cmp}: {k} gap {v:e} > {TOL:e}");
            }
            // certificate: Rust is as close to optimal as R's own answer (same FW gap order)
            assert!(
                fw_rust <= 2.0 * fw_r + 1e-9,
                "{cmp}: Rust FW gap {fw_rust:e} vs R {fw_r:e}"
            );
            assert_eq!(route_mis, 0, "{cmp}: solve routes differ from R's trace");
        }
    }
    println!("worst gaps over {n_cmp} comparisons: {worst:#?}");
    // A run that loses its comparison files must not pass quietly (review deseq2 r2, SE-m4).
    assert_eq!(n_cmp, 11, "ashr comparisons in the corpus");
}
