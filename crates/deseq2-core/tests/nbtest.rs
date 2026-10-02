//! Wald / LRT fit and Cook's distances against `deseq2_fit_initial.csv` and
//! `deseq2_fit_initial_cooks.csv` for every DESeq2 run that reached the fit.

mod common;
use common::*;
use deseq2_core::{disp, fit, nbtest};

const TOL: f64 = 1e-8;

#[test]
fn initial_fit_matches_reference() {
    let runs = deseq2_runs(Some("deseq2_fit_initial.csv"));
    assert!(!runs.is_empty());
    let mut worst = 0.0f64;
    for run in &runs {
        let r = deseq_run(run);
        let dir = reference_dir(run);
        let m = r.samples.len();
        let sf = disp::size_factors(&r.counts, m).unwrap();
        let (x, _) = r.design.model_matrix();
        let d = disp::estimate_dispersions(&r.counts, &sf, &x).unwrap();
        let t = Table::read(&dir.join("deseq2_fit_initial.csv"));
        let lrt = t.has("LRTStatistic");
        let f = nbtest::nbinom_test(
            &r.counts,
            &sf,
            &d.base.all_zero,
            &d.dispersion_all(),
            &r.design,
            lrt,
        )
        .unwrap_or_else(|e| panic!("{run}: {e}"));
        let c = Table::read(&dir.join("deseq2_fit_initial_cooks.csv"));
        check_fit(run, &f, &t, &c, &mut worst);
        eprintln!("{run}: ok ({})", if lrt { "LRT" } else { "Wald" });
    }
    eprintln!("initial fit: {} runs, max rel diff {worst:e}", runs.len());
}

#[test]
fn replacement_and_refit_match_reference() {
    let runs = deseq2_runs(Some("deseq2_fit_initial.csv"));
    let mut worst = 0.0f64;
    let mut n_final = 0;
    for run in &runs {
        let r = deseq_run(run);
        let dir = reference_dir(run);
        let t0 = Table::read(&dir.join("deseq2_fit_initial.csv"));
        let lrt = t0.has("LRTStatistic");
        let f = fit::deseq(&r.counts, &r.design, lrt).unwrap_or_else(|e| panic!("{run}: {e}"));
        let sc = &reference_json(run)["scalars"];
        let any_rep = sc["deseq2_any_replaceable"].as_bool().unwrap();
        assert_eq!(f.replacement.is_some(), any_rep, "{run}: any replaceable");
        let Some(rep) = &f.replacement else { continue };
        let want_cut = sc["deseq2_replace_cooks_cutoff"].as_f64().unwrap();
        worst = worst.max(assert_close(
            &format!("{run} cutoff"),
            &[rep.cutoff],
            &[want_cut],
            TOL,
        ));
        let rt = Table::read(&dir.join("deseq2_replaceable.csv"));
        let want: Vec<bool> = rt.bool("replaceable").iter().map(|b| b.unwrap()).collect();
        assert_eq!(rep.replaceable, want, "{run}: replaceable");
        assert_eq!(
            rep.n_replaced as u64,
            sc["deseq2_n_replaced_genes"].as_u64().unwrap(),
            "{run}: n replaced"
        );
        let t = Table::read(&dir.join("deseq2_fit_final.csv"));
        assert_eq!(rep.replace, t.bool("replace"), "{run}: replace");
        if rep.n_replaced > 0 {
            let rc = Table::read(&dir.join("deseq2_replace_counts.csv"));
            let m = r.samples.len();
            for (j, s) in r.samples.iter().enumerate() {
                let got: Vec<f64> = rep.counts.iter().skip(j).step_by(m).copied().collect();
                assert_exact(&format!("{run} replaceCounts {s}"), &got, &rc.f64(s));
            }
        }
        let mut chk = |name: &str, got: &[f64]| {
            worst = worst.max(assert_close(
                &format!("{run} final {name}"),
                got,
                &t.f64(name),
                TOL,
            ));
        };
        chk("baseMean", &f.base.base_mean);
        chk("baseVar", &f.base.base_var);
        chk("dispGeneEst", &f.disp.gene_est);
        chk("dispFit", &f.disp.fit);
        chk("dispersion", &f.disp.dispersion);
        chk("dispMAP", &f.disp.map);
        assert_exact(
            &format!("{run} final dispGeneIter"),
            &f.disp.gene_iter,
            &t.f64("dispGeneIter"),
        );
        assert_exact(
            &format!("{run} final dispIter"),
            &f.disp.iter,
            &t.f64("dispIter"),
        );
        assert_eq!(
            f.disp.outlier,
            t.bool("dispOutlier"),
            "{run}: final dispOutlier"
        );
        let az: Vec<bool> = t.bool("allZero").iter().map(|b| b.unwrap()).collect();
        assert_eq!(f.base.all_zero, az, "{run}: final allZero");
        // The assays stay the initial ones.
        let c = Table::read(&dir.join("deseq2_fit_initial_cooks.csv"));
        check_fit(&format!("{run} final"), &f.test, &t, &c, &mut worst);
        n_final += 1;
        eprintln!("{run}: final ok ({} replaced)", rep.n_replaced);
    }
    assert!(n_final > 0);
    eprintln!("replacement: {n_final} runs, max rel diff {worst:e}");
}
