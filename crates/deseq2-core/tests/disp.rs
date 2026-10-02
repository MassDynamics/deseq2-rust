//! Size factors and the dispersion stage against `deseq2_size_factors.csv`, `deseq2_disp.csv`
//! and the `reference.json` scalars, for every DESeq2 run that reached the fit.

mod common;
use common::*;
use deseq2_core::disp;

const TOL: f64 = 1e-8;

#[test]
fn dispersion_stage_matches_reference() {
    let runs = deseq2_runs(Some("deseq2_disp.csv"));
    assert!(
        !runs.is_empty(),
        "no DESeq2 runs under {}",
        corpus_dir().display()
    );
    let mut worst = 0.0f64;
    for run in &runs {
        let r = deseq_run(run);
        let dir = reference_dir(run);
        let m = r.samples.len();
        let refj = reference_json(run);
        let sc = &refj["scalars"];

        let sf = disp::size_factors(&r.counts, m).unwrap();
        let sft = Table::read(&dir.join("deseq2_size_factors.csv"));
        assert_eq!(sft.str("id"), r.samples.as_slice(), "{run}: sample order");
        worst = worst.max(assert_close(
            &format!("{run} sf"),
            &sf,
            &sft.f64("size_factor"),
            TOL,
        ));

        let (x, _) = r.design.model_matrix();
        let d =
            disp::estimate_dispersions(&r.counts, &sf, &x).unwrap_or_else(|e| panic!("{run}: {e}"));
        let t = Table::read(&dir.join("deseq2_disp.csv"));
        assert_eq!(t.str("id"), r.ids.as_slice(), "{run}: gene order");
        let az: Vec<bool> = t.bool("allZero").iter().map(|b| b.unwrap()).collect();
        assert_eq!(d.base.all_zero, az, "{run}: allZero");
        let mut chk = |name: &str, got: Vec<f64>| {
            worst = worst.max(assert_close(
                &format!("{run} {name}"),
                &got,
                &t.f64(name),
                TOL,
            ));
        };
        let chk2 = |name: &str, got: Vec<f64>, worst: &mut f64| {
            *worst = worst.max(assert_close(
                &format!("{run} {name}"),
                &got,
                &t.f64(name),
                TOL,
            ));
        };
        let ex = |v: &[f64]| disp::expand(v, &az);
        let exu = |v: &[usize]| disp::expand(&v.iter().map(|i| *i as f64).collect::<Vec<_>>(), &az);
        chk("baseMean", d.base.base_mean.clone());
        chk("baseVar", d.base.base_var.clone());
        chk("alpha_init", ex(&d.gene.alpha_init));
        chk("dispGeneEst", ex(&d.gene.disp));
        chk("dispFit", ex(&d.disp_fit));
        // Discrete columns exact.
        assert_exact(
            &format!("{run} dispGeneIter"),
            &exu(&d.gene.iter),
            &t.f64("dispGeneIter"),
        );
        assert_exact(
            &format!("{run} dispIter"),
            &exu(&d.map.iter),
            &t.f64("dispIter"),
        );
        let out: Vec<Option<bool>> = {
            let mut it = d.map.outlier.iter();
            az.iter()
                .map(|z| if *z { None } else { Some(*it.next().unwrap()) })
                .collect()
        };
        assert_eq!(out, t.bool("dispOutlier"), "{run}: dispOutlier");

        assert_eq!(
            sc["deseq2_linear_mu"].as_bool().unwrap(),
            d.gene.linear_mu,
            "{run}: linearMu"
        );
        assert_eq!(
            sc["deseq2_fit_type"].as_str().unwrap(),
            d.function.fit_type(),
            "{run}: fitType"
        );
        if let disp::DispFunction::Parametric {
            asympt_disp,
            extra_pois,
        } = d.function
        {
            let co: Vec<f64> = sc["deseq2_trend_coefficients"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap())
                .collect();
            worst = worst.max(assert_close(
                &format!("{run} trend"),
                &[asympt_disp, extra_pois],
                &co,
                TOL,
            ));
        }
        let vl = sc["deseq2_varLogDispEsts"].as_f64().unwrap();
        worst = worst.max(assert_close(
            &format!("{run} varLog"),
            &[d.var_log],
            &[vl],
            TOL,
        ));
        let pv = sc["deseq2_dispPriorVar"].as_f64().unwrap();
        worst = worst.max(assert_close(
            &format!("{run} priorVar"),
            &[d.prior_var],
            &[pv],
            TOL,
        ));
        chk2("dispMAP", ex(&d.map.disp_map), &mut worst);
        chk2("dispersion", ex(&d.map.dispersion), &mut worst);
        eprintln!(
            "{run}: ok ({} genes, fit {})",
            r.ids.len(),
            d.function.fit_type()
        );
    }
    eprintln!(
        "dispersion stage: {} runs, max rel diff {worst:e}",
        runs.len()
    );
}
