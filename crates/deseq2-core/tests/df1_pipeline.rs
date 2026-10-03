//! `DESeq()` plus `results()` at m - p = 1 (review r1, stats item 8b): `DESeq(dds)` with
//! `~ cond + dose` on 2 x 2 samples plus a numeric dose (`tests/data/df1_pipeline/make_case.R`,
//! R 4.5.0 + DESeq2 1.50.2), then `results()` for the condition contrast and the dose
//! coefficient. The prior variance comes from the df = 1 simulation with an interior argmin
//! (0.7608, not the 0.25 floor), so this gates dispMAP and the results tables on that path. The
//! Cook's cutoff is checked as a value only: no cell has 3 samples, so it filters nothing here.

mod common;
use common::*;
use deseq2_core::design::{Design, Factor, Var};
use deseq2_core::fit;
use deseq2_core::results::{results, ResultsData, Which};
use std::path::PathBuf;

const TOL: f64 = 1e-8;

fn data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/df1_pipeline")
}

fn scalar(key: &str) -> String {
    let path = data_dir().join("scalars.txt");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path:?}: {e}"));
    text.lines()
        .find_map(|l| l.strip_prefix(&format!("{key} ")))
        .unwrap_or_else(|| panic!("scalars.txt: no {key}"))
        .to_string()
}

#[test]
fn df1_pipeline_matches_r_deseq() {
    let dir = data_dir();
    let cd = Table::read(&dir.join("coldata.csv"));
    let m = cd.nrow;
    let design = Design {
        n: m,
        vars: vec![
            Var::Factor(Factor::new("cond", cd.str("cond"))),
            Var::Numeric {
                name: "dose".into(),
                values: cd.f64("dose"),
            },
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
    assert_eq!(m - f.test.coef_names.len(), 1, "df = m - p");
    let want = Table::read(&dir.join("expected.csv"));

    let sf: Vec<f64> = scalar("sizeFactors")
        .split_whitespace()
        .map(|t| t.parse().unwrap())
        .collect();
    assert_close("sizeFactors", &f.sf, &sf, 1e-14);
    assert_eq!(f.dispersions.function.fit_type(), scalar("fitType"));
    let pv: f64 = scalar("dispPriorVar").parse().unwrap();
    assert!(pv > 0.5, "the reference argmin must be interior");
    // The fine grid is 1000 points on [0, 8], so equal prior variances mean the same argmin.
    assert!(
        rel_diff(f.dispersions.prior_var, pv) < 1e-14,
        "dispPriorVar {} vs {pv}",
        f.dispersions.prior_var
    );
    let vl: f64 = scalar("varLogDispEsts").parse().unwrap();
    assert!(
        rel_diff(f.dispersions.var_log, vl) < 1e-12,
        "varLogDispEsts"
    );

    let mut worst = 0.0f64;
    let mut close = |label: &str, got: &[f64], col: &str| {
        worst = worst.max(assert_close(label, got, &want.f64(col), TOL));
    };
    close("baseMean", &f.base.base_mean, "baseMean");
    close("dispGeneEst", &f.disp.gene_est, "dispGeneEst");
    close("dispFit", &f.disp.fit, "dispFit");
    close("dispMAP", &f.disp.map, "dispMAP");
    close("dispersion", &f.disp.dispersion, "dispersion");
    assert_eq!(f.disp.outlier, want.bool("dispOutlier"), "dispOutlier");
    let n_outlier = f.disp.outlier.iter().filter(|o| **o == Some(true)).count();
    assert_eq!(n_outlier, 15, "the reference has 15 dispersion outliers");

    let data = ResultsData {
        design: &f.design,
        test: &f.test,
        base_mean: &f.base.base_mean,
        all_zero: &f.base.all_zero,
        replace: f.replacement.as_ref().map(|r| r.replace.as_slice()),
        counts: &f.counts,
        fit_counts: f.fit_counts(),
        sf: &f.sf,
        dispersion: &f.disp.dispersion,
    };
    let cond = Which::Contrast {
        factor: "cond".into(),
        num: "B".into(),
        den: "A".into(),
    };
    for (which, pre) in [(cond, ""), (Which::Name("dose".into()), "dose_")] {
        let r = results(&data, &which, true, 0.1).unwrap();
        // R: sprintf("%.17g", qf(.99, 3, 1)). No cell here has 3 samples, so the cutoff filters
        // nothing and only this assertion gates the df = 1 value.
        assert!(
            rel_diff(r.cooks_cutoff, 5_403.352_013_738_532) < 1e-12,
            "cooksCutoff {}",
            r.cooks_cutoff
        );
        close(&format!("{pre}lfc"), &r.lfc, &format!("{pre}lfc"));
        close(&format!("{pre}lfcSE"), &r.se, &format!("{pre}lfcSE"));
        close(&format!("{pre}stat"), &r.stat, &format!("{pre}stat"));
        close(&format!("{pre}pvalue"), &r.pvalue, &format!("{pre}pvalue"));
        close(&format!("{pre}padj"), &r.padj, &format!("{pre}padj"));
        if pre.is_empty() {
            let na = r.padj.iter().filter(|v| v.is_nan()).count();
            let hits = r.padj.iter().filter(|v| **v < 0.1).count();
            assert_eq!((na, hits), (500, 98), "independent filtering NA and hits");
        }
    }
    eprintln!("df1 pipeline: {n} genes, max rel diff {worst:e}");
}
