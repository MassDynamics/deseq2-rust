#![allow(clippy::needless_range_loop)]
//! apeglm against the R golden corpus (`reference-shrink/*_shrink_apeglm*`).
//!
//! Every intermediate is compared and its measured gap printed (`--nocapture`). The final
//! columns must agree to rel 1e-8; a run may fall back to rel 1e-6 only if each row over 1e-8
//! carries a gradient certificate (our gradient at the MAP is no larger than R's `grad_at_map`).
mod common;
use common::*;
use shrink_core::apeglm::{shrink_apeglm, NbRow, SIGMA};
use shrink_core::dense::Mat;
use std::path::Path;

struct Inputs {
    ids: std::collections::HashMap<String, usize>,
    counts: Mat,
    sf: Vec<f64>,
    design: Mat,
    disp: Vec<f64>,
    mle: Vec<f64>,
    se: Vec<f64>,
}

fn load(run: &Path, cmp: &str) -> Inputs {
    let ct = Table::read(&run.join("shrink_counts.csv"));
    let samples: Vec<String> = ct.header[1..].to_vec();
    let g = ct.nrow;
    let n = samples.len();
    let mut cdata = Vec::with_capacity(g * n);
    for s in &samples {
        cdata.extend(ct.f(s));
    }
    let sft = Table::read(&run.join("shrink_size_factors.csv"));
    assert_eq!(sft.s("id"), samples, "size factor order");
    let dt = Table::read(&run.join(format!("{cmp}_design.csv")));
    assert_eq!(dt.s("id"), samples, "design order");
    let p = dt.header.len() - 1;
    let mut ddata = Vec::with_capacity(n * p);
    for h in &dt.header[1..] {
        ddata.extend(dt.f(h));
    }
    let it = Table::read(&run.join(format!("{cmp}_input.csv")));
    assert_eq!(it.nrow, g);
    assert_eq!(it.s("id"), ct.s("id"), "input order");
    Inputs {
        ids: ct
            .s("id")
            .into_iter()
            .enumerate()
            .map(|(i, s)| (s, i))
            .collect(),
        counts: Mat::from_col_major(g, n, cdata),
        sf: sft.f("size_factor"),
        design: Mat::from_col_major(n, p, ddata),
        disp: it.f("dispersion"),
        mle: it.f("lfc_mle"),
        se: it.f("lfc_se"),
    }
}

fn scalars(run: &Path) -> serde_json::Value {
    let s = std::fs::read_to_string(run.join("reference.json")).unwrap();
    let v: serde_json::Value = serde_json::from_str(&s).unwrap();
    v["scalars"].clone()
}

fn rel1(a: f64, b: f64) -> f64 {
    max_rel(&[a], &[b])
}

#[test]
fn apeglm_matches_r() {
    let runs = runs("_shrink_apeglm");
    if runs.is_empty() {
        return;
    }
    let mut worst_final: f64 = 0.0;
    for run in &runs {
        let name = run.file_name().unwrap().to_string_lossy().into_owned();
        let sc = scalars(run);
        for cmp in cmps(run, "apeglm") {
            let inp = load(run, &cmp);
            let p = inp.design.ncol;
            let k = |s: &str| sc[format!("{cmp}_apeglm_{s}")].clone();
            let no_shrink_r: Vec<usize> = match k("no_shrink") {
                serde_json::Value::Array(a) => {
                    a.iter().map(|v| v.as_u64().unwrap() as usize - 1).collect()
                }
                v => vec![v.as_u64().unwrap() as usize - 1],
            };
            let coef = (0..p).find(|j| !no_shrink_r.contains(j)).unwrap();
            let fit = shrink_apeglm(
                &inp.counts,
                &inp.sf,
                &inp.disp,
                &inp.design,
                coef,
                &inp.mle,
                &inp.se,
            )
            .unwrap_or_else(|e| panic!("{name}/{cmp}: {e}"));
            let g = inp.counts.nrow;
            println!("== {name} {cmp} (G = {g}, p = {p}, coef = {coef})");

            // prior
            let pv = rel1(fit.prior.prior_var, k("prior_var").as_f64().unwrap());
            let ps = rel1(fit.prior.prior_scale, k("prior_scale").as_f64().unwrap());
            println!("  prior_var rel {pv:.2e}, prior_scale rel {ps:.2e} (golden has 15 digits)");
            assert!(pv < 1e-13 && ps < 1e-13);
            assert_eq!(
                fit.prior.iter as i64,
                k("uniroot")["iter"].as_i64().unwrap()
            );
            assert_eq!(fit.prior.kept as u64, k("mle_kept").as_u64().unwrap());

            // the per-evaluation objective along R's traced paths
            let pre = Table::read(&run.join(format!("{cmp}_apeglm_prefit.csv")));
            let cnst_r = pre.f("cnst");
            let offset: Vec<f64> = inp.sf.iter().map(|&v| rnum::glibm::ln(v)).collect();
            let shrink = vec![coef];
            let (mut path_f, mut path_g, mut path_n) = (0.0f64, 0.0f64, 0usize);
            for fk in ["fit1", "fit2"] {
                let pt = Table::read(&run.join(format!("{cmp}_apeglm_path_{fk}.csv")));
                let ids = pt.s("id");
                let b = pt.numbered("beta");
                let gr = pt.numbered("grad");
                let f = pt.f("f");
                for r in 0..pt.nrow {
                    let i = inp.ids[&ids[r]];
                    let y: Vec<f64> = (0..inp.counts.ncol).map(|j| inp.counts.at(i, j)).collect();
                    let row = NbRow {
                        x: &inp.design,
                        y: &y,
                        size: 1.0 / inp.disp[i],
                        offset: &offset,
                        s: fit.prior.prior_scale,
                        no_shrink: &fit.no_shrink,
                        shrink: &shrink,
                    };
                    let beta: Vec<f64> = b.iter().map(|c| c[r]).collect();
                    let mut gg = vec![0.0; p];
                    let ff = row.f_grad_cpp(&beta, cnst_r[i], &mut gg);
                    path_f = path_f.max(rel1(ff, f[r]));
                    for j in 0..p {
                        path_g = path_g.max(rel1(gg[j], gr[j][r]));
                    }
                    path_n += 1;
                }
            }
            println!("  path f/grad at R's betas ({path_n} evals): f rel {path_f:.2e}, grad rel {path_g:.2e}");
            let _ = SIGMA;
            assert!(
                path_f == 0.0 && path_g == 0.0,
                "{name}/{cmp}: path objective drifted"
            );

            // prefit
            let nz = pre.b("nonzero");
            let col = |f: &dyn Fn(usize) -> f64| (0..g).map(f).collect::<Vec<f64>>();
            let pf = |i: usize| fit.prefit[i].as_ref();
            let gap_cnst = max_rel(
                &col(&|i| pf(i).map_or(f64::NAN, |r| r.cnst_raw)),
                &pre.f("cnst_raw"),
            );
            let mut gap_beta: f64 = 0.0;
            let mut gap_val: f64 = 0.0;
            let (mut st_mis, mut ne_mis, mut conv_mis) = (0, 0, 0);
            for fk in ["fit1", "fit2"] {
                let rb = pre.numbered(&format!("{fk}_beta"));
                let rv = pre.f(&format!("{fk}_value"));
                let rs = pre.f(&format!("{fk}_status"));
                let rn = pre.f(&format!("{fk}_nevals"));
                for i in 0..g {
                    let Some(r) = pf(i) else {
                        assert!(!nz[i]);
                        continue;
                    };
                    let o = if fk == "fit1" { &r.fit1 } else { &r.fit2 };
                    for j in 0..p {
                        gap_beta = gap_beta.max(max_rel(&[o.x[j]], &[rb[j][i]]));
                    }
                    gap_val = gap_val.max(max_rel(&[o.fx], &[rv[i]]));
                    st_mis += (o.status as f64 != rs[i]) as usize;
                    ne_mis += (o.nevals as f64 != rn[i]) as usize;
                }
            }
            let rconv = pre.f("prefit_conv");
            for i in 0..g {
                if let Some(r) = pf(i) {
                    conv_mis += (r.conv as f64 != rconv[i]) as usize;
                }
            }
            let gap_delta = max_rel(
                &col(&|i| pf(i).map_or(f64::NAN, |r| r.delta)),
                &pre.f("delta"),
            );
            println!(
                "  prefit: cnst rel {gap_cnst:.2e}, betas rel {gap_beta:.2e}, value rel {gap_val:.2e}, delta rel {gap_delta:.2e}; status/nevals/conv mismatches {st_mis}/{ne_mis}/{conv_mis}"
            );

            assert!(
                gap_cnst == 0.0 && gap_beta == 0.0 && gap_val == 0.0,
                "{name}/{cmp}: prefit drifted"
            );
            assert!(
                st_mis + ne_mis + conv_mis == 0,
                "{name}/{cmp}: prefit status/nevals/conv"
            );

            // row pass
            let rp = Table::read(&run.join(format!("{cmp}_apeglm_rowpass.csv")));
            let rw = |i: usize| fit.rows[i].as_ref();
            let gap_cnst2 = max_rel(
                &col(&|i| rw(i).map_or(f64::NAN, |r| r.cnst2)),
                &rp.f("cnst2"),
            );
            let fb_r = rp.b("fallback");
            let fb_mis = (0..g)
                .filter(|&i| rw(i).is_some_and(|r| r.fallback) != fb_r[i])
                .count();
            let n_fb = (0..g)
                .filter(|&i| rw(i).is_some_and(|r| r.fallback))
                .count();
            let gap_fbv = max_rel(
                &col(&|i| rw(i).and_then(|r| r.fb_value).unwrap_or(f64::NAN)),
                &rp.f("fb_value"),
            );
            let fbc_mis = (0..g)
                .filter(|&i| {
                    let a = rw(i)
                        .and_then(|r| r.fb_fncount)
                        .map_or(f64::NAN, |v| v as f64);
                    let b = rp.f("fb_fn")[i];
                    !(a == b || (a.is_nan() && b.is_nan()))
                })
                .count();
            let mut gap_h: f64 = 0.0;
            let mut gap_fh: f64 = 0.0;
            for a in 0..p {
                for b in 0..p {
                    let h = col(&|i| {
                        rw(i)
                            .and_then(|r| r.hess.as_ref())
                            .map_or(f64::NAN, |h| h[a + b * p])
                    });
                    gap_h = gap_h.max(max_rel(&h, &rp.f(&format!("hess_{}_{}", a + 1, b + 1))));
                    let fh = col(&|i| rw(i).map_or(f64::NAN, |r| r.final_hess[a + b * p]));
                    gap_fh = gap_fh.max(max_rel(
                        &fh,
                        &rp.f(&format!("final_hess_{}_{}", a + 1, b + 1)),
                    ));
                }
            }
            println!(
                "  rowpass: cnst2 rel {gap_cnst2:.2e}, hess rel {gap_h:.2e}, final_hess rel {gap_fh:.2e}; fallback {n_fb} (mismatch {fb_mis}), fb fncount mismatch {fbc_mis}, fb value rel {gap_fbv:.2e}"
            );
            assert_eq!(n_fb as u64, k("n_fallback").as_u64().unwrap());
            assert!(
                fb_mis + fbc_mis == 0,
                "{name}/{cmp}: fallback set or counts differ"
            );
            assert!(
                gap_cnst2 < 1e-12 && gap_h < 1e-12 && gap_fh < 1e-12,
                "{name}/{cmp}: row pass drifted"
            );

            // final
            let fin = Table::read(&run.join(format!("{cmp}_apeglm_final.csv")));
            let mut gaps: Vec<(String, f64)> = vec![];
            for j in 0..p {
                gaps.push((
                    format!("map{}", j + 1),
                    max_rel(fit.map.col(j), &fin.f(&format!("map{}", j + 1))),
                ));
                gaps.push((
                    format!("sd{}", j + 1),
                    max_rel(fit.sd.col(j), &fin.f(&format!("sd{}", j + 1))),
                ));
            }
            for (nm, v) in [
                ("interval_lo", &fit.interval_lo),
                ("interval_hi", &fit.interval_hi),
                ("fsr", &fit.fsr),
                ("svalue", &fit.svalue),
                ("diag_conv", &fit.diag_conv),
                ("diag_count", &fit.diag_count),
                ("log2FoldChange", &fit.log2_fold_change),
                ("lfcSE", &fit.lfc_se),
                ("CrILeft", &fit.cri_left),
                ("CrIRight", &fit.cri_right),
            ] {
                gaps.push((nm.to_string(), max_rel(v, &fin.f(nm))));
            }
            let line: Vec<String> = gaps.iter().map(|(n, v)| format!("{n} {v:.1e}")).collect();
            println!("  final: {}", line.join(", "));
            {
                // R's nbinomGr at R's MAP, recomputed: bit-level check of the R-side gradient
                let ga = fin.numbered("grad_at_map");
                let rmap: Vec<Vec<f64>> = (0..p).map(|j| fin.f(&format!("map{}", j + 1))).collect();
                let (mut same, mut tot, mut worst_abs) = (0usize, 0usize, 0.0f64);
                for i in 0..g {
                    if rmap[0][i].is_nan() || ga[0][i].is_nan() {
                        continue;
                    }
                    let y: Vec<f64> = (0..inp.counts.ncol).map(|j| inp.counts.at(i, j)).collect();
                    let row = NbRow {
                        x: &inp.design,
                        y: &y,
                        size: 1.0 / inp.disp[i],
                        offset: &offset,
                        s: fit.prior.prior_scale,
                        no_shrink: &fit.no_shrink,
                        shrink: &shrink,
                    };
                    let beta: Vec<f64> = (0..p).map(|j| rmap[j][i]).collect();
                    let gg = row.nbinom_gr(&beta);
                    tot += 1;
                    let mut eq = true;
                    for j in 0..p {
                        if gg[j] != ga[j][i] {
                            eq = false;
                            worst_abs = worst_abs.max((gg[j] - ga[j][i]).abs());
                        }
                    }
                    same += eq as usize;
                }
                println!("  nbinomGr at R's MAP: {same}/{tot} rows bit-identical, worst abs gap {worst_abs:.2e}");
            }
            let worst = gaps.iter().map(|x| x.1).fold(0.0, f64::max);
            worst_final = worst_final.max(worst);
            if worst > 1e-8 {
                // gradient certificate, row by row, for rows whose MAP is off by more than 1e-8
                let ga = fin.numbered("grad_at_map");
                let rmap: Vec<Vec<f64>> = (0..p).map(|j| fin.f(&format!("map{}", j + 1))).collect();
                let mut uncert = 0;
                for i in 0..g {
                    let off = (0..p).any(|j| max_rel(&[fit.map.at(i, j)], &[rmap[j][i]]) > 1e-8);
                    if !off {
                        continue;
                    }
                    let y: Vec<f64> = (0..inp.counts.ncol).map(|j| inp.counts.at(i, j)).collect();
                    let row = NbRow {
                        x: &inp.design,
                        y: &y,
                        size: 1.0 / inp.disp[i],
                        offset: &offset,
                        s: fit.prior.prior_scale,
                        no_shrink: &fit.no_shrink,
                        shrink: &shrink,
                    };
                    let beta: Vec<f64> = (0..p).map(|j| fit.map.at(i, j)).collect();
                    let ours = row.nbinom_gr(&beta);
                    let n_ours = ours.iter().map(|v| v * v).sum::<f64>().sqrt();
                    let n_r = (0..p).map(|j| ga[j][i] * ga[j][i]).sum::<f64>().sqrt();
                    if n_ours > n_r {
                        uncert += 1;
                    }
                }
                println!("  certificate: {uncert} rows off by > 1e-8 without a gradient no larger than R's");
                assert!(
                    worst < 1e-6 && uncert == 0,
                    "{name}/{cmp}: final gap {worst:e}"
                );
            }
        }
    }
    println!("worst final gap over all runs: {worst_final:.2e}");
}
