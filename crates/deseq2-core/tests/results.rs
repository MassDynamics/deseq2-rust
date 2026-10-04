//! `results()`, the production pairwise / ANOVA engine and `lfcShrink` against the reference:
//! per comparison `deseq2_cmp_NN.csv` (+ filtering, relevel refit, shrunk columns and scalars),
//! the ANOVA contrast and omnibus tables, `reference_output.csv` on the columns the R self
//! check compared, and the expected-error runs.

mod common;
use common::*;
use deseq2_core::engine::{run_deseq2_diag, Comparison, Control, DeseqInput};
use deseq2_core::results::ResultsTable;

const TOL: f64 = 1e-8;

fn param_str(run: &str, key: &str, default: &str) -> String {
    manifest(run)["params"][key]
        .as_str()
        .unwrap_or(default)
        .to_string()
}

/// The engine input for a run, from the reference's input files and the manifest params.
fn engine_input(run: &str) -> DeseqInput {
    let dir = reference_dir(run);
    let man = manifest(run);
    let si = Table::read(&dir.join("input_sample_info.csv"));
    let samples = si.str("replicate").to_vec();
    let cond = man["params"]["condition_col"].as_str().unwrap().to_string();
    let controls = control_specs(run)
        .into_iter()
        .map(|(c, numeric)| Control {
            values: si.str(&c).to_vec(),
            name: c,
            numeric,
        })
        .collect();
    let cnt = Table::read(&dir.join("input_counts.csv"));
    let cols: Vec<Vec<f64>> = samples.iter().map(|s| cnt.f64(s)).collect();
    let ids = cnt.str("id").to_vec();
    let mut counts = Vec::with_capacity(ids.len() * samples.len());
    for g in 0..ids.len() {
        for c in &cols {
            counts.push(c[g]);
        }
    }
    let cmp = Table::read(&dir.join("input_comparisons.csv"));
    let comparisons = (0..cmp.nrow)
        .map(|i| Comparison {
            left: cmp.str("left")[i].clone(),
            right: cmp.str("right")[i].clone(),
            encoded_left: cmp.str("encoded_left")[i].clone(),
            encoded_right: cmp.str("encoded_right")[i].clone(),
        })
        .collect();
    DeseqInput {
        gene_ids: ids,
        condition: si.str(&cond).to_vec(),
        sample_ids: samples,
        counts,
        condition_col: cond,
        controls,
        comparisons,
        anova: man["mode"].as_str() == Some("anova"),
        alpha: man["params"]["deseq2_alpha"].as_f64().unwrap_or(0.05),
        shrink: param_str(run, "deseq2_lfc_shrinkage", "none"),
        entity_type: man["entity_type"].as_str().unwrap_or("gene").to_string(),
    }
}

fn is_expected_error(run: &str) -> bool {
    reference_json(run)["self_check"]["kind"].as_str() == Some("expected_error")
}

fn check_table(
    label: &str,
    r: &ResultsTable,
    t: &Table,
    ids: &[String],
    scalars: &serde_json::Value,
    tag: &str,
    worst: &mut f64,
) {
    assert_eq!(t.str("id"), ids, "{label}: row ids");
    let mut chk = |name: &str, got: &[f64]| {
        *worst = worst.max(assert_close(
            &format!("{label} {name}"),
            got,
            &t.f64(name),
            TOL,
        ));
    };
    chk("baseMean", &r.base_mean);
    chk("log2FoldChange", &r.lfc);
    chk("lfcSE", &r.se);
    chk("stat", &r.stat);
    chk("pvalue", &r.pvalue);
    chk("padj", &r.padj);
    assert_exact(
        &format!("{label} cooks_outlier"),
        &opt_f64(&r.cooks_outlier),
        &opt_f64(&t.bool("cooks_outlier")),
    );
    let b = |v: &[bool]| v.iter().map(|x| *x as u8 as f64).collect::<Vec<_>>();
    assert_exact(
        &format!("{label} cooks_rescued"),
        &b(&r.cooks_rescued),
        &opt_f64(&t.bool("cooks_rescued")),
    );
    assert_exact(
        &format!("{label} now_zero"),
        &b(&r.now_zero),
        &opt_f64(&t.bool("now_zero")),
    );
    assert_eq!(
        scalars[format!("{tag}_path")].as_str(),
        Some(r.path),
        "{label} path"
    );
    *worst = worst.max(assert_close(
        &format!("{label} cooks_cutoff"),
        &[r.cooks_cutoff],
        &[scalars[format!("{tag}_cooks_cutoff")].as_f64().unwrap()],
        TOL,
    ));
    let fpath = reference_dir_of(label).join(format!("{tag}_filtering.csv"));
    match &r.filtering {
        Some(f) => {
            let ft = Table::read(&fpath);
            let mut chk = |name: &str, got: &[f64]| {
                *worst = worst.max(assert_close(
                    &format!("{label} filtering {name}"),
                    got,
                    &ft.f64(name),
                    TOL,
                ));
            };
            chk("theta", &f.theta);
            chk("cutoff", &f.cutoffs);
            chk("numRej", &f.num_rej);
            chk("lowess_y", &f.lowess_y);
            assert_eq!(
                scalars[format!("{tag}_filter_j")].as_u64(),
                Some(f.j as u64),
                "{label} filter_j"
            );
            *worst = worst.max(assert_close(
                &format!("{label} filter_threshold"),
                &[f.threshold],
                &[scalars[format!("{tag}_filter_threshold")].as_f64().unwrap()],
                TOL,
            ));
        }
        None => assert!(!fpath.exists(), "{label}: R ran independent filtering"),
    }
}

/// `label` starts with the run name.
fn reference_dir_of(label: &str) -> std::path::PathBuf {
    reference_dir(label.split(' ').next().unwrap())
}

#[test]
fn results_and_shrinkage_match_reference() {
    let runs: Vec<String> = deseq2_runs(Some("deseq2_fit_initial.csv"));
    assert!(!runs.is_empty());
    let mut worst = 0.0f64;
    let mut worst_shrink = 0.0f64;
    let (mut n_cmp, mut n_shrunk) = (0, 0);
    for run in &runs {
        let dir = reference_dir(run);
        let inp = engine_input(run);
        let (_, diag) = run_deseq2_diag(&inp).unwrap_or_else(|e| panic!("{run}: {e}"));
        let ids: Vec<String> = diag
            .kept_idx
            .iter()
            .map(|&g| inp.gene_ids[g].clone())
            .collect();
        let scalars = &reference_json(run)["scalars"];
        if inp.anova {
            for (i, r) in diag.anova_lfc.iter().enumerate() {
                let tag = format!("deseq2_anova_lfc_{:02}", i + 1);
                let t = Table::read(&dir.join(format!("{tag}.csv")));
                check_table(
                    &format!("{run} {tag}"),
                    r,
                    &t,
                    &ids,
                    scalars,
                    &tag,
                    &mut worst,
                );
                n_cmp += 1;
            }
            let tag = "deseq2_anova_omnibus";
            let t = Table::read(&dir.join(format!("{tag}.csv")));
            check_table(
                &format!("{run} {tag}"),
                diag.omnibus.as_ref().unwrap(),
                &t,
                &ids,
                scalars,
                tag,
                &mut worst,
            );
            eprintln!("{run}: ok (ANOVA, {} contrasts)", diag.anova_lfc.len());
            continue;
        }
        for (i, cd) in diag.comparisons.iter().enumerate() {
            let tag = format!("deseq2_cmp_{:02}", i + 1);
            let label = format!("{run} {tag}");
            let t = Table::read(&dir.join(format!("{tag}.csv")));
            check_table(&label, &cd.results, &t, &ids, scalars, &tag, &mut worst);
            assert_eq!(
                scalars[format!("{tag}_relevel_refit")].as_bool(),
                Some(cd.relevel_fit.is_some()),
                "{label} relevel_refit"
            );
            if let Some(f) = &cd.relevel_fit {
                let ft = Table::read(&dir.join(format!("{tag}_relevel_fit.csv")));
                let fc = Table::read(&dir.join(format!("{tag}_relevel_fit_cooks.csv")));
                check_fit(&format!("{label} relevel"), f, &ft, &fc, &mut worst);
            }
            let sp = dir.join(format!("{tag}_shrunk.csv"));
            if sp.exists() {
                let st = Table::read(&sp);
                assert_eq!(st.str("id"), ids.as_slice(), "{label}: shrunk ids");
                let (l, s) = cd.shrunk.as_ref().expect("shrunk");
                worst_shrink = worst_shrink.max(assert_close(
                    &format!("{label} shrunk lfc"),
                    l,
                    &st.f64("log2FoldChange"),
                    TOL,
                ));
                worst_shrink = worst_shrink.max(assert_close(
                    &format!("{label} shrunk se"),
                    s,
                    &st.f64("lfcSE"),
                    TOL,
                ));
                n_shrunk += 1;
            }
            if let Some(ns) = &cd.normal {
                let want = &scalars[format!("{tag}_beta_prior_var")];
                let (_, mm) = cd.design.model_matrix();
                for (k, name) in mm.iter().enumerate() {
                    let key = if name == "(Intercept)" {
                        "Intercept"
                    } else {
                        name.as_str()
                    };
                    worst_shrink = worst_shrink.max(assert_close(
                        &format!("{label} beta_prior_var {key}"),
                        &[ns.beta_prior_var[k]],
                        &[want[key]
                            .as_f64()
                            .unwrap_or_else(|| panic!("{label}: no prior var {key}"))],
                        TOL,
                    ));
                }
            }
            n_cmp += 1;
        }
        eprintln!(
            "{run}: ok ({} comparisons, shrink {})",
            diag.comparisons.len(),
            inp.shrink
        );
    }
    eprintln!(
        "results: {} runs, {n_cmp} tables, max rel {worst:e}; normal shrink: {n_shrunk} comparisons, max rel {worst_shrink:e}",
        runs.len()
    );
}

/// R's `as.character` of a double, NA as "".
fn r_character(x: f64) -> String {
    if x.is_nan() {
        return String::new();
    }
    deseq2_core::design::r_num_string(x)
}

#[test]
fn engine_matches_reference_output() {
    let runs: Vec<String> = deseq2_runs(Some("reference_output.csv"));
    assert!(!runs.is_empty());
    let mut worst = 0.0f64;
    let mut by_shrink: std::collections::BTreeMap<String, f64> = Default::default();
    for run in &runs {
        let inp = engine_input(run);
        let (out, _) = run_deseq2_diag(&inp).unwrap_or_else(|e| panic!("{run}: {e}"));
        let ref_out = Table::read(&reference_dir(run).join("reference_output.csv"));
        let pos: std::collections::HashMap<&str, usize> = out
            .gene_ids
            .iter()
            .enumerate()
            .map(|(i, g)| (g.as_str(), i))
            .collect();
        let order: Vec<usize> = ref_out
            .str("GroupId")
            .iter()
            .map(|g| {
                *pos.get(g.as_str())
                    .unwrap_or_else(|| panic!("{run}: unknown GroupId {g}"))
            })
            .collect();
        assert_eq!(order.len(), out.gene_ids.len(), "{run}: row count");
        let pick = |v: &[f64]| order.iter().map(|&i| v[i]).collect::<Vec<f64>>();
        let cols = reference_json(run)["self_check"]["columns_compared"].clone();
        let mut run_worst = 0.0f64;
        let (max_pair, max_fc) = match &out.anova {
            Some(a) => {
                let l: Vec<Vec<f64>> = a.log2fc.iter().map(|(_, v)| v.clone()).collect();
                deseq2_core::engine::max_abs_log2fc(&l)
            }
            None => (vec![], vec![]),
        };
        for c in cols.as_array().unwrap() {
            let c = c.as_str().unwrap();
            let label = format!("{run} {c}");
            let got: Vec<f64> = if let Some(a) = &out.anova {
                match c {
                    "AveExpr" => pick(&out.ave_expr),
                    "PValue" => pick(&a.pvalue),
                    "AdjPValue" => pick(&a.adj_pvalue),
                    "LRT" => pick(&a.lrt),
                    "MaxLog2FC" => pick(&max_fc),
                    "MaxLog2FCPair" => {
                        let got: Vec<String> = order
                            .iter()
                            .map(|&i| max_pair[i].map_or(String::new(), |k| a.log2fc[k].0.clone()))
                            .collect();
                        assert_eq!(got.as_slice(), ref_out.str(c), "{label}");
                        continue;
                    }
                    _ => panic!("{label}: unexpected ANOVA column"),
                }
            } else if c == "AveExpr" {
                pick(&out.ave_expr)
            } else {
                let (stat, lab) = c.split_once(' ').unwrap();
                let p = out
                    .pairs
                    .iter()
                    .find(|p| p.label == lab)
                    .unwrap_or_else(|| panic!("{label}: no pair"));
                pick(match stat {
                    "Log2FC" => &p.log2fc,
                    "stat" => &p.stat,
                    "SE" => &p.se,
                    "CILeft" => &p.ci_left,
                    "CIRight" => &p.ci_right,
                    "CrILeft" => &p.cri_left,
                    "CrIRight" => &p.cri_right,
                    "PValue" => &p.pvalue,
                    "AdjPValue" => &p.adj_pvalue,
                    _ => panic!("{label}: unexpected column"),
                })
            };
            if out.anova.is_some() {
                // The ANOVA table is character: compare the 15-digit strings' values.
                let s: Vec<String> = got.iter().map(|v| r_character(*v)).collect();
                let back: Vec<f64> = s.iter().map(|v| parse_f64(v)).collect();
                run_worst = run_worst.max(assert_close(&label, &back, &ref_out.f64(c), TOL));
            } else {
                run_worst = run_worst.max(assert_close(&label, &got, &ref_out.f64(c), TOL));
            }
        }
        let e = by_shrink.entry(inp.shrink.clone()).or_default();
        *e = e.max(run_worst);
        worst = worst.max(run_worst);
        eprintln!(
            "{run}: ok ({} columns, max rel {run_worst:e})",
            cols.as_array().unwrap().len()
        );
    }
    eprintln!(
        "e2e: {} runs, max rel {worst:e}; by shrinkage {by_shrink:?}",
        runs.len()
    );
}

#[test]
fn expected_errors_match_reference() {
    let mut n = 0;
    for run in deseq2_runs(None) {
        if !is_expected_error(&run) {
            continue;
        }
        let expected = reference_json(&run)["self_check"]["expected"]
            .as_str()
            .unwrap()
            .to_string();
        if !reference_dir(&run).join("input_counts.csv").exists() {
            // edge_deseq2_non_integer stops in prepare_inputs before the engine; covered by
            // the engine's own non-integer guard below.
            eprintln!("{run}: no engine inputs (expected '{expected}' upstream)");
            continue;
        }
        let e = run_deseq2_diag(&engine_input(&run))
            .err()
            .unwrap_or_else(|| panic!("{run}: no error"));
        assert!(
            e.contains(&expected),
            "{run}: got '{e}', expected '{expected}'"
        );
        eprintln!("{run}: ok ({expected})");
        n += 1;
    }
    assert_eq!(n, 3);
    // The non-integer guard (MDFlexi .buildCountMatrixFromLongDT).
    let mut inp = engine_input("count_deseq2_airway_all_ctlfactor");
    inp.counts[0] += 0.5;
    let e = run_deseq2_diag(&inp).err().unwrap();
    assert!(e.contains("Non-integer values detected"), "{e}");
}
