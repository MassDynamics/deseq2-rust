//! Shared helpers for reading the count golden corpus.
#![allow(dead_code)]

use std::collections::HashMap;
use std::path::PathBuf;

/// Corpus root: `MD_COUNT_CORPUS_DIR`, default `~/wd/md-count-golden-corpus`.
pub fn corpus_dir() -> PathBuf {
    if let Ok(d) = std::env::var("MD_COUNT_CORPUS_DIR") {
        return PathBuf::from(d);
    }
    let home = std::env::var("HOME").expect("HOME");
    PathBuf::from(home).join("wd/md-count-golden-corpus")
}

pub fn reference_dir(run: &str) -> PathBuf {
    corpus_dir().join("reference").join(run)
}

pub fn run_dir(run: &str) -> PathBuf {
    corpus_dir().join("runs").join(run)
}

/// A CSV read as named string columns.
pub struct Table {
    pub names: Vec<String>,
    pub cols: HashMap<String, Vec<String>>,
    pub nrow: usize,
}

impl Table {
    pub fn read(path: &std::path::Path) -> Table {
        let mut rdr = csv::ReaderBuilder::new()
            .has_headers(true)
            .from_path(path)
            .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        let names: Vec<String> = rdr
            .headers()
            .unwrap()
            .iter()
            .map(|s| s.to_string())
            .collect();
        let mut cols: HashMap<String, Vec<String>> =
            names.iter().map(|n| (n.clone(), Vec::new())).collect();
        let mut nrow = 0;
        for rec in rdr.records() {
            let rec = rec.unwrap();
            for (i, n) in names.iter().enumerate() {
                cols.get_mut(n)
                    .unwrap()
                    .push(rec.get(i).unwrap_or("").to_string());
            }
            nrow += 1;
        }
        Table { names, cols, nrow }
    }

    pub fn has(&self, name: &str) -> bool {
        self.cols.contains_key(name)
    }

    pub fn str(&self, name: &str) -> &[String] {
        self.cols
            .get(name)
            .unwrap_or_else(|| panic!("missing column {name}"))
    }

    /// Numeric column; `NA`/empty become NaN, `Inf`/`-Inf` parse.
    pub fn f64(&self, name: &str) -> Vec<f64> {
        self.str(name).iter().map(|s| parse_f64(s)).collect()
    }

    pub fn bool(&self, name: &str) -> Vec<Option<bool>> {
        self.str(name)
            .iter()
            .map(|s| match s.as_str() {
                "TRUE" | "true" => Some(true),
                "FALSE" | "false" => Some(false),
                _ => None,
            })
            .collect()
    }
}

pub fn parse_f64(s: &str) -> f64 {
    match s {
        "" | "NA" | "NaN" => f64::NAN,
        "Inf" => f64::INFINITY,
        "-Inf" => f64::NEG_INFINITY,
        _ => s.parse().unwrap_or_else(|_| panic!("bad number {s}")),
    }
}

pub fn reference_json(run: &str) -> serde_json::Value {
    let p = reference_dir(run).join("reference.json");
    serde_json::from_str(&std::fs::read_to_string(&p).unwrap()).unwrap()
}

pub fn manifest(run: &str) -> serde_json::Value {
    let p = run_dir(run).join("manifest.json");
    serde_json::from_str(&std::fs::read_to_string(&p).unwrap()).unwrap()
}

/// Relative difference with an absolute floor for values near zero.
pub fn rel_diff(a: f64, b: f64) -> f64 {
    if a.is_nan() && b.is_nan() {
        return 0.0;
    }
    if a == b {
        return 0.0;
    }
    // data.table::fwrite prints every subnormal double wrongly (R's 9 * 2^-1074 comes out as
    // 1.11253692925361e-308, checked in the image), so values below DBL_MIN compare equal.
    if a.abs() < f64::MIN_POSITIVE && b.abs() < f64::MIN_POSITIVE {
        return 0.0;
    }
    (a - b).abs() / a.abs().max(b.abs()).max(1e-300)
}

/// Assert two numeric vectors agree: identical NA pattern and relative tolerance `tol`.
/// Returns the maximum relative difference seen.
pub fn assert_close(label: &str, got: &[f64], want: &[f64], tol: f64) -> f64 {
    assert_eq!(got.len(), want.len(), "{label}: length");
    let mut worst = 0.0f64;
    let mut worst_i = 0;
    for i in 0..got.len() {
        assert_eq!(
            got[i].is_nan(),
            want[i].is_nan(),
            "{label}[{i}]: NA pattern got {} want {}",
            got[i],
            want[i]
        );
        let d = rel_diff(got[i], want[i]);
        if d > worst {
            worst = d;
            worst_i = i;
        }
    }
    assert!(
        worst <= tol,
        "{label}: max rel diff {worst:e} at {worst_i} (got {:e} want {:e})",
        got[worst_i],
        want[worst_i]
    );
    worst
}

/// DESeq2 runs with a reference directory, optionally only those holding `file`.
pub fn deseq2_runs(file: Option<&str>) -> Vec<String> {
    let mut v: Vec<String> = std::fs::read_dir(corpus_dir().join("reference"))
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().to_string())
        .filter(|n| n.contains("deseq2"))
        .filter(|n| file.is_none_or(|f| reference_dir(n).join(f).exists()))
        .collect();
    v.sort();
    v
}

/// `(column, is_numeric)` for the manifest's `control_cols`.
pub fn control_specs(run: &str) -> Vec<(String, bool)> {
    let cc = &manifest(run)["params"]["control_cols"];
    if cc.is_null() {
        return vec![];
    }
    let as_vec = |v: &serde_json::Value| -> Vec<String> {
        match v {
            serde_json::Value::Array(a) => {
                a.iter().map(|s| s.as_str().unwrap().to_string()).collect()
            }
            s => vec![s.as_str().unwrap().to_string()],
        }
    };
    as_vec(&cc["Column"])
        .into_iter()
        .zip(as_vec(&cc["Type"]))
        .map(|(c, t)| (c, t == "numerical"))
        .collect()
}

/// A DESeq2 run's fitted inputs: the genes kept by `deseq2_filter.csv`, their counts
/// (row-major, samples in `input_sample_info` order) and the full design.
pub struct DeseqRun {
    pub ids: Vec<String>,
    pub samples: Vec<String>,
    pub counts: Vec<f64>,
    pub design: deseq2_core::design::Design,
}

pub fn deseq_run(run: &str) -> DeseqRun {
    use deseq2_core::design::{Design, Factor, Var};
    let dir = reference_dir(run);
    let si = Table::read(&dir.join("input_sample_info.csv"));
    let samples = si.str("replicate").to_vec();
    let cond = manifest(run)["params"]["condition_col"]
        .as_str()
        .unwrap()
        .to_string();
    let mut vars = vec![Var::Factor(Factor::new(&cond, si.str(&cond)))];
    for (c, numeric) in control_specs(run) {
        vars.push(if numeric {
            Var::Numeric {
                name: c.clone(),
                values: si.f64(&c),
            }
        } else {
            Var::Factor(Factor::new(&c, si.str(&c)))
        });
    }
    let design = Design {
        n: samples.len(),
        vars,
    };
    let cnt = Table::read(&dir.join("input_counts.csv"));
    let flt = Table::read(&dir.join("deseq2_filter.csv"));
    let keep: std::collections::HashMap<&str, bool> = flt
        .str("id")
        .iter()
        .zip(flt.bool("keep"))
        .map(|(i, k)| (i.as_str(), k.unwrap()))
        .collect();
    let cols: Vec<Vec<f64>> = samples.iter().map(|s| cnt.f64(s)).collect();
    let mut ids = Vec::new();
    let mut counts = Vec::new();
    for (g, id) in cnt.str("id").iter().enumerate() {
        if keep[id.as_str()] {
            ids.push(id.clone());
            for c in &cols {
                counts.push(c[g]);
            }
        }
    }
    DeseqRun {
        ids,
        samples,
        counts,
        design,
    }
}

/// Exact equality of numeric vectors (NA == NA), reporting the first mismatch.
pub fn assert_exact(label: &str, got: &[f64], want: &[f64]) {
    assert_eq!(got.len(), want.len(), "{label}: length");
    for i in 0..got.len() {
        let same = (got[i].is_nan() && want[i].is_nan()) || got[i] == want[i];
        assert!(same, "{label}[{i}]: got {} want {}", got[i], want[i]);
    }
}

/// `R NA / TRUE / FALSE` as NaN / 1 / 0.
pub fn opt_f64(v: &[Option<bool>]) -> Vec<f64> {
    v.iter()
        .map(|b| b.map_or(f64::NAN, |b| b as u8 as f64))
        .collect()
}

pub fn check_fit(
    label: &str,
    f: &deseq2_core::nbtest::TestFit,
    t: &Table,
    cooks: &Table,
    worst: &mut f64,
) {
    let p = f.p();
    let mut chk = |name: &str, got: &[f64]| {
        *worst = worst.max(assert_close(
            &format!("{label} {name}"),
            got,
            &t.f64(name),
            1e-8,
        ));
    };
    for (k, c) in f.coef_names.iter().enumerate() {
        chk(c, &deseq2_core::nbtest::TestFit::column(&f.beta, p, k));
        chk(
            &format!("SE_{c}"),
            &deseq2_core::nbtest::TestFit::column(&f.se, p, k),
        );
        if f.kind == deseq2_core::nbtest::TestKind::Wald {
            chk(
                &format!("WaldStatistic_{c}"),
                &deseq2_core::nbtest::TestFit::column(&f.stat, p, k),
            );
            chk(
                &format!("WaldPvalue_{c}"),
                &deseq2_core::nbtest::TestFit::column(&f.pvalue, p, k),
            );
        }
    }
    let conv_name = if f.kind == deseq2_core::nbtest::TestKind::Wald {
        "betaConv"
    } else {
        chk("LRTStatistic", &f.stat);
        chk("LRTPvalue", &f.pvalue);
        assert_exact(
            &format!("{label} reducedBetaConv"),
            &opt_f64(&f.reduced_conv),
            &opt_f64(&t.bool("reducedBetaConv")),
        );
        "fullBetaConv"
    };
    assert_exact(
        &format!("{label} {conv_name}"),
        &opt_f64(&f.conv),
        &opt_f64(&t.bool(conv_name)),
    );
    assert_exact(&format!("{label} betaIter"), &f.iter, &t.f64("betaIter"));
    chk("deviance", &f.deviance);
    chk("maxCooks", &f.max_cooks);
    let m = f.x.nrow;
    let samples: Vec<&String> = cooks.names.iter().skip(1).collect();
    assert_eq!(samples.len(), m);
    for (j, s) in samples.iter().enumerate() {
        let got: Vec<f64> = f.cooks.iter().skip(j).step_by(m).copied().collect();
        *worst = worst.max(assert_close(
            &format!("{label} cooks {s}"),
            &got,
            &cooks.f64(s),
            1e-8,
        ));
    }
}
