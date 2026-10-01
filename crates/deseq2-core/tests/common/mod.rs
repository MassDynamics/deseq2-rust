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
        let names: Vec<String> = rdr.headers().unwrap().iter().map(|s| s.to_string()).collect();
        let mut cols: HashMap<String, Vec<String>> =
            names.iter().map(|n| (n.clone(), Vec::new())).collect();
        let mut nrow = 0;
        for rec in rdr.records() {
            let rec = rec.unwrap();
            for (i, n) in names.iter().enumerate() {
                cols.get_mut(n).unwrap().push(rec.get(i).unwrap_or("").to_string());
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
        "{label}: max rel diff {worst:e} at {worst_i} (got {} want {})",
        got[worst_i],
        want[worst_i]
    );
    worst
}
