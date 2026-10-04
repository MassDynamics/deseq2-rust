//! Golden-corpus helpers shared by the shrink-core integration tests.
#![allow(dead_code)]

use std::collections::{BTreeSet, HashMap};
use std::path::PathBuf;

/// `MD_COUNT_CORPUS_DIR`, default `~/wd/md-count-golden-corpus`.
pub fn corpus_dir() -> PathBuf {
    if let Ok(d) = std::env::var("MD_COUNT_CORPUS_DIR") {
        return PathBuf::from(d);
    }
    let home = std::env::var("HOME").expect("HOME");
    PathBuf::from(home).join("wd/md-count-golden-corpus")
}

/// Whether the corpus is the committed CI tier (`scripts/build_small_corpus.py`).
pub fn is_small_tier() -> bool {
    let small = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/corpus-small");
    match (corpus_dir().canonicalize(), small.canonicalize()) {
        (Ok(a), Ok(b)) => a == b,
        _ => false,
    }
}

pub fn shrink_dir() -> Option<PathBuf> {
    let d = corpus_dir().join("reference-shrink");
    if d.is_dir() {
        Some(d)
    } else {
        eprintln!("SKIP: {} not found (set MD_COUNT_CORPUS_DIR)", d.display());
        None
    }
}

/// Runs under `reference-shrink/` whose id contains `pattern`, sorted.
pub fn runs(pattern: &str) -> Vec<PathBuf> {
    let Some(d) = shrink_dir() else { return vec![] };
    let mut v: Vec<PathBuf> = std::fs::read_dir(&d)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_dir() && p.file_name().unwrap().to_string_lossy().contains(pattern))
        .collect();
    v.sort();
    v
}

/// Comparison tags (`shrink_cmp_01`, ...) present for `kind` (`ashr` / `apeglm`).
pub fn cmps(run: &PathBuf, kind: &str) -> Vec<String> {
    let mut v: Vec<String> = std::fs::read_dir(run)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter_map(|f| {
            f.strip_suffix(&format!("_{kind}_final.csv"))
                .map(|s| s.to_string())
        })
        .collect();
    v.sort();
    v
}

/// What `reference-shrink/index.json` lists for runs whose id contains `pattern`: the run ids,
/// and the `(run id, comparison tag)` pairs that have a `kind` final table. The tests compare
/// both sets with what they walked on disk, so a run or comparison file that goes missing fails,
/// and the floors (5 runs, 11 comparisons, the corpus as of review deseq2 r3; the small tier
/// holds exactly 1 and 3) stop a coordinated removal from the index and the disk passing quietly
/// (review overnight r1, SE-m3 and SE-n1).
pub struct Index {
    pub runs: BTreeSet<String>,
    pub cmps: BTreeSet<(String, String)>,
}

pub fn index(pattern: &str, kind: &str) -> Index {
    let path = corpus_dir().join("reference-shrink/index.json");
    let s = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let v: serde_json::Value = serde_json::from_str(&s).unwrap();
    let suffix = format!("_{kind}_final.csv");
    let mut idx = Index {
        runs: BTreeSet::new(),
        cmps: BTreeSet::new(),
    };
    for r in v["runs"].as_array().unwrap() {
        let id = r["run_id"].as_str().unwrap();
        if !id.contains(pattern) {
            continue;
        }
        idx.runs.insert(id.to_string());
        for f in r["files"].as_object().unwrap().keys() {
            if let Some(tag) = f.strip_suffix(&suffix) {
                idx.cmps.insert((id.to_string(), tag.to_string()));
            }
        }
    }
    if is_small_tier() {
        assert!(
            idx.runs.len() == 1 && idx.cmps.len() == 3,
            "small tier index.json lists {} {pattern} runs and {} {kind} comparisons, not 1 and 3",
            idx.runs.len(),
            idx.cmps.len()
        );
    } else {
        assert!(
            idx.runs.len() >= 5 && idx.cmps.len() >= 11,
            "index.json lists {} {pattern} runs and {} {kind} comparisons, below the floor of 5 and 11",
            idx.runs.len(),
            idx.cmps.len()
        );
    }
    idx
}

/// A run's id, its directory name.
pub fn name_of(run: &std::path::Path) -> String {
    run.file_name().unwrap().to_string_lossy().into_owned()
}

/// The run ids of `runs`, as a set to compare with `Index::runs`.
pub fn run_ids(runs: &[PathBuf]) -> BTreeSet<String> {
    runs.iter().map(|r| name_of(r)).collect()
}

/// A CSV as named string columns.
pub struct Table {
    pub header: Vec<String>,
    pub cols: HashMap<String, Vec<String>>,
    pub nrow: usize,
}

impl Table {
    pub fn read(path: &PathBuf) -> Table {
        let mut rdr = csv::ReaderBuilder::new()
            .has_headers(true)
            .from_path(path)
            .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        let header: Vec<String> = rdr
            .headers()
            .unwrap()
            .iter()
            .map(|s| s.to_string())
            .collect();
        let mut cols: HashMap<String, Vec<String>> =
            header.iter().map(|h| (h.clone(), Vec::new())).collect();
        let mut nrow = 0;
        for rec in rdr.records() {
            let rec = rec.unwrap();
            for (h, v) in header.iter().zip(rec.iter()) {
                cols.get_mut(h).unwrap().push(v.to_string());
            }
            nrow += 1;
        }
        Table { header, cols, nrow }
    }
    pub fn f(&self, name: &str) -> Vec<f64> {
        self.cols
            .get(name)
            .unwrap_or_else(|| panic!("no column {name}"))
            .iter()
            .map(|s| parse_f64(s))
            .collect()
    }
    pub fn b(&self, name: &str) -> Vec<bool> {
        self.cols[name].iter().map(|s| s == "TRUE").collect()
    }
    pub fn s(&self, name: &str) -> Vec<String> {
        self.cols[name].clone()
    }
    /// Columns whose names start with `prefix` followed by digits, in numeric order.
    pub fn numbered(&self, prefix: &str) -> Vec<Vec<f64>> {
        let mut names: Vec<(usize, String)> = self
            .header
            .iter()
            .filter_map(|h| {
                h.strip_prefix(prefix)
                    .and_then(|r| r.parse::<usize>().ok())
                    .map(|k| (k, h.clone()))
            })
            .collect();
        names.sort();
        names.into_iter().map(|(_, h)| self.f(&h)).collect()
    }
}

pub fn parse_f64(s: &str) -> f64 {
    match s {
        "NA" | "" | "NaN" => f64::NAN,
        "Inf" => f64::INFINITY,
        "-Inf" => f64::NEG_INFINITY,
        _ => s.parse().unwrap_or_else(|_| panic!("bad number {s}")),
    }
}

/// Max relative gap `|a - b| / |b|` (abs gap where `b == 0`); NaN pairs must match.
pub fn max_rel(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "length mismatch");
    let mut m: f64 = 0.0;
    for (x, y) in a.iter().zip(b) {
        if x.is_nan() || y.is_nan() {
            assert!(x.is_nan() && y.is_nan(), "NaN mismatch {x} vs {y}");
            continue;
        }
        if x == y {
            continue;
        }
        let d = (x - y).abs();
        let r = if *y == 0.0 { d } else { d / y.abs() };
        m = m.max(r);
    }
    m
}

pub fn max_abs(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}
