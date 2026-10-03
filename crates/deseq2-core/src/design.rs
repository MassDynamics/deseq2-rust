//! The design side of DESeq2: factors with R's level order, `model.matrix` with an intercept,
//! DESeq2's coefficient names (`renameModelMatrixColumns`), `make.names`, and the cell helpers
//! `nOrMoreInCell` / `modelMatrixGroups` that decide Cook's samples, outlier replacement and the
//! linear-mu shortcut.

use crate::la::Mat;
use std::collections::HashMap;

/// A treatment-coded factor. `levels[0]` is the reference level.
#[derive(Clone, Debug)]
pub struct Factor {
    /// Variable name (the `colData` column).
    pub name: String,
    /// Levels in R order (byte order unless relevelled).
    pub levels: Vec<String>,
    /// Level index per sample.
    pub codes: Vec<usize>,
}

impl Factor {
    /// `factor(values)`: levels sorted byte-wise (C collation, as in the production image).
    pub fn new(name: &str, values: &[String]) -> Factor {
        let mut levels = values.to_vec();
        levels.sort();
        levels.dedup();
        let codes = values
            .iter()
            .map(|v| levels.iter().position(|l| l == v).unwrap())
            .collect();
        Factor {
            name: name.to_string(),
            levels,
            codes,
        }
    }

    /// `relevel(f, ref)`: `reference` first, the other levels in their current order.
    pub fn relevel(&self, reference: &str) -> Option<Factor> {
        let r = self.levels.iter().position(|l| l == reference)?;
        let mut order = vec![r];
        order.extend((0..self.levels.len()).filter(|&i| i != r));
        let levels = order.iter().map(|&i| self.levels[i].clone()).collect();
        let codes = self
            .codes
            .iter()
            .map(|c| order.iter().position(|o| o == c).unwrap())
            .collect();
        Some(Factor {
            name: self.name.clone(),
            levels,
            codes,
        })
    }

    /// The level of each sample.
    pub fn values(&self) -> Vec<&str> {
        self.codes
            .iter()
            .map(|&c| self.levels[c].as_str())
            .collect()
    }
}

/// One term of the design formula.
#[derive(Clone, Debug)]
pub enum Var {
    /// A factor (treatment coded against its first level).
    Factor(Factor),
    /// A numeric covariate (one column, as is).
    Numeric {
        /// Variable name.
        name: String,
        /// Value per sample.
        values: Vec<f64>,
    },
}

impl Var {
    /// The variable's name.
    pub fn name(&self) -> &str {
        match self {
            Var::Factor(f) => &f.name,
            Var::Numeric { name, .. } => name,
        }
    }
}

/// A design `~ v1 + v2 + ...` with an intercept (`~ 1` when `vars` is empty).
#[derive(Clone, Debug)]
pub struct Design {
    /// Number of samples.
    pub n: usize,
    /// Terms in formula order.
    pub vars: Vec<Var>,
}

impl Design {
    /// `stats::model.matrix(~ vars)`: intercept, then each term's columns. Returns the matrix
    /// and R's column names (`(Intercept)`, `<var><level>`, `<var>`).
    pub fn model_matrix(&self) -> (Mat, Vec<String>) {
        let mut data = vec![1.0; self.n];
        let mut names = vec!["(Intercept)".to_string()];
        for v in &self.vars {
            match v {
                Var::Factor(f) => {
                    for (k, l) in f.levels.iter().enumerate().skip(1) {
                        data.extend(f.codes.iter().map(|&c| if c == k { 1.0 } else { 0.0 }));
                        names.push(format!("{}{}", f.name, l));
                    }
                }
                Var::Numeric { name, values } => {
                    data.extend_from_slice(values);
                    names.push(name.clone());
                }
            }
        }
        let p = names.len();
        (Mat::from_col_major(self.n, p, data), names)
    }

    /// The coefficient names `fitNbinomGLMs` gives the model matrix columns:
    /// `(Intercept)` becomes `Intercept`, every name goes through `make.names`, and factor
    /// columns are renamed `<var>_<level>_vs_<ref>` (also through `make.names`).
    pub fn coef_names(&self) -> Vec<String> {
        let (_, raw) = self.model_matrix();
        let mut names: Vec<String> = raw
            .iter()
            .map(|n| {
                if n == "(Intercept)" {
                    "Intercept".to_string()
                } else {
                    make_name(n)
                }
            })
            .collect();
        for v in &self.vars {
            if let Var::Factor(f) = v {
                for l in f.levels.iter().skip(1) {
                    let from = make_name(&format!("{}{}", f.name, l));
                    let to = make_name(&format!("{}_{}_vs_{}", f.name, l, f.levels[0]));
                    if let Some(i) = names.iter().position(|n| *n == from) {
                        names[i] = to;
                    }
                }
            }
        }
        names
    }

    /// The model matrix with DESeq2's coefficient names.
    pub fn deseq_matrix(&self) -> (Mat, Vec<String>) {
        (self.model_matrix().0, self.coef_names())
    }

    /// The design without its first term (the LRT reduced model `~ controls`, or `~ 1`).
    pub fn drop_first(&self) -> Design {
        Design {
            n: self.n,
            vars: self.vars[1..].to_vec(),
        }
    }

    /// `design == ~ 1`.
    pub fn is_intercept_only(&self) -> bool {
        self.vars.is_empty()
    }

    /// The single design variable when it is a two-level factor (the Cook's rescue condition).
    pub fn single_two_level_factor(&self) -> bool {
        matches!(self.vars.as_slice(), [Var::Factor(f)] if f.levels.len() == 2)
    }
}

/// R's `make.names` for one name (UTF-8 locale: any Unicode letter or digit is valid).
pub fn make_name(s: &str) -> String {
    let mut out: String = s
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '.' || c == '_' {
                c
            } else {
                '.'
            }
        })
        .collect();
    let mut it = out.chars();
    let first = it.next();
    let second = it.next();
    let needs_x = match first {
        None => true,
        Some(c) if c.is_alphabetic() => false,
        Some('.') => second.is_some_and(|d| d.is_ascii_digit()),
        Some(_) => true,
    };
    if needs_x {
        out.insert(0, 'X');
    }
    const RESERVED: [&str; 19] = [
        "if",
        "else",
        "repeat",
        "while",
        "function",
        "for",
        "next",
        "break",
        "TRUE",
        "FALSE",
        "NULL",
        "Inf",
        "NaN",
        "NA",
        "NA_integer_",
        "NA_real_",
        "NA_character_",
        "NA_complex_",
        "in",
    ];
    if RESERVED.contains(&out.as_str()) {
        out.push('.');
    }
    out
}

/// Row keys of a model matrix: equal rows share a key (`paste0(row, collapse = "_")`, so
/// values are compared at 15 significant digits and -0 equals 0, as R does).
fn row_keys(x: &Mat) -> Vec<String> {
    (0..x.nrow)
        .map(|i| {
            (0..x.ncol)
                .map(|j| r_num_string(x.at(i, j)))
                .collect::<Vec<_>>()
                .join("_")
        })
        .collect()
}

/// `nOrMoreInCell(modelMatrix, n)`: for each sample, whether at least `n` samples share its
/// model matrix row.
pub fn n_or_more_in_cell(x: &Mat, n: usize) -> Vec<bool> {
    let keys = row_keys(x);
    let mut count: HashMap<&String, usize> = HashMap::new();
    for k in &keys {
        *count.entry(k).or_insert(0) += 1;
    }
    keys.iter().map(|k| count[k] >= n).collect()
}

/// `nlevels(modelMatrixGroups(x))`: the number of distinct model matrix rows.
pub fn n_groups(x: &Mat) -> usize {
    let mut keys = row_keys(x);
    keys.sort();
    keys.dedup();
    keys.len()
}

/// R's `as.character` of a double as `paste0` uses it: the fewest significant digits (up to 15)
/// that reproduce the value at 15, fixed or scientific by width (review deseq2 r2, M-1), so 1e5 is
/// `"1e+05"`, not C's `"100000"`.
pub fn r_num_string(x: f64) -> String {
    rnum::rformat::r_as_character(x)
}

/// C's `%.<sig>g` for a finite non-zero double.
pub fn c_fmt_g(x: f64, sig: usize) -> String {
    let e = format!("{:.*e}", sig - 1, x);
    let (mant, exp) = e.split_once('e').unwrap();
    let exp: i32 = exp.parse().unwrap();
    if exp < -4 || exp >= sig as i32 {
        let mant = trim_zeros(mant);
        let sign = if exp < 0 { '-' } else { '+' };
        format!("{mant}e{sign}{:02}", exp.abs())
    } else {
        let decimals = (sig as i32 - 1 - exp).max(0) as usize;
        trim_zeros(&format!("{:.*}", decimals, x))
    }
}

fn trim_zeros(s: &str) -> String {
    if s.contains('.') {
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    } else {
        s.to_string()
    }
}

/// The robust method-of-moments cells: model matrix rows pasted with no separator, numbered in
/// order of first appearance (`factor(cells, levels = unique(cells))`).
pub fn moment_cells(x: &Mat) -> Vec<usize> {
    let strs: Vec<String> = (0..x.nrow)
        .map(|i| (0..x.ncol).map(|j| r_num_string(x.at(i, j))).collect())
        .collect();
    let mut uniq: Vec<&String> = Vec::new();
    strs.iter()
        .map(|s| match uniq.iter().position(|u| *u == s) {
            Some(k) => k,
            None => {
                uniq.push(s);
                uniq.len() - 1
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Review r1, D-9: R keys model matrix rows by `as.character` at 15 digits, so -0 and
    /// 0.1 + 0.2 share cells with 0 and 0.3; exact float bits split them.
    #[test]
    fn cells_use_the_fifteen_digit_key() {
        let col = |v: &[f64]| {
            let mut d = vec![1.0; v.len()];
            d.extend_from_slice(v);
            Mat::from_col_major(v.len(), 2, d)
        };
        let x = col(&[0.0, -0.0, 0.0, 0.3, 0.1 + 0.2, 0.3]);
        assert_eq!(n_or_more_in_cell(&x, 3), vec![true; 6]);
        assert_eq!(n_groups(&x), 2);
        // A real difference at the 15th digit still splits.
        let y = col(&[0.0, 0.0, 1e-15, 0.3, 0.3, 0.3]);
        assert_eq!(n_groups(&y), 3);
    }

    #[test]
    fn c_fmt_g_matches_printf() {
        assert_eq!(c_fmt_g(1.163_640_000_1e-16, 6), "1.16364e-16");
        assert_eq!(c_fmt_g(0.000_123_4, 6), "0.0001234");
        assert_eq!(c_fmt_g(1e-5, 6), "1e-05");
        assert_eq!(c_fmt_g(123_456_789.0, 6), "1.23457e+08");
        assert_eq!(r_num_string(1e15), "1e+15");
        assert_eq!(r_num_string(-0.0), "0");
    }

    #[test]
    fn make_names_matches_r() {
        assert_eq!(make_name("condition_a b"), "condition_a.b");
        assert_eq!(make_name("1x"), "X1x");
        assert_eq!(make_name(".5"), "X.5");
        assert_eq!(make_name(""), "X");
        assert_eq!(make_name("if"), "if.");
        assert_eq!(make_name("a-b"), "a.b");
    }

    #[test]
    fn num_strings() {
        assert_eq!(r_num_string(1.0), "1");
        assert_eq!(r_num_string(0.1), "0.1");
        assert_eq!(r_num_string(2.5), "2.5");
        assert_eq!(r_num_string(1.0 / 3.0), "0.333333333333333");
        assert_eq!(r_num_string(1e-20), "1e-20");
        // R 4.5.0: as.character(1e5) is "1e+05" (C's %.15g gives "100000"), 110000 stays fixed.
        assert_eq!(r_num_string(1e5), "1e+05");
        assert_eq!(r_num_string(110000.0), "110000");
        assert_eq!(r_num_string(1e-4), "1e-04");
        assert_eq!(r_num_string(f64::NAN), "NaN");
    }

    #[test]
    fn moment_cells_do_not_collide_on_r_strings() {
        // Review deseq2 r2, M-1 (probe d9_sep_collision): R pastes (1,0,1,1e5) as "1011e+05" and
        // (1,0,110000,0) as "101100000"; with %.15g both were "101100000", one Cook's cell.
        let x = Mat::from_col_major(2, 4, vec![1.0, 1.0, 0.0, 0.0, 1.0, 110000.0, 1e5, 0.0]);
        assert_eq!(moment_cells(&x), vec![0, 1]);
    }
}
