//! The production DESeq2 engine (`MDFlexiComparisons` `R/deseq2StatsFun.R`, as the corpus
//! reference `ref_deseq2.R` replays it): input guards, `filterByExpr`, `checkMatrixRank`,
//! `DESeq()` (Wald, or LRT against `~ controls` for ANOVA), then per comparison the relevel
//! refit, `results(name = ...)` with independent filtering, the Wald CI and the optional
//! `lfcShrink` (normal, ashr or apeglm), or for ANOVA the per-pair contrasts and the LRT
//! omnibus. Every output vector covers all input genes in input order (NA where
//! `filterByExpr` dropped the gene); the left join to `GroupId` order and the ANOVA string
//! formatting are left to the caller.

use crate::design::{Design, Factor, Var};
use crate::fit::{deseq, DeseqFit};
use crate::nbtest::{nbinom_test, TestFit};
use crate::results::{results, ResultsData, ResultsTable, Which};
use crate::shrink::{shrink_normal, NormalShrink};

/// One control column, one value per sample (in sample order).
#[derive(Clone, Debug)]
pub struct Control {
    /// Column name.
    pub name: String,
    /// `numerical` (one column, parsed as a number) rather than `categorical` (a factor).
    pub numeric: bool,
    /// The values as strings.
    pub values: Vec<String>,
}

/// One comparison `left - right`: `left` / `right` label the output columns, `encoded_*` are
/// the condition values as they appear in the sample info.
#[derive(Clone, Debug)]
pub struct Comparison {
    /// Output label of the numerator.
    pub left: String,
    /// Output label of the denominator.
    pub right: String,
    /// Numerator condition value.
    pub encoded_left: String,
    /// Denominator condition value.
    pub encoded_right: String,
}

/// Engine inputs. `counts` is gene-major (`gene_ids.len() x sample_ids.len()`).
#[derive(Clone, Debug)]
pub struct DeseqInput {
    /// Gene ids.
    pub gene_ids: Vec<String>,
    /// Sample ids (column order of `counts`).
    pub sample_ids: Vec<String>,
    /// Raw counts.
    pub counts: Vec<f64>,
    /// Condition column name.
    pub condition_col: String,
    /// Condition value per sample.
    pub condition: Vec<String>,
    /// Control columns.
    pub controls: Vec<Control>,
    /// Comparisons.
    pub comparisons: Vec<Comparison>,
    /// `mode == "anova"`: LRT omnibus plus per-pair contrasts.
    pub anova: bool,
    /// `deseq2_alpha` (independent-filtering target FDR; default 0.05).
    pub alpha: f64,
    /// `deseq2_lfc_shrinkage`: `none`, `normal`, `ashr` or `apeglm`.
    pub shrink: String,
    /// Must be `gene`.
    pub entity_type: String,
}

/// Per-comparison pairwise columns, each over every input gene.
#[derive(Clone, Debug)]
pub struct PairColumns {
    /// `"{left} - {right}"`.
    pub label: String,
    /// `Log2FC` (shrunk when shrinkage is on).
    pub log2fc: Vec<f64>,
    /// `stat` (unshrunk Wald statistic).
    pub stat: Vec<f64>,
    /// `SE` (shrunk when shrinkage is on).
    pub se: Vec<f64>,
    /// Wald CI from the unshrunk fit.
    pub ci_left: Vec<f64>,
    /// Wald CI from the unshrunk fit.
    pub ci_right: Vec<f64>,
    /// Credible interval (NA without shrinkage).
    pub cri_left: Vec<f64>,
    /// Credible interval (NA without shrinkage).
    pub cri_right: Vec<f64>,
    /// `PValue`.
    pub pvalue: Vec<f64>,
    /// `AdjPValue`.
    pub adj_pvalue: Vec<f64>,
}

/// The ANOVA columns, each over every input gene.
#[derive(Clone, Debug)]
pub struct AnovaColumns {
    /// `("{left} - {right}", Log2FC)` per comparison.
    pub log2fc: Vec<(String, Vec<f64>)>,
    /// LRT statistic.
    pub lrt: Vec<f64>,
    /// Omnibus p-value.
    pub pvalue: Vec<f64>,
    /// Omnibus adjusted p-value.
    pub adj_pvalue: Vec<f64>,
}

/// Engine output.
#[derive(Clone, Debug)]
pub struct DeseqOutput {
    /// Input gene ids, input order.
    pub gene_ids: Vec<String>,
    /// `filterByExpr` keep flags.
    pub kept: Vec<bool>,
    /// `AveExpr` (`baseMean`).
    pub ave_expr: Vec<f64>,
    /// Pairwise columns (empty for ANOVA).
    pub pairs: Vec<PairColumns>,
    /// ANOVA columns.
    pub anova: Option<AnovaColumns>,
}

/// Shrunk log2FC, SE and the credible-interval bounds (left, right) of one comparison.
type ShrunkCols = (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>);

/// One pairwise comparison's intermediates (over the kept genes).
#[derive(Clone, Debug)]
pub struct ComparisonDiag {
    /// The relevelled design.
    pub design: Design,
    /// The Wald refit after `relevel(ref = right)`, when the reference changed.
    pub relevel_fit: Option<TestFit>,
    /// `results(name = coefName, independentFiltering = TRUE, alpha)`.
    pub results: ResultsTable,
    /// The normal-prior fit, for `normal` shrinkage.
    pub normal: Option<NormalShrink>,
    /// The normal-shrinkage `results()` table.
    pub normal_results: Option<ResultsTable>,
    /// Shrunk `log2FoldChange` / `lfcSE` (any shrinkage type).
    pub shrunk: Option<(Vec<f64>, Vec<f64>)>,
    /// apeglm: rows whose MAP fit did not converge (`fit$diag[, "conv"]` not 0). DESeq2 and
    /// production drop this silently; it is surfaced here as a count (review r1, stats item 7).
    pub shrink_nonconverged: Option<usize>,
}

/// The engine's intermediates.
#[derive(Clone, Debug)]
pub struct DeseqDiag {
    /// Indices of the genes `filterByExpr` kept.
    pub kept_idx: Vec<usize>,
    /// The `DESeq()` fit over the kept genes.
    pub fit: DeseqFit,
    /// Pairwise comparisons.
    pub comparisons: Vec<ComparisonDiag>,
    /// ANOVA per-pair contrast tables.
    pub anova_lfc: Vec<ResultsTable>,
    /// ANOVA omnibus table.
    pub omnibus: Option<ResultsTable>,
}

fn spread(v: &[f64], kept_idx: &[usize], n: usize) -> Vec<f64> {
    let mut out = vec![f64::NAN; n];
    for (k, &g) in kept_idx.iter().enumerate() {
        out[g] = v[k];
    }
    out
}

/// Build `~ condition + controls`.
pub fn build_design(input: &DeseqInput) -> Result<Design, String> {
    let m = input.sample_ids.len();
    let mut vars = vec![Var::Factor(Factor::new(
        &input.condition_col,
        &input.condition,
    ))];
    for c in &input.controls {
        if c.values.len() != m {
            return Err(format!(
                "control column '{}' has {} values for {m} samples",
                c.name,
                c.values.len()
            ));
        }
        vars.push(if c.numeric {
            let values = c
                .values
                .iter()
                .map(|v| {
                    v.trim().parse::<f64>().map_err(|_| {
                        format!(
                            "control column '{}' is numerical but has value '{v}'",
                            c.name
                        )
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            Var::Numeric {
                name: c.name.clone(),
                values,
            }
        } else {
            Var::Factor(Factor::new(&c.name, &c.values))
        });
    }
    Ok(Design { n: m, vars })
}

/// Run the engine. Errors carry the production messages (without the `md_error` markers).
pub fn run_deseq2(input: &DeseqInput) -> Result<DeseqOutput, String> {
    run_deseq2_diag(input).map(|(o, _)| o)
}

/// [`run_deseq2`], also returning the intermediates.
pub fn run_deseq2_diag(input: &DeseqInput) -> Result<(DeseqOutput, DeseqDiag), String> {
    let ng = input.gene_ids.len();
    let m = input.sample_ids.len();
    if input.entity_type != "gene" {
        return Err(if input.anova {
            "de_method 'DESeq2' is only supported for gene entity type (count data). Use de_method = 'limma' for protein or peptide data."
        } else {
            "de_method 'DESeq2' is only supported for gene entity type (count data). Use de_method = 'limma' for protein, peptide, metabolite, or PTM data."
        }
        .into());
    }
    if input.counts.len() != ng * m || input.condition.len() != m {
        return Err("counts, gene ids, sample ids and condition values disagree in size".into());
    }
    if input.counts.iter().any(|v| *v < 0.0) {
        return Err("The data contains negative intensities. Please check if your data was log-transformed before starting the analysis.".into());
    }
    let nonfinite = input.counts.iter().filter(|v| !v.is_finite()).count();
    if nonfinite > 0 {
        return Err(format!("Count matrix contains {nonfinite} non-finite (Inf / -Inf / NaN) values. This indicates an upstream data-integrity bug; refusing to silently coerce to zero."));
    }
    if input.counts.iter().any(|v| (v - v.round()).abs() > 1e-6) {
        let looks_cpm = (0..m).all(|j| {
            let s: f64 =
                crate::ext::sum(&(0..ng).map(|g| input.counts[g * m + j]).collect::<Vec<_>>());
            (s - 1e6).abs() < 1e3
        });
        return Err(if looks_cpm {
            "Input data appears to be CPM/TPM-normalised (non-integer values, per-sample sums ~1e6). edgeR and DESeq2 require raw integer counts. Use de_method = 'limma' for pre-normalised data, or re-upload raw counts."
        } else {
            "Non-integer values detected in the count column. edgeR and DESeq2 require raw integer counts as input. Use de_method = 'limma' for pre-normalised or continuous data."
        }
        .into());
    }
    let counts: Vec<f64> = input.counts.iter().map(|v| v.round()).collect();
    let design = build_design(input)?;
    let cond = match &design.vars[0] {
        Var::Factor(f) => f.clone(),
        _ => unreachable!(),
    };
    // model.matrix refuses any single-level factor, the condition or a categorical control.
    if design
        .vars
        .iter()
        .any(|v| matches!(v, Var::Factor(f) if f.levels.len() < 2))
    {
        return Err("contrasts can be applied only to factors with 2 or more levels".into());
    }
    let (x, _) = design.model_matrix();
    let p = x.ncol;

    let fb =
        edger_core::filter::filter_by_expr(&counts, m, &x.data, p).map_err(|e| e.to_string())?;
    let kept_idx: Vec<usize> = (0..ng).filter(|&g| fb.keep[g]).collect();
    if kept_idx.is_empty() {
        return Err("filterByExpr removed every gene. Check input count matrix and sample-size per condition.".into());
    }
    if rnum::linpack::qr_decompose(&x.data, m, p, 1e-7).rank < p {
        let mut preds = vec![input.condition_col.clone()];
        preds.extend(input.controls.iter().map(|c| c.name.clone()));
        return Err(format!(
            "Model creation failed because one or more variables '{}' are perfectly collinear. Each variable should represent unique information.",
            preds.join(", ")
        ));
    }
    let mut y = Vec::with_capacity(kept_idx.len() * m);
    for &g in &kept_idx {
        y.extend_from_slice(&counts[g * m..(g + 1) * m]);
    }
    // Production's as.integer(round()) runs on the whole matrix before filterByExpr, so a count
    // above .Machine$integer.max stops DESeq2 even in a gene filterByExpr drops (review deseq2 r2,
    // M-2).
    if counts.iter().any(|v| *v > 2147483647.0) {
        return Err("NA counts not allowed".into());
    }
    let fit = deseq(&y, &design, input.anova)?;
    let ave_expr = spread(&fit.base.base_mean, &kept_idx, ng);
    let replace = fit.replacement.as_ref().map(|r| r.replace.as_slice());

    let mut diag = DeseqDiag {
        kept_idx: kept_idx.clone(),
        fit: fit.clone(),
        comparisons: vec![],
        anova_lfc: vec![],
        omnibus: None,
    };

    if input.anova {
        let data = ResultsData {
            design: &design,
            test: &fit.test,
            base_mean: &fit.base.base_mean,
            all_zero: &fit.base.all_zero,
            replace,
            counts: &fit.counts,
            fit_counts: fit.fit_counts(),
            sf: &fit.sf,
            dispersion: &fit.disp.dispersion,
        };
        let mut lfcs = Vec::new();
        for c in &input.comparisons {
            let which = Which::Contrast {
                factor: input.condition_col.clone(),
                num: c.encoded_left.clone(),
                den: c.encoded_right.clone(),
            };
            let r = results(&data, &which, false, 0.1)?;
            lfcs.push((
                format!("{} - {}", c.left, c.right),
                spread(&r.lfc, &kept_idx, ng),
            ));
            diag.anova_lfc.push(r);
        }
        let om = results(&data, &Which::Last, true, input.alpha)?;
        let anova = AnovaColumns {
            log2fc: lfcs,
            lrt: spread(&om.stat, &kept_idx, ng),
            pvalue: spread(&om.pvalue, &kept_idx, ng),
            adj_pvalue: spread(&om.padj, &kept_idx, ng),
        };
        diag.omnibus = Some(om);
        let out = DeseqOutput {
            gene_ids: input.gene_ids.clone(),
            kept: fb.keep.clone(),
            ave_expr,
            pairs: vec![],
            anova: Some(anova),
        };
        return Ok((out, diag));
    }

    let shrink = input.shrink.as_str();
    if !["none", "apeglm", "ashr", "normal"].contains(&shrink) {
        return Err(format!(
            "Invalid deseq2_lfc_shrinkage value: '{shrink}'. Must be one of: none, apeglm, ashr, normal."
        ));
    }
    let qn = rnum::nmath::qnorm(0.975, 0.0, 1.0, true, false);
    let mut pairs = Vec::new();
    for c in &input.comparisons {
        let left = &c.encoded_left;
        let right = &c.encoded_right;
        // The dispatch-contrast results() call production makes first fails on these.
        crate::results::check_alpha(input.alpha)?;
        crate::results::check_levels_differ(left, right)?;
        let (test, des, relevel_fit) = if cond.levels[0] != *right {
            let f2 = cond
                .relevel(right)
                .ok_or("'ref' must be an existing level")?;
            let mut d2 = design.clone();
            d2.vars[0] = Var::Factor(f2);
            let t = nbinom_test(
                &fit.counts,
                &fit.sf,
                &fit.base.all_zero,
                &fit.disp.dispersion,
                &d2,
                false,
            )?;
            (t.clone(), d2, Some(t))
        } else {
            (fit.test.clone(), design.clone(), None)
        };
        let coef_name = format!("{}_{}_vs_{}", input.condition_col, left, right);
        if !test.coef_names.contains(&coef_name) {
            // Name the caller's labels, never the encoded tokens (review deseq2 r2, m-6).
            return Err(format!(
                "DESeq2 results: '{}' is not a level of {} (comparison {} - {}).",
                c.left, input.condition_col, c.left, c.right
            ));
        }
        let coef_k = test.coef_index(&coef_name).unwrap();
        let data = ResultsData {
            design: &des,
            test: &test,
            base_mean: &fit.base.base_mean,
            all_zero: &fit.base.all_zero,
            replace,
            counts: &fit.counts,
            fit_counts: fit.fit_counts(),
            sf: &fit.sf,
            dispersion: &fit.disp.dispersion,
        };
        let which = Which::Name(coef_name.clone());
        let r = results(&data, &which, true, input.alpha)?;
        let nk = r.lfc.len();
        let ci_l: Vec<f64> = (0..nk).map(|g| r.lfc[g] - qn * r.se[g]).collect();
        let ci_r: Vec<f64> = (0..nk).map(|g| r.lfc[g] + qn * r.se[g]).collect();
        let mut cd = ComparisonDiag {
            design: des.clone(),
            relevel_fit,
            results: r.clone(),
            normal: None,
            normal_results: None,
            shrunk: None,
            shrink_nonconverged: None,
        };
        let shrunk: Option<ShrunkCols> = match shrink {
            "normal" => {
                let ns = shrink_normal(
                    &fit.counts,
                    &fit.sf,
                    &fit.base.all_zero,
                    &fit.disp.dispersion,
                    &fit.base.base_mean,
                    &fit.disp.fit,
                    &des,
                    &test,
                )?;
                let mut ts = test.clone();
                ts.beta = ns.beta.clone();
                ts.se = ns.se.clone();
                let ds = ResultsData { test: &ts, ..data };
                let rs = results(&ds, &which, true, input.alpha)?;
                let lo = (0..nk).map(|g| rs.lfc[g] - qn * rs.se[g]).collect();
                let hi = (0..nk).map(|g| rs.lfc[g] + qn * rs.se[g]).collect();
                let s = (rs.lfc.clone(), rs.se.clone(), lo, hi);
                cd.normal = Some(ns);
                cd.normal_results = Some(rs);
                Some(s)
            }
            "ashr" => {
                let a = shrink_core::shrink_ashr(&r.lfc, &r.se).map_err(|e| e.to_string())?;
                let t = a.table;
                Some((t.posterior_mean, t.posterior_sd, t.cri_left, t.cri_right))
            }
            "apeglm" => {
                let (xm, _) = des.model_matrix();
                let mut cm = vec![0.0; nk * m];
                for g in 0..nk {
                    for j in 0..m {
                        cm[g + j * nk] = fit.counts[g * m + j];
                    }
                }
                let a = shrink_core::shrink_apeglm(
                    &shrink_core::dense::Mat::from_col_major(nk, m, cm),
                    &fit.sf,
                    &fit.disp.dispersion,
                    &shrink_core::dense::Mat::from_col_major(xm.nrow, xm.ncol, xm.data.clone()),
                    coef_k,
                    &r.lfc,
                    &r.se,
                )
                .map_err(|e| e.to_string())?;
                cd.shrink_nonconverged = Some(count_nonconverged(&a.diag_conv));
                Some((a.log2_fold_change, a.lfc_se, a.cri_left, a.cri_right))
            }
            _ => None,
        };
        let nan = vec![f64::NAN; nk];
        let (lfc, se, cr_l, cr_r) = match shrunk {
            Some((l, s, lo, hi)) => {
                cd.shrunk = Some((l.clone(), s.clone()));
                (l, s, lo, hi)
            }
            None => (r.lfc.clone(), r.se.clone(), nan.clone(), nan),
        };
        let sp = |v: &[f64]| spread(v, &kept_idx, ng);
        pairs.push(PairColumns {
            label: format!("{} - {}", c.left, c.right),
            log2fc: sp(&lfc),
            stat: sp(&r.stat),
            se: sp(&se),
            ci_left: sp(&ci_l),
            ci_right: sp(&ci_r),
            cri_left: sp(&cr_l),
            cri_right: sp(&cr_r),
            pvalue: sp(&r.pvalue),
            adj_pvalue: sp(&r.padj),
        });
        diag.comparisons.push(cd);
    }
    let out = DeseqOutput {
        gene_ids: input.gene_ids.clone(),
        kept: fb.keep.clone(),
        ave_expr,
        pairs,
        anova: None,
    };
    Ok((out, diag))
}

/// `get_max_fc` of `.packageANOVAOutput`: per gene, the comparison with the largest |Log2FC|
/// (first on ties, NA ignored) and its Log2FC; `None` / NA when every Log2FC is NA.
pub fn max_abs_log2fc(lfcs: &[Vec<f64>]) -> (Vec<Option<usize>>, Vec<f64>) {
    let ng = lfcs.first().map_or(0, |v| v.len());
    let mut idx = vec![None; ng];
    let mut val = vec![f64::NAN; ng];
    for g in 0..ng {
        let mut best: Option<usize> = None;
        for (k, v) in lfcs.iter().enumerate() {
            if v[g].is_nan() {
                continue;
            }
            if best.is_none_or(|b| v[g].abs() > lfcs[b][g].abs()) {
                best = Some(k);
            }
        }
        if let Some(b) = best {
            idx[g] = Some(b);
            val[g] = lfcs[b][g];
        }
    }
    (idx, val)
}

/// apeglm rows whose MAP fit did not converge: `fit$diag[, "conv"]` not 0, NA (rows apeglm
/// skipped) not counted.
fn count_nonconverged(conv: &[f64]) -> usize {
    conv.iter().filter(|&&c| !c.is_nan() && c != 0.0).count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nonconverged_rows_are_counted() {
        // Review deseq2 r2, SE-m6: the end-to-end test only sees 0, which a hardcoded zero passes.
        assert_eq!(count_nonconverged(&[0.0, 1.0, f64::NAN, -1.0]), 2);
        assert_eq!(count_nonconverged(&[0.0, f64::NAN]), 0);
    }
}
