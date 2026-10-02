//! `deseq2_rust._core`: thin PyO3 surface over `deseq2-core`. Takes a numpy count matrix and
//! plain Python lists, returns a dict of numpy arrays and lists. The production join, column
//! naming and ANOVA formatting live in `python/deseq2_rust`.

use numpy::ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use deseq2_core::engine::{max_abs_log2fc, run_deseq2_diag, Comparison, Control, DeseqInput};

fn vector<'py>(py: Python<'py>, data: Vec<f64>) -> Bound<'py, PyAny> {
    Array1::from_vec(data).into_pyarray(py).into_any()
}

fn matrix<'py>(
    py: Python<'py>,
    data: Vec<f64>,
    rows: usize,
    cols: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let a = Array2::from_shape_vec((rows, cols), data)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(a.into_pyarray(py).into_any())
}

/// The DESeq2 engine on `counts`, `ngenes x nsamples` in sample order.
///
/// `controls` is a list of `(name, kind, values)` with `kind` `"categorical"` or `"numerical"`;
/// `comparisons` a list of `(left, right, encoded_left, encoded_right)`. Returns a dict with
/// `gene_ids`, `kept`, `ave_expr` and either `pairs` (dicts with `label`, `Log2FC`, `stat`, `SE`,
/// `CILeft`, `CIRight`, `CrILeft`, `CrIRight`, `PValue`, `AdjPValue`) or, for `anova`,
/// `anova_labels`, `anova_log2fc` (one array per comparison), `LRT`, `PValue`, `AdjPValue`,
/// `max_pair` and `max_log2fc`. Every vector covers all input genes in input order. Engine
/// errors raise `ValueError` with the production message.
///
/// With `diagnostics`, the dict also has `diag`: `kept_idx`, `size_factors`, `coef_names`, and
/// over the kept genes `base_mean`, `base_var`, `disp_gene_est`, `disp_fit`, `dispersion`,
/// `beta` and `se` (`nkept x p`, log2), `max_cooks`.
#[pyfunction]
#[pyo3(signature = (counts, gene_ids, sample_ids, condition_col, condition, controls, comparisons, anova = false, alpha = 0.05, shrink = "none", entity_type = "gene", diagnostics = false))]
#[allow(clippy::too_many_arguments)]
fn deseq2_pipeline<'py>(
    py: Python<'py>,
    counts: PyReadonlyArray2<'py, f64>,
    gene_ids: Vec<String>,
    sample_ids: Vec<String>,
    condition_col: String,
    condition: Vec<String>,
    controls: Vec<(String, String, Vec<String>)>,
    comparisons: Vec<(String, String, String, String)>,
    anova: bool,
    alpha: f64,
    shrink: &str,
    entity_type: &str,
    diagnostics: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let a = counts.as_array();
    let (ng, m) = a.dim();
    if ng != gene_ids.len() || m != sample_ids.len() {
        return Err(PyValueError::new_err(format!(
            "counts is {ng} x {m} but there are {} gene ids and {} sample ids",
            gene_ids.len(),
            sample_ids.len()
        )));
    }
    let controls = controls
        .into_iter()
        .map(|(name, kind, values)| {
            let numeric = match kind.as_str() {
                "categorical" => false,
                "numerical" => true,
                k => {
                    return Err(PyValueError::new_err(format!(
                        "control '{name}': unknown kind '{k}'"
                    )))
                }
            };
            Ok(Control {
                name,
                numeric,
                values,
            })
        })
        .collect::<PyResult<Vec<_>>>()?;
    let input = DeseqInput {
        gene_ids,
        sample_ids,
        counts: a.iter().copied().collect(),
        condition_col,
        condition,
        controls,
        comparisons: comparisons
            .into_iter()
            .map(|(left, right, encoded_left, encoded_right)| Comparison {
                left,
                right,
                encoded_left,
                encoded_right,
            })
            .collect(),
        anova,
        alpha,
        shrink: shrink.to_string(),
        entity_type: entity_type.to_string(),
    };
    let (out, diag) = py
        .allow_threads(|| run_deseq2_diag(&input))
        .map_err(PyValueError::new_err)?;

    let d = PyDict::new(py);
    d.set_item("gene_ids", out.gene_ids)?;
    d.set_item("kept", out.kept)?;
    d.set_item("ave_expr", vector(py, out.ave_expr))?;
    let pairs = PyList::empty(py);
    for p in out.pairs {
        let pd = PyDict::new(py);
        pd.set_item("label", p.label)?;
        pd.set_item("Log2FC", vector(py, p.log2fc))?;
        pd.set_item("stat", vector(py, p.stat))?;
        pd.set_item("SE", vector(py, p.se))?;
        pd.set_item("CILeft", vector(py, p.ci_left))?;
        pd.set_item("CIRight", vector(py, p.ci_right))?;
        pd.set_item("CrILeft", vector(py, p.cri_left))?;
        pd.set_item("CrIRight", vector(py, p.cri_right))?;
        pd.set_item("PValue", vector(py, p.pvalue))?;
        pd.set_item("AdjPValue", vector(py, p.adj_pvalue))?;
        pairs.append(pd)?;
    }
    d.set_item("pairs", pairs)?;
    if let Some(an) = out.anova {
        let lfcs: Vec<Vec<f64>> = an.log2fc.iter().map(|(_, v)| v.clone()).collect();
        let (max_pair, max_fc) = max_abs_log2fc(&lfcs);
        let labels: Vec<String> = an.log2fc.iter().map(|(l, _)| l.clone()).collect();
        let arrs = PyList::empty(py);
        for v in lfcs {
            arrs.append(vector(py, v))?;
        }
        d.set_item("anova_labels", labels)?;
        d.set_item("anova_log2fc", arrs)?;
        d.set_item("LRT", vector(py, an.lrt))?;
        d.set_item("PValue", vector(py, an.pvalue))?;
        d.set_item("AdjPValue", vector(py, an.adj_pvalue))?;
        d.set_item("max_pair", max_pair)?;
        d.set_item("max_log2fc", vector(py, max_fc))?;
    }
    if diagnostics {
        let f = &diag.fit;
        let p = f.test.p();
        let nk = diag.kept_idx.len();
        let dd = PyDict::new(py);
        dd.set_item("kept_idx", diag.kept_idx.clone())?;
        dd.set_item("size_factors", vector(py, f.sf.clone()))?;
        dd.set_item("coef_names", f.test.coef_names.clone())?;
        dd.set_item("base_mean", vector(py, f.base.base_mean.clone()))?;
        dd.set_item("base_var", vector(py, f.base.base_var.clone()))?;
        dd.set_item("disp_gene_est", vector(py, f.disp.gene_est.clone()))?;
        dd.set_item("disp_fit", vector(py, f.disp.fit.clone()))?;
        dd.set_item("dispersion", vector(py, f.disp.dispersion.clone()))?;
        dd.set_item("beta", matrix(py, f.test.beta.clone(), nk, p)?)?;
        dd.set_item("se", matrix(py, f.test.se.clone(), nk, p)?)?;
        dd.set_item("max_cooks", vector(py, f.test.max_cooks.clone()))?;
        d.set_item("diag", dd)?;
    }
    Ok(d)
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(deseq2_pipeline, m)?)?;
    Ok(())
}
