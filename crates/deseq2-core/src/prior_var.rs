//! Prior variance of the log dispersion when `m - p <= 3`
//! (`DESeq2:::estimateDispersionsPriorVar`, simulation branch).
//!
//! Under `set.seed(2)`, for each of 200 grid variances `x` in `seq(0, 8, length = 200)` the
//! routine draws 1e4 values of `log(rchisq(df)) + rnorm(0, sqrt(x)) - log(df)`, compares their
//! histogram on breaks `-10..10` by `0.5` with the histogram of the observed log dispersion
//! residuals by KL divergence, smooths the 200 KL values with `loess(span = 0.2)`, and takes
//! the argmin of the smoothed curve on a 1000-point grid. The result is floored at 0.25.

use rnum::hist::hist;
use rnum::loess::loess;
use rnum::rng::RRng;

/// Every intermediate of the simulation, for checking against the corpus goldens.
#[derive(Debug, Clone)]
pub struct PriorVarSim {
    /// Residuals kept strictly inside (-10, 10).
    pub obs: Vec<f64>,
    /// Observed histogram counts (40 bins).
    pub obs_counts: Vec<i64>,
    /// Observed histogram density (40 bins).
    pub obs_density: Vec<f64>,
    /// The 200-point variance grid.
    pub grid: Vec<f64>,
    /// Simulated histogram counts, one row of 40 per grid point.
    pub sim_counts: Vec<Vec<i64>>,
    /// KL divergence per grid point.
    pub kl: Vec<f64>,
    /// `fitted(loess(kl ~ grid, span = 0.2))`.
    pub loess_fitted: Vec<f64>,
    /// The 1000-point fine grid.
    pub fine_grid: Vec<f64>,
    /// `predict(loess, fine_grid)`.
    pub fine_predicted: Vec<f64>,
    /// 1-based index of the minimum of `fine_predicted` (`which.min`).
    pub argmin_index: usize,
    /// `fine_grid[argmin_index]`.
    pub argmin: f64,
    /// `max(argmin, 0.25)`: the dispersion prior variance.
    pub prior_var: f64,
}

/// `seq(from, to, length.out = n)` as R computes it for `n > 2`.
pub(crate) fn seq_len_out(from: f64, to: f64, n: usize) -> Vec<f64> {
    // seq.default: by <- (to - from)/(length.out - 1); from + (0:(n-1)) * by, last forced to `to`.
    let by = (to - from) / (n as f64 - 1.0);
    let mut v: Vec<f64> = (0..n).map(|i| from + i as f64 * by).collect();
    if n > 0 {
        v[n - 1] = to;
    }
    v
}

/// Run the simulation for residuals `resid` (already restricted to `dispGeneEst >= 1e-6`)
/// and residual degrees of freedom `df = m - p`.
pub fn prior_var_simulation(resid: &[f64], df: f64) -> Result<PriorVarSim, String> {
    let mut rng = RRng::set_seed(2);
    let brks: Vec<f64> = (-20..=20).map(|i| i as f64 / 2.0).collect();
    let (bmin, bmax) = (brks[0], brks[brks.len() - 1]);
    let obs: Vec<f64> = resid.iter().copied().filter(|&o| o > bmin && o < bmax).collect();
    let grid = seq_len_out(0.0, 8.0, 200);
    let oh = hist(&obs, &brks)?;
    let mut sim_counts = Vec::with_capacity(grid.len());
    let mut kl = Vec::with_capacity(grid.len());
    let ldf = df.ln();
    let mut chis = vec![0.0; 10000];
    for &g in &grid {
        for c in chis.iter_mut() {
            *c = rng.rchisq(df);
        }
        let sd = g.sqrt();
        let mut rd = Vec::with_capacity(10000);
        for &c in &chis {
            let z = rng.rnorm(0.0, sd);
            let v = (c.ln() + z) - ldf;
            if v > bmin && v < bmax {
                rd.push(v);
            }
        }
        let rh = hist(&rd, &brks)?;
        let small = oh
            .density
            .iter()
            .chain(rh.density.iter())
            .copied()
            .filter(|&z| z > 0.0)
            .fold(f64::INFINITY, f64::min);
        let mut s = 0.0;
        for (o, r) in oh.density.iter().zip(rh.density.iter()) {
            s += o * ((o + small).ln() - (r + small).ln());
        }
        kl.push(s);
        sim_counts.push(rh.counts);
    }
    let fit = loess(&grid, &kl, 0.2, 2)?;
    let fine_grid = seq_len_out(0.0, 8.0, 1000);
    let fine_predicted = fit.predict(&fine_grid);
    let mut j = 0usize;
    for (i, &v) in fine_predicted.iter().enumerate() {
        if !v.is_nan() && (fine_predicted[j].is_nan() || v < fine_predicted[j]) {
            j = i;
        }
    }
    let argmin = fine_grid[j];
    Ok(PriorVarSim {
        obs,
        obs_counts: oh.counts,
        obs_density: oh.density,
        grid,
        sim_counts,
        kl,
        loess_fitted: fit.fitted().to_vec(),
        fine_grid,
        fine_predicted,
        argmin_index: j + 1,
        argmin,
        prior_var: argmin.max(0.25),
    })
}
