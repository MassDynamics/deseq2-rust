//! `shrink-core`: lfcShrink apeglm and ashr, ported to Rust. No R at runtime.
//!
//! Inputs are the unshrunk DESeq2 fit as plain arrays; this crate does not depend on
//! deseq2-core.

// Index loops mirror the Fortran / C++ sources each kernel is ported from.
#![allow(clippy::needless_range_loop)]

pub mod apeglm;
pub mod ashr;
pub mod dense;
pub mod eigen;
pub mod lapack;
pub mod linalg;
pub mod mixsqp;
pub mod xld;

#[derive(Debug, thiserror::Error)]
pub enum ShrinkError {
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("numerical failure: {0}")]
    Numerical(String),
    #[error("unsupported path: {0}")]
    Unsupported(String),
}

pub use apeglm::{shrink_apeglm, ApeglmFit};
pub use ashr::{ash_shrink, AshrFit, AshrTable};

/// `lfcShrink(type = "ashr")`: shrunken log2 fold changes and their SEs (plus the full ash
/// table) from the MLE `lfc_mle` and `lfc_se` of one coefficient.
pub fn shrink_ashr(lfc_mle: &[f64], lfc_se: &[f64]) -> Result<AshrFit, ShrinkError> {
    ash_shrink(lfc_mle, lfc_se)
}
