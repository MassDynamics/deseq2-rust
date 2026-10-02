//! `deseq2-core`: DESeq2 1.50.2 ported to Rust. No R at runtime.

// The numerics mirror DESeq2's C++, LAPACK and R loops index for index (and Armadillo's
// `(ad < eps || ad > 1/eps)` acceptance tests verbatim), so the summation order is easy to check
// against the reference.
#![allow(clippy::needless_range_loop, clippy::manual_range_contains)]

pub mod cpp;
pub mod design;
pub mod disp;
pub mod engine;
pub mod ext;
pub mod fit;
pub mod glm;
mod gnm;
pub mod la;
pub mod linpack;
pub mod nbtest;
pub mod prior_var;
pub mod results;
pub mod shrink;
