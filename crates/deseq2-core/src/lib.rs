//! `deseq2-core`: DESeq2 1.50.2 ported to Rust. No R at runtime.

pub mod cpp;
pub mod ext;
mod gnm;
pub mod la;
pub mod linpack;
pub mod prior_var;
