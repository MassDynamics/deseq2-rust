//! `deseq2-core`: DESeq2 1.50.2 ported to Rust. No R at runtime.

pub mod cpp;
pub mod design;
pub mod disp;
pub mod ext;
pub mod glm;
mod gnm;
pub mod la;
pub mod linpack;
pub mod prior_var;
