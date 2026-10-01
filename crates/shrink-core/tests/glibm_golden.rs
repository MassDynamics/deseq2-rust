//! `glibm::{exp, log, log1p}` against the reference R's glibc, bit for bit
//! (`reference-glibm/glibm_ref.bin`, from `corpus/count-reference/ref_glibm.R`).
mod common;
use shrink_core::glibm;

#[test]
fn glibm_matches_reference_glibc() {
    let path = common::corpus_dir().join("reference-glibm/glibm_ref.bin");
    let Ok(b) = std::fs::read(&path) else {
        eprintln!("SKIP: {} not found", path.display());
        return;
    };
    let v: Vec<f64> = b
        .chunks(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let (ne, nl, np) = (v[0] as usize, v[1] as usize, v[2] as usize);
    let mut o = 3;
    let mut take = |n: usize| {
        let s = v[o..o + n].to_vec();
        o += n;
        s
    };
    let (xe, ye, xl, yl, xp, yp) = (take(ne), take(ne), take(nl), take(nl), take(np), take(np));
    let bad = |x: &[f64], y: &[f64], f: fn(f64) -> f64| {
        x.iter()
            .zip(y)
            .filter(|(a, b)| f(**a).to_bits() != b.to_bits())
            .count()
    };
    let (be, bl, bp) = (
        bad(&xe, &ye, glibm::exp),
        bad(&xl, &yl, glibm::log),
        bad(&xp, &yp, glibm::log1p),
    );
    println!("glibm mismatches: exp {be}/{ne}, log {bl}/{nl}, log1p {bp}/{np}");
    assert_eq!((be, bl, bp), (0, 0, 0));
}
