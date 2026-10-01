//! Active-set solves against `_ashr_trace_solve.bin` (written by
//! `corpus/count-reference/trace_mixsqp.R`): per QP step, the exact system R solved, its
//! solution, LAPACK's rcond estimates on Armadillo's two routes, and y after the step.
//! (1) kernel: Rust's `arma_solve` on R's (B, rhs); (2) path: Rust's own B, rhs, p, y.

mod common;
use common::*;
use shrink_core::ashr::ash_shrink;
use shrink_core::dense::Mat;
use shrink_core::linalg::{arma_solve, dgecon_1, dpocon_l};

struct Rec {
    sqp: usize,
    qp: usize,
    rc_po: f64,
    rc_ge: f64,
    b: Vec<f64>,
    rhs: Vec<f64>,
    p: Vec<f64>,
    y: Vec<f64>,
}

fn read_solve(path: &std::path::Path) -> Vec<Rec> {
    let raw = std::fs::read(path).unwrap();
    let v: Vec<f64> = raw.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < v.len() {
        let (n, m) = (v[i + 2] as usize, v[i + 3] as usize);
        let mut o = i + 6;
        let mut take = |k: usize| {
            let s = v[o..o + k].to_vec();
            o += k;
            s
        };
        let b = take(n * n);
        let rhs = take(n);
        let p = take(n);
        let y = take(m);
        out.push(Rec { sqp: v[i] as usize, qp: v[i + 1] as usize, rc_po: v[i + 4], rc_ge: v[i + 5], b, rhs, p, y });
        i = o;
    }
    out
}

fn bits_rel(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() {
        return f64::INFINITY;
    }
    a.iter().zip(b).map(|(x, y)| if x == y { 0.0 } else { ((x - y) / y.abs().max(f64::MIN_POSITIVE)).abs() }).fold(0.0, f64::max)
}

#[test]
fn mixsqp_solve() {
    let mut kernel_bad = 0;
    let mut path_bad = 0;
    for run in runs("_shrink_ashr") {
        for cmp in cmps(&run, "ashr") {
            let f = run.join(format!("{cmp}_ashr_trace_solve.bin"));
            if !f.exists() {
                continue;
            }
            let recs = read_solve(&f);
            let name = format!("{} {cmp}", run.file_name().unwrap().to_string_lossy());
            // (1) kernel on R's inputs
            for (q, r) in recs.iter().enumerate() {
                let n = r.rhs.len();
                let b = Mat::from_col_major(n, n, r.b.clone());
                let (p, route, _) = arma_solve(&b, &r.rhs);
                let po = dpocon_l(&b).unwrap_or(-1.0);
                let ge = dgecon_1(&b).unwrap_or(-1.0);
                let gp = bits_rel(&p, &r.p);
                if gp != 0.0 || po != r.rc_po || ge != r.rc_ge {
                    kernel_bad += 1;
                    println!(
                        "KERNEL {name} q{q} (it{} qp{}) n{n} {route:?}: p rel {gp:.1e} | rc_po R {:.17e} Rust {:.17e} | rc_ge R {:.17e} Rust {:.17e}",
                        r.sqp, r.qp, r.rc_po, po, r.rc_ge, ge
                    );
                }
            }
            // (2) path: first QP step where Rust's system or iterate differs from R's
            let data = Table::read(&run.join(format!("{cmp}_ashr_data.csv")));
            let fit = ash_shrink(&data.f("x"), &data.f("s")).unwrap();
            let qp = &fit.mixsqp.as_ref().unwrap().qp;
            let mut first = None;
            for (q, r) in recs.iter().enumerate() {
                let Some(s) = qp.get(q) else {
                    first = Some(format!("q{q}: Rust has only {} QP steps", qp.len()));
                    break;
                };
                let (gb, gr, gp, gy) = (bits_rel(&s.b, &r.b), bits_rel(&s.rhs, &r.rhs), bits_rel(&s.p, &r.p), bits_rel(&s.y, &r.y));
                if gb != 0.0 || gr != 0.0 || gp != 0.0 || gy != 0.0 {
                    first = Some(format!("q{q} (it{} qp{} n{}): B {gb:.1e} rhs {gr:.1e} p {gp:.1e} y {gy:.1e} route {:?}", r.sqp, r.qp, r.rhs.len(), s.route));
                    break;
                }
            }
            match first {
                Some(msg) => {
                    path_bad += 1;
                    println!("PATH {name}: {} R steps, first difference at {msg}", recs.len());
                }
                None => println!("PATH {name}: {} steps bit-identical (Rust {})", recs.len(), qp.len()),
            }
        }
    }
    println!("kernel mismatches {kernel_bad}, paths differing {path_bad}");
}
