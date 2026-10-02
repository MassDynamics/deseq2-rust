//! `optim(method = "L-BFGS-B")` against R 4.5.0 in the reference image, bit for bit. The R side
//! is `tests/data/lbfgsb/optim_cases.R` (bounded Rosenbrock problems: interior optimum, active
//! upper bound, 4 and 7 parameters, a start outside the box).

// Expected values are R's %.17g output, kept as printed.
#![allow(clippy::excessive_precision)]

use deseq2_core::lbfgsb::optim_lbfgsb;

fn fr(x: &[f64]) -> f64 {
    let mut s = 0.0;
    for i in 0..x.len() - 1 {
        let a = x[i + 1] - x[i] * x[i];
        let b = 1.0 - x[i];
        s = s + 100.0 * a * a + b * b;
    }
    s
}

struct Case {
    par: Vec<f64>,
    lo: Vec<f64>,
    up: Vec<f64>,
    conv: i32,
    counts: i64,
    value: f64,
    msg: &'static str,
    out: Vec<f64>,
}

#[test]
fn optim_lbfgsb_matches_r_bit_for_bit() {
    let rel = "CONVERGENCE: REL_REDUCTION_OF_F <= FACTR*EPSMCH";
    let cases = [
        Case {
            par: vec![-1.2, 1.0],
            lo: vec![-30.0; 2],
            up: vec![30.0; 2],
            conv: 0,
            counts: 53,
            value: 3.9982186996471701e-08,
            msg: rel,
            out: vec![0.99980004453756177, 0.9996001284442364],
        },
        Case {
            par: vec![-1.2, 1.0],
            lo: vec![-2.0, -2.0],
            up: vec![0.5, 2.0],
            conv: 0,
            counts: 31,
            value: 0.25,
            msg: "CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL",
            out: vec![0.5, 0.24999999999999994],
        },
        Case {
            par: vec![3.0, -3.0, 2.0, 0.5],
            lo: vec![-4.0; 4],
            up: vec![4.0; 4],
            conv: 0,
            counts: 65,
            value: 8.9267435916092381e-08,
            msg: rel,
            out: vec![
                0.99993510680126119,
                0.99986990672583453,
                0.99973910017132628,
                0.9994783259134955,
            ],
        },
        Case {
            par: vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
            lo: vec![-1.0; 7],
            up: vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 0.8],
            conv: 0,
            counts: 62,
            value: 0.015014464953836945,
            msg: rel,
            out: vec![
                0.99659032143122706,
                0.99317629130705309,
                0.98635765293610123,
                0.9728121732651186,
                0.94617901961718009,
                0.89487382403707982,
                0.80000000000000004,
            ],
        },
        Case {
            par: vec![40.0, -35.0],
            lo: vec![-30.0; 2],
            up: vec![30.0; 2],
            conv: 0,
            counts: 36,
            value: 3.9917814451070946e-08,
            msg: rel,
            out: vec![0.99980020558187233, 0.99960044405581938],
        },
    ];
    for (k, c) in cases.iter().enumerate() {
        let o = optim_lbfgsb(&mut |p: &[f64]| fr(p), &c.par, &c.lo, &c.up).unwrap();
        assert_eq!(o.convergence, c.conv, "case {}", k + 1);
        assert_eq!(o.counts, c.counts, "case {}", k + 1);
        assert_eq!(o.message, c.msg, "case {}", k + 1);
        assert_eq!(o.value.to_bits(), c.value.to_bits(), "case {} value", k + 1);
        for (a, b) in o.par.iter().zip(&c.out) {
            assert_eq!(a.to_bits(), b.to_bits(), "case {} par {:?}", k + 1, o.par);
        }
    }
}
