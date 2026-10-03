//! mix-SQP iterate-by-iterate comparison with the instrumented R trace
//! (`corpus/count-reference/mixsqp_trace.cpp`). Prints the per-iteration gaps; asserts the
//! deterministic part (the first SQP iteration's EM update, objective and gradient
//! statistic) at rel 1e-8.

mod common;
use common::*;
use shrink_core::ashr::ash_shrink;

#[test]
fn mixsqp_trace() {
    let runs = runs("_shrink_ashr");
    // No corpus fails the test rather than passing it empty (review r1, D-11).
    assert_eq!(
        runs.len(),
        5,
        "_shrink_ashr runs in the corpus (set MD_COUNT_CORPUS_DIR)"
    );
    for run in runs {
        for cmp in cmps(&run, "ashr") {
            let p = |s: &str| run.join(format!("{cmp}_ashr_{s}.csv"));
            let data = Table::read(&p("data"));
            let fit = ash_shrink(&data.f("x"), &data.f("s")).unwrap();
            let ms = fit.mixsqp.as_ref().unwrap();
            let tr = Table::read(&p("trace_sqp"));
            let xem = tr.numbered("xem");
            let ys = tr.numbered("y");
            let obj = tr.f("obj");
            let gmin = tr.f("gmin");
            let step = tr.f("step");
            println!(
                "== {} {cmp}: R {} SQP iters, Rust {}",
                run.file_name().unwrap().to_string_lossy(),
                tr.nrow,
                ms.sqp.len()
            );
            for it in 0..tr.nrow.min(ms.sqp.len()).min(6) {
                let r_xem: Vec<f64> = xem.iter().map(|c| c[it]).collect();
                let s = &ms.sqp[it];
                let gx = max_abs(&s.x_em, &r_xem);
                let gy = match &s.y {
                    Some(y) if !ys[0][it].is_nan() => {
                        max_abs(y, &ys.iter().map(|c| c[it]).collect::<Vec<_>>())
                    }
                    _ => f64::NAN,
                };
                println!(
                    "  it {it}: xem abs {gx:.1e} obj rel {:.1e} gmin R {:.3e} Rust {:.3e} y abs {gy:.1e} step R {} Rust {:?} nqp {}",
                    ((s.obj - obj[it]) / obj[it]).abs(),
                    gmin[it],
                    s.gmin,
                    step[it],
                    s.step,
                    s.nqp
                );
            }
            let s0 = &ms.sqp[0];
            let r0: Vec<f64> = xem.iter().map(|c| c[0]).collect();
            assert!(max_rel(&s0.x_em, &r0) <= 1e-8);
            assert!(((s0.obj - obj[0]) / obj[0]).abs() <= 1e-8);
            assert!(((s0.gmin - gmin[0]) / gmin[0]).abs() <= 1e-8);
        }
    }
}

#[test]
#[ignore]
fn qp_steps_one() {
    let pat = std::env::var("RUN").unwrap_or("count_synth_shrink_ashr".into());
    let cmp = std::env::var("CMP").unwrap_or("shrink_cmp_01".into());
    let run = runs(&pat).into_iter().next().unwrap();
    let p = |s: &str| run.join(format!("{cmp}_ashr_{s}.csv"));
    let data = Table::read(&p("data"));
    let fit = ash_shrink(&data.f("x"), &data.f("s")).unwrap();
    let ms = fit.mixsqp.as_ref().unwrap();
    let qp = Table::read(&p("trace_qp"));
    let (it, nws, rc, ok, pn, kind, k, st) = (
        qp.f("sqp_iter"),
        qp.f("n_ws"),
        qp.f("rcond"),
        qp.f("noapprox_ok"),
        qp.f("pnorm_inf"),
        qp.f("kind"),
        qp.f("k"),
        qp.f("step"),
    );
    for q in 0..qp.nrow.min(ms.qp.len()).min(45) {
        let r = &ms.qp[q];
        println!(
            "R it{} ws{} rc {:.2e} ok{} pn {:.3e} kind{} k{} a {:.3e} || Rust it{} ws{} rc {:.2e} {:?} pn {:.3e} kind{} k{} a {:.3e} acorr {:.1e}",
            it[q], nws[q], rc[q], ok[q], pn[q], kind[q], k[q], st[q], r.sqp_iter, r.n_ws, r.rcond, r.route, r.pnorm_inf, r.kind, r.k, r.step, r.a_corr
        );
    }
    let tr = Table::read(&p("trace_sqp"));
    let ys = tr.numbered("y");
    let xn = tr.numbered("xnew");
    for (it, s) in ms.sqp.iter().enumerate() {
        if let Some(y) = &s.y {
            println!(
                "it{it} y Rust {:?}",
                y.iter().map(|v| format!("{v:.3e}")).collect::<Vec<_>>()
            );
            if it < tr.nrow {
                println!(
                    "it{it} y R    {:?}",
                    ys.iter()
                        .map(|c| format!("{:.3e}", c[it]))
                        .collect::<Vec<_>>()
                );
            }
        }
    }
    println!(
        "x Rust {:?}",
        ms.x.iter().map(|v| format!("{v:.3e}")).collect::<Vec<_>>()
    );
    println!(
        "x R    {:?}",
        Table::read(&p("mixsqp_x"))
            .f("x")
            .iter()
            .map(|v| format!("{v:.3e}"))
            .collect::<Vec<_>>()
    );
    let _ = xn;
}
