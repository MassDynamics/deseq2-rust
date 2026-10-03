//! Review deseq2 r2, SE-m3. D-12 is deferred: ashr input whose likelihood matrix is low rank (many identical (lfc, se)
//! pairs) takes R's unseeded irlba path, which is not ported. The deferral is only safe while the
//! port refuses that input loudly; this locks the refusal so it cannot turn into a silent answer.

use shrink_core::{shrink_ashr, ShrinkError};

#[test]
fn ashr_low_rank_input_is_refused_not_answered() {
    let x = vec![0.5; 60];
    let s = vec![0.2; 60];
    match shrink_ashr(&x, &s) {
        Err(ShrinkError::Unsupported(msg)) => assert!(msg.contains("low-rank"), "{msg}"),
        Err(e) => panic!("refused, but not as the deferred low-rank path: {e}"),
        Ok(_) => panic!("low-rank ashr input returned an answer; D-12 was deferred as a refusal"),
    }
}
