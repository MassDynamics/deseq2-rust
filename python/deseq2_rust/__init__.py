"""DESeq2 differential expression, ported to Rust.

``run`` takes what MDFlexiComparisons hands the DESeq2 engine (count matrix, sample info,
comparisons, params) and returns the production output table.
"""

from deseq2_rust.deseq2 import run

__all__ = ["run"]
