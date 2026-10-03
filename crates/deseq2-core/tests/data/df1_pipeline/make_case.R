# Builds a full-pipeline DESeq2 run with m - p = 1 (review r1, stats item 8b) and writes R's
# results for tests/df1_pipeline.rs. Run in the production image (R 4.5.0, DESeq2 1.50.2):
#   docker run --rm -v "$PWD":/w -w /w md-flexi-r45-local:latest Rscript make_case.R
# Two conditions x two replicates plus a numeric dose: 4 samples, 3 coefficients, so
# estimateDispersionsPriorVar takes the df = 1 simulation (rchisq's gamma GS branch). Log
# dispersions scatter around the trend with variance sigma2, chosen so the KL argmin is interior
# (dispPriorVar above 0.5, not the 0.25 floor); the script stops otherwise.
suppressPackageStartupMessages(library(DESeq2))
set.seed(23)
n <- 3000
sigma2 <- 4  # 1 and 2 floor at 0.25, 3 gives 0.4965
cd <- data.frame(cond = factor(c("A", "A", "B", "B")), dose = c(0.3, 1.1, 0.7, 1.6))
mu0 <- rgamma(n, 0.6, 0.004) + 0.5
disp <- exp(log(0.04 + 1.5 / mu0) + rnorm(n, 0, sqrt(sigma2)))
lfc <- ifelse(runif(n) < 0.15, rnorm(n, 0, 1.5), 0)
x <- model.matrix(~ cond + dose, cd)
beta <- cbind(log(mu0), lfc * log(2), rnorm(n, 0, 0.3))
mu <- exp(beta %*% t(x)) * rep(c(0.8, 1.2, 1.0, 0.9), each = n)
cnt <- matrix(rnbinom(n * 4, mu = mu, size = 1 / disp), n)
storage.mode(cnt) <- "integer"
write.csv(cnt, "counts.csv", row.names = FALSE)
write.csv(cd, "coldata.csv", row.names = FALSE, quote = FALSE)

dds <- DESeqDataSetFromMatrix(cnt, cd, ~ cond + dose)
dds <- DESeq(dds, quiet = TRUE)
pv <- attr(dispersionFunction(dds), "dispPriorVar")
if (pv <= 0.5) stop("dispPriorVar ", pv, " is not interior; change sigma2 or the seed")
mc <- mcols(dds)
r <- results(dds, contrast = c("cond", "B", "A"))
rd <- results(dds, name = "dose")
f17 <- function(v) ifelse(is.na(v), "NA", sprintf("%.17g", v))
out <- data.frame(
  baseMean = f17(mc$baseMean), allZero = mc$allZero,
  dispGeneEst = f17(mc$dispGeneEst), dispFit = f17(mc$dispFit), dispMAP = f17(mc$dispMAP),
  dispersion = f17(mc$dispersion), dispOutlier = mc$dispOutlier,
  lfc = f17(r$log2FoldChange), lfcSE = f17(r$lfcSE), stat = f17(r$stat),
  pvalue = f17(r$pvalue), padj = f17(r$padj),
  dose_lfc = f17(rd$log2FoldChange), dose_lfcSE = f17(rd$lfcSE), dose_stat = f17(rd$stat),
  dose_pvalue = f17(rd$pvalue), dose_padj = f17(rd$padj)
)
write.csv(out, "expected.csv", row.names = FALSE, quote = FALSE)
writeLines(c(
  sprintf("dispPriorVar %.17g", pv),
  sprintf("varLogDispEsts %.17g", attr(dispersionFunction(dds), "varLogDispEsts")),
  sprintf("fitType %s", attr(dispersionFunction(dds), "fitType")),
  sprintf("sizeFactors %s", paste(sprintf("%.17g", sizeFactors(dds)), collapse = " "))
), "scalars.txt")
cat("dispPriorVar", pv, "fitType", attr(dispersionFunction(dds), "fitType"),
    "padj < 0.1 (cond):", sum(r$padj < 0.1, na.rm = TRUE),
    "NA padj:", sum(is.na(r$padj)), "dispOutlier:", sum(mc$dispOutlier, na.rm = TRUE), "\n")
