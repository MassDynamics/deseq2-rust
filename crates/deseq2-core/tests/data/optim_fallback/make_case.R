# Builds a DESeq2 input whose IRLS fit diverges (|beta| > 30 on the natural log scale) on several
# rows, so DESeq() refits them with fitNbinomGLMsOptim, and writes R's results for
# tests/optim_fallback.rs. Run in md-flexi-r45-local:latest (R 4.5.0, DESeq2 1.50.2):
#   docker run --rm -v "$PWD":/w -w /w md-flexi-r45-local:latest Rscript make_case.R
suppressPackageStartupMessages(library(DESeq2))
set.seed(7)
n <- 300; m <- 9
cd <- data.frame(batch = factor(rep(c("b1", "b2", "b3"), 3)),
                 cond = factor(rep(c("A", "B", "C"), each = 3)))
mu <- matrix(rep(rgamma(n, 1, 0.01), m), n)
cnt <- matrix(rnbinom(n * m, mu = mu, size = 5), n)
# One large sample with zeros elsewhere sends the fit past |beta| = 30.
eng <- rbind(
  c(0, 0, 0, 0, 0, 0, 0, 0, 90000),
  c(0, 0, 0, 0, 0, 0, 0, 40000, 0),
  c(0, 0, 0, 0, 0, 0, 7000, 0, 0),
  c(0, 0, 0, 0, 0, 0, 3, 0, 0),
  c(0, 0, 0, 0, 0, 0, 0, 0, 500),
  c(0, 0, 60000, 0, 0, 0, 0, 0, 0),
  c(0, 0, 0, 0, 25000, 0, 0, 0, 0),
  c(0, 0, 0, 0, 0, 0, 2000, 2500, 1800),
  c(5, 0, 0, 0, 0, 0, 0, 0, 80000)
)
cnt <- rbind(eng, cnt)
storage.mode(cnt) <- "integer"
write.csv(cnt, "counts.csv", row.names = FALSE)
write.csv(cd, "coldata.csv", row.names = FALSE, quote = FALSE)

dds <- DESeqDataSetFromMatrix(cnt, cd, ~ batch + cond)
dds <- DESeq(dds, quiet = TRUE)
mc <- mcols(dds)
coefs <- resultsNames(dds)
f17 <- function(v) ifelse(is.na(v), "NA", sprintf("%.17g", v))
out <- data.frame(betaIter = mc$betaIter, betaConv = mc$betaConv)
for (cf in coefs) {
  out[[paste0("beta_", cf)]] <- f17(mc[[cf]])
  out[[paste0("SE_", cf)]] <- f17(mc[[paste0("SE_", cf)]])
  out[[paste0("stat_", cf)]] <- f17(mc[[paste0("WaldStatistic_", cf)]])
  out[[paste0("p_", cf)]] <- f17(mc[[paste0("WaldPvalue_", cf)]])
}
write.csv(out, "expected.csv", row.names = FALSE, quote = FALSE)
cat("coefs:", coefs, "\n")
cat("rows refit by optim (betaIter == 100):", which(mc$betaIter == 100), "\n")
