# Scalar and small-matrix goldens for the limma -> Rust port.
#
# Two output families, both under $MD_GOLDEN_CORPUS_DIR:
#
#   scalar/<fn>.csv   — R nmath and limma helper functions evaluated on grids.
#                       Used by the Rust nmath / lowess port's unit tests.
#   matrix/<case>/    — lmFit -> contrasts.fit -> eBayes intermediates on a small
#                       real matrix (bojkova2020, first 600 GroupIds), for the
#                       lm / contrasts / ebayes / toptable cards. Every array limma
#                       produces is dumped so a divergence can be localised.
#
# Run:
#   cd /Users/giuseppeinfusini/wd/md-repos/MDFlexiComparisons
#   source .claude-r-env
#   env LC_COLLATE=C "$RSCRIPT_EXEC" data-raw/golden-corpus/scalar_goldens.R

suppressPackageStartupMessages({ library(data.table); library(limma); library(jsonlite) })

corpus_dir <- Sys.getenv("MD_GOLDEN_CORPUS_DIR", "/Users/giuseppeinfusini/wd/md-limma-golden-corpus")
scalar_dir <- file.path(corpus_dir, "scalar")
matrix_dir <- file.path(corpus_dir, "matrix")
dir.create(scalar_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(matrix_dir, showWarnings = FALSE, recursive = TRUE)

w <- function(df, path) { fwrite(as.data.frame(df, check.names = FALSE), path, na = "NA"); cat("wrote", path, nrow(df), "rows\n") }
wmat <- function(m, path) {
  m <- as.matrix(m)
  df <- data.frame(row = if (is.null(rownames(m))) seq_len(nrow(m)) else rownames(m), m, check.names = FALSE)
  w(df, path)
}

# ---------------------------------------------------------------- scalar grids
x_wide <- c(seq(-40, 40, by = 0.25), -1e3, -100, 100, 1e3, 1e6, -1e6)
df_grid <- c(0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 7.5, 10, 15, 20, 30, 50, 100, 300, 1e3, 1e4, 1e6)
p_grid <- c(10^seq(-300, -1, by = 1), seq(0.05, 0.95, by = 0.05), 1 - 10^seq(-1, -15, by = -1))

g <- expand.grid(x = x_wide, df = df_grid)
w(data.table(g, lower = pt(g$x, g$df), upper = pt(g$x, g$df, lower.tail = FALSE),
             log_lower = pt(g$x, g$df, log.p = TRUE), log_upper = pt(g$x, g$df, lower.tail = FALSE, log.p = TRUE)),
  file.path(scalar_dir, "pt.csv"))
g <- expand.grid(p = p_grid, df = df_grid)
w(data.table(g, q = qt(g$p, g$df), q_upper = qt(g$p, g$df, lower.tail = FALSE)), file.path(scalar_dir, "qt.csv"))

w(data.table(x = x_wide, lower = pnorm(x_wide), upper = pnorm(x_wide, lower.tail = FALSE),
             log_lower = pnorm(x_wide, log.p = TRUE), log_upper = pnorm(x_wide, lower.tail = FALSE, log.p = TRUE)),
  file.path(scalar_dir, "pnorm.csv"))
w(data.table(p = p_grid, q = qnorm(p_grid), q_upper = qnorm(p_grid, lower.tail = FALSE)), file.path(scalar_dir, "qnorm.csv"))

xf <- c(0, 1e-8, 1e-4, 0.01, 0.1, 0.5, 1, 1.5, 2, 3, 5, 10, 20, 50, 100, 1e3, 1e5)
g <- expand.grid(x = xf, df1 = c(1, 2, 3, 5, 10, 30, 100), df2 = df_grid)
w(data.table(g, lower = pf(g$x, g$df1, g$df2), upper = pf(g$x, g$df1, g$df2, lower.tail = FALSE),
             log_upper = pf(g$x, g$df1, g$df2, lower.tail = FALSE, log.p = TRUE)), file.path(scalar_dir, "pf.csv"))
g <- expand.grid(p = p_grid, df1 = c(1, 2, 3, 5, 10, 30), df2 = c(1, 2, 5, 10, 30, 100, 1e4))
w(data.table(g, q = qf(g$p, g$df1, g$df2), q_upper = qf(g$p, g$df1, g$df2, lower.tail = FALSE)), file.path(scalar_dir, "qf.csv"))

g <- expand.grid(x = xf, df = df_grid)
w(data.table(g, lower = pchisq(g$x, g$df), upper = pchisq(g$x, g$df, lower.tail = FALSE)), file.path(scalar_dir, "pchisq.csv"))
g <- expand.grid(p = p_grid, df = df_grid)
w(data.table(g, q = qchisq(g$p, g$df), q_upper = qchisq(g$p, g$df, lower.tail = FALSE)), file.path(scalar_dir, "qchisq.csv"))

xb <- c(0, 1e-10, 1e-5, 0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999, 1 - 1e-8, 1)
ab <- c(0.1, 0.5, 1, 1.5, 2, 5, 10, 50, 100, 1e3)
g <- expand.grid(x = xb, a = ab, b = ab)
w(data.table(g, lower = pbeta(g$x, g$a, g$b), upper = pbeta(g$x, g$a, g$b, lower.tail = FALSE)), file.path(scalar_dir, "pbeta.csv"))
g <- expand.grid(p = p_grid, a = ab, b = ab)
w(data.table(g, q = qbeta(g$p, g$a, g$b)), file.path(scalar_dir, "qbeta.csv"))

g <- expand.grid(x = xf, shape = ab)
w(data.table(g, lower = pgamma(g$x, g$shape), upper = pgamma(g$x, g$shape, lower.tail = FALSE)), file.path(scalar_dir, "pgamma.csv"))
g <- expand.grid(p = p_grid, shape = ab)
w(data.table(g, q = qgamma(g$p, g$shape)), file.path(scalar_dir, "qgamma.csv"))

xg <- c(1e-8, 1e-4, 0.01, 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 5, 7.5, 10, 20, 50, 100, 1e3, 1e5, 1e8)
w(data.table(x = xg, lgamma = lgamma(xg), digamma = digamma(xg), trigamma = trigamma(xg)), file.path(scalar_dir, "gamma_family.csv"))
xt <- c(1e-6, 1e-4, 0.001, 0.01, 0.05, 0.1, 0.5, 1, 1.5, 2, 5, 10, 100, 1e4, 1e6)
w(data.table(x = xt, trigamma_inverse = limma::trigammaInverse(xt)), file.path(scalar_dir, "trigamma_inverse.csv"))

# lowess: R's C clowess (stats::lowess), the exact routine limma's trend fit uses.
set.seed(20260925)
for (case in list(list(n = 50, f = 2/3), list(n = 500, f = 0.5), list(n = 3000, f = 0.5), list(n = 200, f = 0.3))) {
  x <- sort(rnorm(case$n, 8, 2)); y <- log(0.05 + 0.02 * (x - 8)^2 + rgamma(case$n, 2, 4))
  x[sample(case$n, ceiling(case$n / 10))] <- x[sample(case$n, ceiling(case$n / 10))] # ties on purpose
  lo <- lowess(x, y, f = case$f, iter = 3)
  w(data.table(x = x, y = y, lowess_x = lo$x, lowess_y = lo$y), file.path(scalar_dir, sprintf("lowess_n%d_f%s.csv", case$n, gsub("\\.", "p", format(case$f, digits = 3)))))
}

# fitFDist / fitFDistRobustly / squeezeVar on synthetic variance vectors (the eBayes prior fit).
set.seed(1)
for (case in list(list(n = 200, df = 4), list(n = 2000, df = 10), list(n = 5000, df = 3))) {
  s2 <- (0.05 + 0.5 * rgamma(case$n, 2, 3)) * rchisq(case$n, case$df) / case$df
  s2[sample(case$n, 5)] <- s2[sample(case$n, 5)] * 40 # outliers so robust differs from plain
  amean <- sort(rnorm(case$n, 9, 2))
  tag <- sprintf("n%d_df%d", case$n, case$df)
  plain <- limma::fitFDist(s2, df1 = case$df)
  rob <- limma::fitFDistRobustly(s2, df1 = case$df)
  trend <- limma::fitFDist(s2, df1 = case$df, covariate = amean)
  robtrend <- limma::fitFDistRobustly(s2, df1 = case$df, covariate = amean)
  w(data.table(s2 = s2, df1 = case$df, amean = amean,
               trend_scale = trend$scale, robust_df2_shrunk = rob$df2.shrunk, robtrend_scale = robtrend$scale, robtrend_df2_shrunk = robtrend$df2.shrunk),
    file.path(scalar_dir, sprintf("fitfdist_%s_vectors.csv", tag)))
  writeLines(toJSON(list(n = case$n, df1 = case$df,
                         plain = list(scale = plain$scale, df2 = plain$df2),
                         robust = list(scale = rob$scale, df2 = rob$df2),
                         trend = list(df2 = trend$df2),
                         robtrend = list(df2 = robtrend$df2)), auto_unbox = TRUE, digits = NA),
             file.path(scalar_dir, sprintf("fitfdist_%s_scalars.json", tag)))
}

# ---------------------------------------------------------------- matrix goldens
shared <- file.path(corpus_dir, "shared", "bojkova2020")
intensity <- as.data.table(readRDS(file.path(shared, "intensity.rds")))
design_df <- readRDS(file.path(shared, "experiment_design.rds"))
keep_ids <- sort(unique(intensity$GroupId))[1:600]
sub <- intensity[GroupId %in% keep_ids]
sub[, value := ifelse(Imputed == 1 | NormalisedIntensity <= 0, NA_real_, log2(NormalisedIntensity))]
wide <- dcast(sub, GroupId ~ replicate, value.var = "value")
mat <- as.matrix(wide[, -1]); rownames(mat) <- wide$GroupId
mat <- mat[, design_df$sample_name] # column order = experiment design order
cond <- factor(design_df$condition, levels = sort(unique(design_df$condition), method = "radix"))
set.seed(7)
design_df$numcov <- round(rnorm(nrow(design_df), 0, 1), 6)
design_df$batch <- factor(rep(c("A", "B"), length.out = nrow(design_df)))

# Production prefixes levels with "condition" (model.matrix default), which also makes them valid names.
lev <- paste0("condition", levels(cond))
pairs <- combn(lev, 2)
contrast_names <- apply(pairs, 2, function(p) sprintf("%s - %s", p[1], p[2]))

run_case <- function(name, design, contrasts) {
  d <- file.path(matrix_dir, name); dir.create(d, showWarnings = FALSE)
  wmat(mat, file.path(d, "input_log2.csv"))
  wmat(design, file.path(d, "design.csv"))
  wmat(contrasts, file.path(d, "contrasts.csv"))
  fit <- lmFit(mat, design)
  wmat(fit$coefficients, file.path(d, "lmfit_coefficients.csv"))
  wmat(fit$stdev.unscaled, file.path(d, "lmfit_stdev_unscaled.csv"))
  w(data.table(row = rownames(mat), sigma = fit$sigma, df_residual = fit$df.residual, Amean = fit$Amean), file.path(d, "lmfit_scalars.csv"))
  wmat(fit$cov.coefficients, file.path(d, "lmfit_cov_coefficients.csv"))
  cf <- contrasts.fit(fit, contrasts)
  wmat(cf$coefficients, file.path(d, "contrasts_coefficients.csv"))
  wmat(cf$stdev.unscaled, file.path(d, "contrasts_stdev_unscaled.csv"))
  wmat(cf$cov.coefficients, file.path(d, "contrasts_cov_coefficients.csv"))
  for (robust in c(FALSE, TRUE)) for (trend in c(FALSE, TRUE)) {
    eb <- eBayes(cf, robust = robust, trend = trend)
    tag <- sprintf("ebayes_rob%s_trend%s", robust, trend)
    wmat(eb$t, file.path(d, paste0(tag, "_t.csv")))
    wmat(eb$p.value, file.path(d, paste0(tag, "_p.csv")))
    wmat(eb$lods, file.path(d, paste0(tag, "_lods.csv")))
    w(data.table(row = rownames(mat), s2_post = eb$s2.post, df_total = eb$df.total, F = eb$F, F_p = eb$F.p.value,
                 s2_prior = if (length(eb$s2.prior) == 1) rep(eb$s2.prior, nrow(mat)) else eb$s2.prior,
                 df_prior = if (length(eb$df.prior) == 1) rep(eb$df.prior, nrow(mat)) else eb$df.prior),
      file.path(d, paste0(tag, "_scalars.csv")))
    for (j in seq_len(ncol(contrasts))) {
      tt <- topTable(eb, coef = j, number = Inf, sort.by = "none", confint = 0.95)
      w(data.table(row = rownames(tt), tt), file.path(d, sprintf("%s_toptable_%d.csv", tag, j)))
    }
    dt <- decideTests(eb)
    wmat(unclass(dt), file.path(d, paste0(tag, "_decidetests.csv")))
    ttF <- topTable(eb, number = Inf, sort.by = "none")
    w(data.table(row = rownames(ttF), ttF), file.path(d, paste0(tag, "_toptable_F.csv")))
  }
  writeLines(toJSON(list(case = name, n_rows = nrow(mat), n_samples = ncol(mat), limma = as.character(packageVersion("limma")),
                         r = R.version.string, conditions = levels(cond), contrasts = colnames(contrasts)), auto_unbox = TRUE, pretty = TRUE),
             file.path(d, "case.json"))
}

d0 <- model.matrix(~ 0 + cond); colnames(d0) <- lev
c0 <- makeContrasts(contrasts = contrast_names, levels = d0)
run_case("conditions_only", d0, c0)

d1 <- model.matrix(~ 0 + cond + numcov, data = design_df); colnames(d1) <- c(lev, "numcov")
c1 <- makeContrasts(contrasts = contrast_names, levels = d1)
run_case("conditions_numeric_covariate", d1, c1)

d2 <- model.matrix(~ 0 + cond + batch + numcov, data = design_df); colnames(d2) <- c(lev, "batchB", "numcov")
c2 <- makeContrasts(contrasts = contrast_names, levels = d2)
run_case("conditions_categorical_numeric_covariates", d2, c2)

cat("done\n")
