# DESeq2 1.50.2 reference (PR-E2): DESeq() rebuilt stage by stage.
#
# The iterative fits (gene-wise and MAP dispersion, the NB GLM IRLS, the beta
# prior fit) are DESeq2's own stage functions, called in the order DESeq() and
# results() call them, so each stage's output can be dumped as its own golden.
# Everything closed-form between those stages (size factors, base means, the
# parametric trend, dispersion prior variance, outlier flags, Wald and LRT
# p-values, Cook's distances, outlier replacement, the Cook's filter and its
# two-group rescue, independent filtering, BH) is recomputed in plain R and
# checked against the package value.
#
# Production behaviour is reproduced as shipped, including the two known
# defects: the per-comparison relevel refit (nbinomWaldTest on the original
# counts with the post-replacement dispersions, Cook's recomputed without the
# replaceable-column zeroing) and the dead first results() call (skipped,
# since it has no effect on the output).

DESEQ2_MIN_DISP <- 1e-8

n_or_more_in_cell <- function(mm, n) {
  h <- apply(mm, 1, paste0, collapse = "_")
  as.vector(unname(table(h)[h])) >= n
}

dump_df <- function(d, name, df, ids) {
  df <- as.data.frame(df)
  fwrite(cbind(data.table(id = ids), as.data.table(df)), file.path(d$dir, paste0(name, ".csv")), na = "NA")
}

# Compare every atomic mcols column, the size factors and the assays of two
# DESeqDataSets. Used to prove the stage-by-stage rebuild equals DESeq().
check_dds <- function(d, prefix, ours, pkg) {
  d$check(paste0(prefix, "_sizeFactors"), sizeFactors(ours), sizeFactors(pkg))
  mo <- mcols(ours); mp <- mcols(pkg)
  if (!identical(names(mo), names(mp)))
    d$checks[[paste0(prefix, "_mcols_names")]] <- list(pass = FALSE, detail = paste(setdiff(union(names(mo), names(mp)), intersect(names(mo), names(mp))), collapse = ","))
  for (nm in intersect(names(mo), names(mp)))
    if (is.atomic(mp[[nm]])) d$check(paste0(prefix, "_", nm), mo[[nm]], mp[[nm]])
  for (a in assayNames(pkg)) d$check(paste0(prefix, "_assay_", a), assays(ours)[[a]], assays(pkg)[[a]])
  d$check(paste0(prefix, "_dispPriorVar"), attr(dispersionFunction(ours), "dispPriorVar"),
          attr(dispersionFunction(pkg), "dispPriorVar"))
}

# ---- plain-R closed forms ------------------------------------------------------
plain_size_factors <- function(cts) {
  lgm <- rowMeans(log(cts))
  apply(cts, 2, function(c) exp(median((log(c) - lgm)[is.finite(lgm) & c > 0])))
}

plain_norm_counts <- function(cts, sf) t(t(cts) / sf)

trimmed_cell_variance <- function(cnts, cells) {
  trimratio <- c(1/3, 1/4, 1/8)
  trimfn <- function(n) as.integer(cut(n, breaks = c(0, 3.5, 23.5, Inf)))
  cellMeans <- matrix(sapply(levels(cells), function(lvl) {
    n <- sum(cells == lvl)
    apply(cnts[, cells == lvl, drop = FALSE], 1, mean, trim = trimratio[trimfn(n)])
  }), nrow = nrow(cnts))
  qmat <- cellMeans[, as.integer(cells), drop = FALSE]
  sqerror <- (cnts - qmat)^2
  varEst <- matrix(sapply(levels(cells), function(lvl) {
    n <- sum(cells == lvl)
    scale.c <- c(2.04, 1.86, 1.51)[trimfn(n)]
    scale.c * apply(sqerror[, cells == lvl, drop = FALSE], 1, mean, trim = trimratio[trimfn(n)])
  }), nrow = nrow(sqerror))
  apply(varEst, 1, max)
}

# robustMethodOfMomentsDisp: the dispersion Cook's distance uses.
plain_robust_disp <- function(ncts, mm) {
  three <- n_or_more_in_cell(mm, 3)
  v <- if (any(three)) {
    cells <- apply(mm, 1, paste0, collapse = "")
    cells <- unname(factor(cells, levels = unique(cells)))
    levels(cells) <- seq_along(levels(cells))
    idx <- cells %in% levels(cells)[table(cells) >= 3]
    trimmed_cell_variance(ncts[, idx, drop = FALSE], factor(cells[idx]))
  } else {
    rm <- apply(ncts, 1, mean, trim = 1/8)
    1.51 * apply((ncts - rm)^2, 1, mean, trim = 1/8)
  }
  m <- rowMeans(ncts)
  pmax((v - m) / m^2, 0.04)
}

plain_cooks <- function(y, mu, H, ncts, mm) {
  disp <- plain_robust_disp(ncts, mm)
  V <- mu + disp * mu^2
  ((y - mu)^2 / V) / ncol(mm) * H / (1 - H)^2
}

plain_max_cooks <- function(cooks, mm) {
  s <- n_or_more_in_cell(mm, 3)
  if (nrow(mm) > ncol(mm) && any(s)) apply(cooks[, s, drop = FALSE], 1, max) else rep(NA, nrow(cooks))
}

# DESeq2:::estimateDispersionsPriorVar when m - p <= 3: under set.seed(2), for
# each of 200 grid variances x draw 1e4 of log(chisq_df) + N(0, x) - log(df),
# score its histogram on breaks -10..10 by 0.5 (hist(): right-closed, lowest
# included) by KL against the residuals' histogram, smooth the 200 KL values
# with loess(span = 0.2), and take the argmin over a 1000-point grid.
plain_prior_var_sim <- function(obs, df) {
  set.seed(2)
  brks <- -20:20 / 2
  obs <- obs[obs > min(brks) & obs < max(brks)]
  grid <- seq(from = 0, to = 8, length = 200)
  obs_hist <- hist(obs, breaks = brks, plot = FALSE)
  sim_counts <- matrix(0L, length(grid), length(brks) - 1, dimnames = list(NULL, paste0("bin", seq_len(length(brks) - 1))))
  kl <- numeric(length(grid))
  for (i in seq_along(grid)) {
    rd <- log(rchisq(10000, df = df)) + rnorm(10000, 0, sqrt(grid[i])) - log(df)
    rd <- rd[rd > min(brks) & rd < max(brks)]
    rh <- hist(rd, breaks = brks, plot = FALSE)
    sim_counts[i, ] <- rh$counts
    z <- c(obs_hist$density, rh$density)
    small <- min(z[z > 0])
    kl[i] <- sum(obs_hist$density * (log(obs_hist$density + small) - log(rh$density + small)))
  }
  lofit <- loess(kl ~ grid, span = 0.2)
  fine_grid <- seq(from = 0, to = 8, length = 1000)
  fine_fitted <- predict(lofit, fine_grid)
  j <- which.min(fine_fitted)
  list(obs = obs, breaks = brks, obs_hist = obs_hist, grid = grid, sim_counts = sim_counts, kl = kl,
       loess_fitted = unname(fitted(lofit)), fine_grid = fine_grid, fine_fitted = fine_fitted, j = j,
       argmin_kl = fine_grid[j], prior_var = pmax(fine_grid[j], 0.25))
}

# filtered_p + the rejection-curve rule of DESeq2:::pvalueAdjustment.
plain_independent_filtering <- function(filter, p, alpha) {
  lowerQuantile <- mean(filter == 0)
  upperQuantile <- if (lowerQuantile < 0.95) 0.95 else 1
  theta <- seq(lowerQuantile, upperQuantile, length = 50)
  cutoffs <- quantile(filter, theta)
  filtPadj <- matrix(NA_real_, length(p), length(theta))
  for (i in seq_along(cutoffs)) {
    use <- filter >= cutoffs[i]
    if (any(use)) filtPadj[use, i] <- plain_bh(p[use])
  }
  numRej <- colSums(filtPadj < alpha, na.rm = TRUE)
  lo.fit <- lowess(numRej ~ theta, f = 1/5)
  if (max(numRej) <= 10) {
    j <- 1
  } else {
    residual <- if (all(numRej == 0)) 0 else numRej[numRej > 0] - lo.fit$y[numRej > 0]
    maxFit <- max(lo.fit$y)
    thresh <- maxFit - sqrt(mean(residual^2))
    j <- if (any(numRej > thresh)) which(numRej > thresh)[1]
         else if (any(numRej > 0.9 * maxFit)) which(numRej > 0.9 * maxFit)[1]
         else if (any(numRej > 0.8 * maxFit)) which(numRej > 0.8 * maxFit)[1]
         else 1
  }
  list(padj = filtPadj[, j], theta = theta, cutoffs = unname(cutoffs), numRej = unname(numRej),
       lowess_y = lo.fit$y, j = j, threshold = unname(cutoffs[j]))
}

# ---- DESeq(): stage by stage ---------------------------------------------------
ref_deseq_fit <- function(dds, test, full, reduced, d) {
  cts <- counts(dds); ids <- rownames(cts)
  mm <- stats::model.matrix.default(design(dds), data = as.data.frame(colData(dds)))

  # Size factors: median-of-ratios.
  sf <- plain_size_factors(cts)
  dds <- estimateSizeFactors(dds, quiet = TRUE)
  d$check("deseq2_size_factors", sf, sizeFactors(dds))
  d$vec("deseq2_size_factors", colnames(cts), size_factor = sf)

  # Gene-wise dispersion (C++ Cox-Reid line search, then the grid refit).
  ncts <- plain_norm_counts(cts, sf)
  allZero <- rowSums(cts) == 0
  nz <- !allZero
  rough <- DESeq2:::roughDispEstimate(ncts[nz, , drop = FALSE], mm)
  xim <- mean(1 / sf)
  bm <- rowMeans(ncts); bv <- rowVars(ncts)
  moments <- ((bv - xim * bm) / bm^2)[nz]
  alpha_init <- pmin(pmax(DESEQ2_MIN_DISP, pmin(rough, moments)), max(10, ncol(cts)))
  dds <- estimateDispersionsGeneEst(dds, quiet = TRUE)
  d$check("deseq2_baseMean", bm, mcols(dds)$baseMean)
  d$check("deseq2_baseVar", bv, mcols(dds)$baseVar)
  d$check("deseq2_allZero", allZero, mcols(dds)$allZero)
  d$scalar("deseq2_linear_mu", nlevels(DESeq2:::modelMatrixGroups(mm)) == ncol(mm))

  # Parametric trend a + b/mean, with DESeq2's own local fallback on failure.
  dds <- estimateDispersionsFit(dds, fitType = "parametric", quiet = TRUE)
  dfun <- dispersionFunction(dds)
  fit_type <- attr(dfun, "fitType")
  d$scalar("deseq2_fit_type", fit_type)
  geneEst <- mcols(dds)$dispGeneEst
  if (identical(fit_type, "parametric")) {
    co <- attr(dfun, "coefficients")
    d$scalar("deseq2_trend_coefficients", unname(co))
    d$check("deseq2_dispFit", co[1] + co[2] / bm[nz], mcols(dds)$dispFit[nz])
  }
  above <- geneEst[nz] >= DESEQ2_MIN_DISP * 100
  resid <- log(geneEst[nz]) - log(mcols(dds)$dispFit[nz])
  varLog <- mad(resid[above], na.rm = TRUE)^2
  d$check("deseq2_varLogDispEsts", varLog, attr(dfun, "varLogDispEsts"))
  d$scalar("deseq2_varLogDispEsts", varLog)

  # Prior variance of log dispersion. With m - p <= 3 DESeq2 matches a
  # simulated distribution under set.seed(2), rebuilt in plain_prior_var_sim.
  m <- nrow(mm); p <- ncol(mm)
  rng_path <- (m - p) <= 3 && m > p
  priorVar <- if (rng_path) {
    sim <- plain_prior_var_sim(resid[above], m - p)
    d$check("deseq2_prior_var_sim", sim$prior_var, DESeq2:::estimateDispersionsPriorVar(dds, modelMatrix = mm))
    fwrite(data.table(lower = head(sim$breaks, -1), upper = tail(sim$breaks, -1), count = sim$obs_hist$counts,
                      density = sim$obs_hist$density), file.path(d$dir, "deseq2_prior_var_obs_hist.csv"))
    d$matrix("deseq2_prior_var_sim_counts", cbind(grid = sim$grid, sim$sim_counts), ids = NULL)
    fwrite(data.table(grid = sim$grid, kl = sim$kl, loess_fitted = sim$loess_fitted),
           file.path(d$dir, "deseq2_prior_var_kl.csv"))
    fwrite(data.table(fine_grid = sim$fine_grid, loess_predicted = sim$fine_fitted),
           file.path(d$dir, "deseq2_prior_var_fine.csv"))
    d$scalar("deseq2_prior_var_sim_n_obs", length(sim$obs))
    d$scalar("deseq2_prior_var_sim_argmin_index", sim$j)
    d$scalar("deseq2_prior_var_sim_argmin_kl", sim$argmin_kl)
    sim$prior_var
  } else if (m > p) pmax(varLog - trigamma((m - p) / 2), 0.25) else varLog
  d$scalar("deseq2_prior_var_rng_path", rng_path)
  d$scalar("deseq2_dispPriorVar", priorVar)

  # MAP dispersion (C++), then the outlier rule.
  dds <- estimateDispersionsMAP(dds, quiet = TRUE)
  d$check("deseq2_dispPriorVar", priorVar, attr(dispersionFunction(dds), "dispPriorVar"))
  mc <- mcols(dds)
  out <- log(mc$dispGeneEst) > log(mc$dispFit) + 2 * sqrt(varLog)
  out[is.na(out) & nz] <- FALSE
  disp_final <- ifelse(out, mc$dispGeneEst, mc$dispMAP)
  d$check("deseq2_dispOutlier", out, mc$dispOutlier)
  d$check("deseq2_dispersion", disp_final, mc$dispersion)
  dump_df(d, "deseq2_disp", data.frame(baseMean = mc$baseMean, baseVar = mc$baseVar, allZero = mc$allZero,
          alpha_init = replace(rep(NA_real_, length(ids)), nz, alpha_init),
          dispGeneEst = mc$dispGeneEst, dispGeneIter = mc$dispGeneIter, dispFit = mc$dispFit,
          dispMAP = mc$dispMAP, dispIter = mc$dispIter, dispOutlier = mc$dispOutlier,
          dispersion = mc$dispersion), ids)

  dds <- ref_test(dds, test, full, reduced, d, "deseq2_fit_initial")

  # Outlier replacement and refit, only when some cell has >= 7 samples.
  mmFit <- attr(dds, "modelMatrix")
  replaceable <- n_or_more_in_cell(mmFit, 7)
  d$scalar("deseq2_any_replaceable", any(replaceable))
  if (any(replaceable)) {
    p <- ncol(mmFit); m <- ncol(dds)
    cutoff <- qf(0.99, p, m - p)
    cooks <- assays(dds)[["cooks"]]
    replace <- apply(cooks, 1, function(row) any(row > cutoff))
    trimBaseMean <- apply(counts(dds, normalized = TRUE), 1, mean, trim = 0.2)
    replacement <- as.integer(outer(trimBaseMean, sizeFactors(dds), "*"))
    newCounts <- counts(dds)
    idx <- which(cooks > cutoff)
    newCounts[idx] <- replacement[idx]
    replaced <- counts(dds); replaced[, replaceable] <- newCounts[, replaceable, drop = FALSE]
    d$scalar("deseq2_replace_cooks_cutoff", cutoff)
    d$vec("deseq2_replaceable", colnames(cts), replaceable = replaceable)

    dds <- DESeq2:::refitWithoutOutliers(dds, test = test, betaPrior = FALSE, full = full,
                                         reduced = reduced, quiet = TRUE, minReplicatesForReplace = 7,
                                         modelMatrix = NULL, modelMatrixType = NULL)
    d$check("deseq2_replace", replace, mcols(dds)$replace)
    if (sum(replace, na.rm = TRUE) > 0) {
      d$check("deseq2_replaceCounts", replaced, assays(dds)[["replaceCounts"]])
      d$matrix("deseq2_replace_counts", assays(dds)[["replaceCounts"]], ids)
    }
    dump_df(d, "deseq2_fit_final", mcols(dds), ids)
    d$scalar("deseq2_n_replaced_genes", sum(replace, na.rm = TRUE))
  }
  dds
}

# nbinomWaldTest / nbinomLRT, with the closed-form parts checked.
ref_test <- function(dds, test, full, reduced, d, tag) {
  dds <- if (test == "Wald") nbinomWaldTest(dds, quiet = TRUE)
         else nbinomLRT(dds, full = full, reduced = reduced, quiet = TRUE)
  mc <- mcols(dds); nz <- !mc$allZero; ids <- rownames(dds)
  if (test == "Wald") {
    for (nm in resultsNames(dds)) {
      st <- mc[[nm]] / mc[[paste0("SE_", nm)]]
      d$check(paste0(tag, "_WaldStatistic_", nm), st, mc[[paste0("WaldStatistic_", nm)]])
      d$check(paste0(tag, "_WaldPvalue_", nm), 2 * pnorm(abs(st), lower.tail = FALSE), mc[[paste0("WaldPvalue_", nm)]])
    }
  } else {
    df <- ncol(attr(dds, "modelMatrix")) - ncol(attr(dds, "reducedModelMatrix"))
    d$scalar(paste0(tag, "_lrt_df"), df)
    d$check(paste0(tag, "_LRTPvalue"), pchisq(mc$LRTStatistic, df = df, lower.tail = FALSE), mc$LRTPvalue)
  }
  dmm <- attr(dds, "dispModelMatrix")
  y <- counts(dds)[nz, , drop = FALSE]
  cooks <- plain_cooks(y, assays(dds)[["mu"]][nz, , drop = FALSE], assays(dds)[["H"]][nz, , drop = FALSE],
                       counts(dds, normalized = TRUE)[nz, , drop = FALSE], dmm)
  d$check(paste0(tag, "_cooks"), cooks, assays(dds)[["cooks"]][nz, , drop = FALSE])
  d$check(paste0(tag, "_maxCooks"), plain_max_cooks(cooks, dmm), mc$maxCooks[nz])
  dump_df(d, tag, mc, ids)
  d$matrix(paste0(tag, "_cooks"), assays(dds)[["cooks"]], ids)
  dds
}

# ---- results() -----------------------------------------------------------------
# One results() call in plain R. `name` gives the coefficient path; `contrast`
# gives the cleanContrast path (coefficient, negated coefficient, or a
# difference of coefficients). Returns the table plus the filtering detail.
ref_results <- function(dds, name = NULL, contrast = NULL, independentFiltering, alpha) {
  test <- attr(dds, "test")
  mc <- mcols(dds); nz <- !mc$allZero
  coef_cols <- function(nm, sign = 1) {
    list(lfc = sign * mc[[nm]], se = mc[[paste0("SE_", nm)]],
         stat = if (test == "Wald") sign * mc[[paste0("WaldStatistic_", nm)]] else mc$LRTStatistic,
         p = if (test == "Wald") mc[[paste0("WaldPvalue_", nm)]] else mc$LRTPvalue)
  }
  path <- "coef"
  if (!is.null(contrast)) {
    f <- contrast[1]; num <- contrast[2]; den <- contrast[3]
    base <- levels(colData(dds)[[f]])[1]
    if (den == base) {
      r <- coef_cols(make.names(paste0(f, "_", num, "_vs_", den)))
    } else if (num == base) {
      r <- coef_cols(make.names(paste0(f, "_", den, "_vs_", num)), -1)
      path <- "negated_coef"
    } else {
      # getContrast: fitBeta at maxit 0, i.e. c'beta on the natural-log scale.
      cn <- make.names(paste0(f, "_", num, "_vs_", base)); cd <- make.names(paste0(f, "_", den, "_vs_", base))
      cv <- rep(0, length(resultsNames(dds))); cv[resultsNames(dds) == cn] <- 1; cv[resultsNames(dds) == cd] <- -1
      g <- DESeq2:::getContrast(dds, cv, useT = FALSE, minmu = 0.5)
      lfc_plain <- log2(exp(1)) * (log(2) * mc[[cn]] - log(2) * mc[[cd]])
      r <- list(lfc = g$log2FoldChange, se = g$lfcSE, stat = g$stat, p = g$pvalue, lfc_plain = lfc_plain)
      path <- "contrast"
      if (test == "LRT") { r$stat <- mc$LRTStatistic; r$p <- mc$LRTPvalue }
    }
    cts <- counts(dds)
    grp <- colData(dds)[[f]] %in% c(num, den)
    allZeroPair <- rowSums(cts[, grp, drop = FALSE] == 0) == sum(grp) & nz
    r$lfc[allZeroPair] <- 0; r$stat[allZeroPair] <- 0; r$p[allZeroPair] <- 1
    if (test == "LRT") { r$stat <- mc$LRTStatistic; r$p <- mc$LRTPvalue }
  } else {
    r <- coef_cols(if (is.null(name)) tail(resultsNames(dds), 1) else name)
  }

  # Cook's filter, with the rescue for a single two-level design factor.
  dmm <- attr(dds, "dispModelMatrix")
  cutoff <- qf(0.99, ncol(dmm), nrow(dmm) - ncol(dmm))
  outlier <- mc$maxCooks > cutoff
  rescued <- rep(FALSE, length(outlier))
  vars <- all.vars(design(dds))
  if (any(outlier, na.rm = TRUE) && length(vars) == 1) {
    v <- colData(dds)[[vars]]
    if (is.factor(v) && nlevels(v) == 2) {
      for (ii in which(outlier)) {
        outCount <- counts(dds)[ii, which.max(assays(dds)[["cooks"]][ii, ])]
        if (sum(counts(dds)[ii, ] > outCount) >= 3) rescued[ii] <- TRUE
      }
      outlier[rescued] <- FALSE
    }
  }
  r$p[which(outlier)] <- NA

  # Genes whose replaced counts are all zero.
  nowZero <- rep(FALSE, nrow(dds))
  if (sum(mc$replace, na.rm = TRUE) > 0) {
    nowZero[which(mc$replace & mc$baseMean == 0)] <- TRUE
    r$lfc[nowZero] <- 0; r$se[nowZero] <- 0; r$stat[nowZero] <- 0; r$p[nowZero] <- 1
  }

  inf <- NULL
  padj <- if (independentFiltering) {
    inf <- plain_independent_filtering(mc$baseMean, r$p, alpha)
    inf$padj
  } else plain_bh(r$p)
  list(table = data.frame(baseMean = mc$baseMean, log2FoldChange = r$lfc, lfcSE = r$se, stat = r$stat,
                          pvalue = r$p, padj = padj, row.names = rownames(dds)),
       cooks_cutoff = cutoff, cooks_outlier = outlier, cooks_rescued = rescued, now_zero = nowZero,
       filtering = inf, path = path, lfc_plain = r$lfc_plain)
}

check_results <- function(d, tag, ours, pkg) {
  md <- metadata(pkg)
  if (!is.null(ours$filtering)) {
    # The IF golden against what pvalueAdjustment stored on the package result.
    f <- ours$filtering
    d$check(paste0(tag, "_filter_threshold"), f$threshold, md$filterThreshold)
    d$check(paste0(tag, "_filter_theta"), f$theta[f$j], md$filterTheta)
    d$check(paste0(tag, "_filter_theta_grid"), f$theta, md$filterNumRej$theta)
    d$check(paste0(tag, "_filter_numRej"), f$numRej, md$filterNumRej$numRej)
    d$check(paste0(tag, "_filter_lowess_y"), f$lowess_y, md$lo.fit$y)
  }
  pkg <- as.data.frame(pkg)
  for (cc in c("baseMean", "log2FoldChange", "lfcSE", "stat", "pvalue", "padj"))
    d$check(paste0(tag, "_", cc), ours$table[[cc]], pkg[[cc]])
  if (!is.null(ours$lfc_plain)) d$check(paste0(tag, "_lfc_plain"), ours$lfc_plain, ours$table$log2FoldChange)
  ids <- rownames(ours$table)
  dump_df(d, tag, cbind(ours$table, cooks_outlier = ours$cooks_outlier, cooks_rescued = ours$cooks_rescued,
                        now_zero = ours$now_zero), ids)
  d$scalar(paste0(tag, "_path"), ours$path)
  d$scalar(paste0(tag, "_cooks_cutoff"), ours$cooks_cutoff)
  if (!is.null(ours$filtering)) {
    f <- ours$filtering
    fwrite(data.table(theta = f$theta, cutoff = f$cutoffs, numRej = f$numRej, lowess_y = f$lowess_y),
           file.path(d$dir, paste0(tag, "_filtering.csv")))
    d$scalar(paste0(tag, "_filter_j"), f$j)
    d$scalar(paste0(tag, "_filter_threshold"), f$threshold)
    d$scalar(paste0(tag, "_filter_theta"), f$theta[f$j])
  }
}

# ---- the engine ------------------------------------------------------------------
ref_deseq2 <- function(inp, d) {
  withr::local_preserve_seed()
  cond <- inp$conditionCol; ctl <- inp$controlCols
  si <- inp$sampleInfo; p <- inp$params
  alpha <- if (is.null(p$deseq2_alpha)) 0.05 else p$deseq2_alpha
  shrink <- if (is.null(p$deseq2_lfc_shrinkage)) "none" else p$deseq2_lfc_shrinkage
  apeglm_seed <- if (is.null(p$apeglm_seed)) 1L else p$apeglm_seed
  has_ctl <- !is.null(ctl) && length(ctl) > 0
  full <- as.formula(paste("~", cond, if (has_ctl) paste("+", paste(ctl, collapse = " + ")) else ""))
  reduced <- if (has_ctl) as.formula(paste("~", paste(ctl, collapse = " + "))) else ~ 1
  design <- model.matrix(full, data = si)

  y0 <- inp$countMatrix[, rownames(si), drop = FALSE]
  flt <- plain_filter_by_expr(y0, design)
  d$check("filter_keep", flt$keep, edgeR::filterByExpr(DGEList(counts = y0), design = design))
  d$vec("deseq2_filter", rownames(y0), keep = flt$keep, n_above_cutoff = flt$n_above, total_count = flt$total)
  d$scalar("filter_min_sample_size", flt$min_sample_size)
  d$scalar("filter_cpm_cutoff", flt$cpm_cutoff)
  if (sum(flt$keep) == 0)
    stop(md_error("filterByExpr removed every gene. Check input count matrix and sample-size per condition."))
  y <- inp$countMatrix[flt$keep, , drop = FALSE]
  checkMatrixRank(designMat = design, predictors = c(cond, ctl))

  dds0 <- DESeqDataSetFromMatrix(countData = y, colData = si, design = full)
  anova <- identical(inp$mode, "anova")
  test <- if (anova) "LRT" else "Wald"
  dds_pkg <- if (anova) DESeq(dds0, test = "LRT", reduced = reduced, quiet = TRUE) else DESeq(dds0, quiet = TRUE)
  dds <- ref_deseq_fit(dds0, test, full, if (anova) reduced else NULL, d)
  check_dds(d, "deseq2_vs_DESeq", dds, dds_pkg)

  aveExpr <- mcols(dds)$baseMean
  d$check("deseq2_aveExpr", aveExpr, results(dds, independentFiltering = FALSE)$baseMean)
  ids <- rownames(dds)
  enc <- inp$encoded; cdf <- inp$comparisonDF

  if (anova) {
    out <- data.table(GroupId = ids)
    for (i in seq_len(nrow(enc))) {
      tag <- sprintf("deseq2_anova_lfc_%02d", i)
      r <- ref_results(dds, contrast = c(cond, enc$left[i], enc$right[i]), independentFiltering = FALSE, alpha = 0.1)
      check_results(d, tag, r, results(dds, contrast = c(cond, enc$left[i], enc$right[i]), independentFiltering = FALSE))
      out[[paste0("Log2FC ", cdf$left[i], " - ", cdf$right[i])]] <- r$table$log2FoldChange
    }
    r <- ref_results(dds, independentFiltering = TRUE, alpha = alpha)
    check_results(d, "deseq2_anova_omnibus", r, results(dds, independentFiltering = TRUE, alpha = alpha))
    out[, `:=`(AveExpr = r$table$baseMean, LRT = r$table$stat, PValue = r$table$pvalue, AdjPValue = r$table$padj)]
    return(as.data.frame(out))
  }

  stats <- data.table(GroupId = ids)
  for (i in seq_len(nrow(enc))) {
    left <- enc$left[i]; right <- enc$right[i]
    lab <- paste0(cdf$left[i], " - ", cdf$right[i])
    tag <- sprintf("deseq2_cmp_%02d", i)
    ddsShrink <- dds
    relevelled <- levels(colData(ddsShrink)[[cond]])[1] != right
    d$scalar(paste0(tag, "_relevel_refit"), relevelled)
    if (relevelled) {
      colData(ddsShrink)[[cond]] <- relevel(colData(ddsShrink)[[cond]], ref = right)
      ddsShrink <- ref_test(ddsShrink, "Wald", NULL, NULL, d, paste0(tag, "_relevel_fit"))
    }
    coefName <- paste0(cond, "_", left, "_vs_", right)
    if (!coefName %in% resultsNames(ddsShrink))
      stop(md_error(paste0("DESeq2 results: coefficient '", coefName, "' not found in resultsNames after releveling. ",
                           "Available: ", paste(resultsNames(ddsShrink), collapse = ", "), ".")))
    r <- ref_results(ddsShrink, name = coefName, independentFiltering = TRUE, alpha = alpha)
    check_results(d, tag, r, results(ddsShrink, name = coefName, independentFiltering = TRUE, alpha = alpha))
    lfc <- r$table$log2FoldChange; se <- r$table$lfcSE
    ci_l <- lfc - qnorm(0.975) * se; ci_r <- lfc + qnorm(0.975) * se
    cr_l <- cr_r <- rep(NA_real_, length(ids))

    if (shrink == "normal") {
      # lfcShrink(type = "normal"): weighted upper-quantile beta prior, then a
      # penalised refit from the MLE mu and H.
      dn <- ddsShrink
      bc <- grep("log2 fold change \\(MLE\\)", mcols(mcols(dn))$description)
      names(mcols(dn))[bc] <- paste0("MLE_", names(mcols(dn))[bc])
      attr(dn, "modelMatrixType") <- "standard"
      bpv <- DESeq2:::estimateBetaPriorVar(dn, modelMatrix = NULL)
      d$scalar(paste0(tag, "_beta_prior_var"), as.list(bpv))
      dshr <- nbinomWaldTest(dn, betaPrior = TRUE, betaPriorVar = bpv, modelMatrix = NULL,
                             modelMatrixType = "standard", quiet = TRUE)
      rs <- ref_results(dshr, name = coefName, independentFiltering = TRUE, alpha = alpha)
      pkg <- as.data.frame(lfcShrink(ddsShrink, coef = coefName, type = "normal", quiet = TRUE))
      d$check(paste0(tag, "_shrunk_lfc"), rs$table$log2FoldChange, pkg$log2FoldChange)
      d$check(paste0(tag, "_shrunk_se"), rs$table$lfcSE, pkg$lfcSE)
      d$vec(paste0(tag, "_shrunk"), ids, log2FoldChange = rs$table$log2FoldChange, lfcSE = rs$table$lfcSE)
      lfc <- rs$table$log2FoldChange; se <- rs$table$lfcSE
      cr_l <- lfc - qnorm(0.975) * se; cr_r <- lfc + qnorm(0.975) * se
    } else if (shrink == "ashr") {
      # Out of scope for the Rust v1 port (G1): called exactly as production does.
      s <- as.data.frame(lfcShrink(ddsShrink, coef = coefName, type = "ashr", quiet = TRUE))
      lfc <- s$log2FoldChange; se <- s$lfcSE
      cr_l <- lfc - qnorm(0.975) * se; cr_r <- lfc + qnorm(0.975) * se
    } else if (shrink == "apeglm") {
      s <- withr::with_seed(apeglm_seed, lfcShrink(ddsShrink, coef = coefName, type = "apeglm",
                                                   returnList = TRUE, quiet = TRUE))
      lfc <- as.data.frame(s$res)$log2FoldChange; se <- as.data.frame(s$res)$lfcSE
      cr_l <- log2(exp(1)) * s$fit$interval[, 1]; cr_r <- log2(exp(1)) * s$fit$interval[, 2]
    }

    stats[[paste0("Log2FC ", lab)]]    <- lfc
    stats[[paste0("stat ", lab)]]      <- r$table$stat
    stats[[paste0("SE ", lab)]]        <- se
    stats[[paste0("CILeft ", lab)]]    <- ci_l
    stats[[paste0("CIRight ", lab)]]   <- ci_r
    stats[[paste0("CrILeft ", lab)]]   <- cr_l
    stats[[paste0("CrIRight ", lab)]]  <- cr_r
    stats[[paste0("PValue ", lab)]]    <- r$table$pvalue
    stats[[paste0("AdjPValue ", lab)]] <- r$table$padj
  }
  stats$AveExpr <- aveExpr
  as.data.frame(stats)
}
