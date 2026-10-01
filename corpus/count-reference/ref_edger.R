# edgeR QL reference (edgeR 4.8.2, legacy = FALSE), as MDFlexiComparisons runs it:
# filterByExpr -> calcNormFactors -> estimateDisp -> glmQLFit -> glmQLFTest per contrast.
#
# Each step is rebuilt from the package internals in the order the package runs
# them, every intermediate is dumped, and each rebuilt step is checked against
# the corresponding exported edgeR function (d$check). The C++ kernels
# (glmFit IRLS, adjusted profile likelihood, ave_qd, adj_vec, aveLogCPM) are
# called directly: those are what PR-E3 ports, and their inputs and outputs
# are now goldens.

plain_tmm <- function(obs, ref, nO, nR, logratioTrim = 0.3, sumTrim = 0.05) {
  obs <- as.numeric(obs); ref <- as.numeric(ref)
  logR <- log2((obs / nO) / (ref / nR))
  absE <- (log2(obs / nO) + log2(ref / nR)) / 2
  v <- (nO - obs) / nO / obs + (nR - ref) / nR / ref
  fin <- is.finite(logR) & is.finite(absE) & (absE > -1e10)
  logR <- logR[fin]; absE <- absE[fin]; v <- v[fin]
  if (max(abs(logR)) < 1e-6) return(1)
  n <- length(logR)
  loL <- floor(n * logratioTrim) + 1; hiL <- n + 1 - loL
  loS <- floor(n * sumTrim) + 1;      hiS <- n + 1 - loS
  keep <- (rank(logR) >= loL & rank(logR) <= hiL) & (rank(absE) >= loS & rank(absE) <= hiS)
  f <- sum(logR[keep] / v[keep], na.rm = TRUE) / sum(1 / v[keep], na.rm = TRUE)
  if (is.na(f)) f <- 0
  2^f
}

plain_norm_factors <- function(x, lib, method) {
  x <- x[rowSums(x > 0) != 0, , drop = FALSE]
  ns <- ncol(x)
  if (nrow(x) == 0 || ns == 1) method <- "none"
  q75 <- function() apply(x, 2, function(col) quantile(col, probs = 0.75)) / lib
  ref <- NA_integer_
  f <- switch(method,
    TMM = {
      f75 <- suppressWarnings(q75())
      ref <- if (median(f75) < 1e-20) which.max(colSums(sqrt(x))) else which.min(abs(f75 - mean(f75)))
      vapply(seq_len(ns), function(i) plain_tmm(x[, i], x[, ref], lib[i], lib[ref]), 0)
    },
    RLE = {
      gm <- exp(rowMeans(log(x)))
      apply(x, 2, function(u) median((u / gm)[gm > 0])) / lib
    },
    upperquartile = q75(),
    none = rep_len(1, ns),
    stop("plain_norm_factors: no plain-R recompute for ", method))
  f <- unname(f / exp(mean(log(f))))
  list(norm_factors = f, ref_column = unname(ref))
}

# estimateDisp.default with a design, rebuilt step by step.
ref_estimate_disp <- function(y, design, lib_eff, offset, d) {
  ntags <- nrow(y); nlibs <- ncol(y)
  weights <- edgeR:::.compressWeights(y, NULL)
  sel <- rowSums(y) >= 5
  sely <- y[sel, , drop = FALSE]; seloffset <- offset[sel, , drop = FALSE]
  selweights <- weights[sel, , drop = FALSE]
  spline.pts <- seq(-10, 10, length.out = 21)
  spline.disp <- 0.1 * 2^spline.pts
  l0 <- matrix(0, sum(sel), 21)
  glmfit <- edgeR::glmFit(sely, design, offset = seloffset, dispersion = 0.05, prior.count = 0)
  zerofit <- glmfit$counts < 1e-4 & glmfit$fitted.values < 1e-4
  by.group <- edgeR:::.comboGroups(zerofit)
  for (subg in by.group) {
    cur.nzero <- !zerofit[subg[1], ]
    if (!any(cur.nzero)) next
    if (all(cur.nzero)) {
      redesign <- design
    } else {
      redesign <- design[cur.nzero, , drop = FALSE]
      QR <- qr(redesign)
      redesign <- redesign[, QR$pivot[1:QR$rank], drop = FALSE]
      if (nrow(redesign) == ncol(redesign)) next
    }
    cury <- sely[subg, cur.nzero, drop = FALSE]
    curo <- seloffset[subg, cur.nzero, drop = FALSE]
    curw <- selweights[subg, cur.nzero, drop = FALSE]
    last.beta <- NULL
    for (i in seq_len(21)) {
      out <- edgeR::adjustedProfileLik(spline.disp[i], y = cury, design = redesign, offset = curo,
                                       weights = curw, start = last.beta, get.coef = TRUE)
      l0[subg, i] <- out$apl
      last.beta <- out$beta
    }
  }
  overall <- edgeR::maximizeInterpolant(spline.pts, matrix(colSums(l0), nrow = 1))
  common <- 0.1 * 2^overall
  ave_common <- edgeR::aveLogCPM(y, lib.size = lib_eff, dispersion = common, weights = weights)
  out.1 <- edgeR::WLEB(theta = spline.pts, loglik = l0, covariate = ave_common[sel], trend.method = "locfit",
                       span = NULL, legacy.span = FALSE, overall = FALSE, individual = FALSE, m0.out = TRUE)
  span <- out.1$span; m0 <- out.1$shared.loglik
  disp.trend <- 0.1 * 2^out.1$trend
  trended <- rep(disp.trend[which.min(ave_common[sel])], ntags)
  trended[sel] <- disp.trend
  glmfit2 <- edgeR::glmFit(sely, offset = seloffset, weights = selweights, design = design,
                           dispersion = disp.trend, prior.count = 0)
  zerofit2 <- glmfit2$counts < 1e-4 & glmfit2$fitted.values < 1e-4
  df.residual <- edgeR:::.residDF(zerofit2, design)
  s2 <- glmfit2$deviance / df.residual
  s2[df.residual == 0] <- 0
  s2 <- pmax(s2, 0)
  # legacy is not passed here, so squeezeVar picks it from whether the df are equal.
  dfp <- df.residual[df.residual > 0]
  sv_legacy <- identical(min(dfp), max(dfp))
  s2.fit <- limma::squeezeVar(s2, df = df.residual, covariate = ave_common[sel], robust = FALSE)
  prior.df <- s2.fit$df.prior
  prior.n <- prior.df / (nlibs - ncol(design))
  tagwise <- trended
  out.2 <- edgeR::WLEB(theta = spline.pts, loglik = l0, prior.n = min(prior.n, 1e6), covariate = ave_common[sel],
                       trend.method = "locfit", span = span, legacy.span = FALSE, overall = FALSE,
                       trend = FALSE, m0 = m0)
  if (prior.n <= 1e6) tagwise[sel] <- 0.1 * 2^out.2$individual

  ids <- rownames(y)
  d$vec("edger_disp_sel", ids, sel = sel)
  d$matrix("edger_disp_l0", l0, ids[sel])
  d$matrix("edger_disp_m0", m0, ids[sel])
  d$vec("edger_disp_glmfit005", ids[sel], deviance = glmfit$deviance)
  d$vec("edger_disp_prior_df_inputs", ids[sel], deviance = glmfit2$deviance, df_residual = df.residual, s2 = s2)
  d$vec("edger_disp", ids, ave_logcpm_common = ave_common, trended = trended, tagwise = tagwise)
  d$scalar("edger_disp_common", common)
  d$scalar("edger_disp_overall_log2", overall)
  d$scalar("edger_disp_span", span)
  d$scalar("edger_disp_prior_df", prior.df)
  d$scalar("edger_disp_prior_n", prior.n)
  d$scalar("edger_disp_squeezevar_legacy", sv_legacy)
  d$scalar("edger_disp_squeezevar_var_prior", unname(s2.fit$var.prior))
  list(common = common, trended = trended, tagwise = tagwise, prior.df = prior.df, prior.n = prior.n,
       span = span, ave_common = ave_common)
}

# glmQLFit.DGEList -> glmQLFit.default with legacy = FALSE, rebuilt.
ref_ql_fit <- function(y, design, offset, ave, trended, d) {
  weights <- edgeR:::.compressWeights(y, NULL)
  ntop <- ceiling(0.1 * nrow(y))
  top <- order(ave, decreasing = TRUE)[1:ntop]
  disp_raw <- mean(trended[top])
  disp <- min(disp_raw, 4)
  fit0 <- edgeR::glmFit(y, design = design, dispersion = disp, offset = offset, lib.size = NULL, weights = weights)
  disp.mat <- edgeR:::.compressDispersions(y, disp)
  aqd <- .Call(edgeR:::.cxx_compute_ave_qd, y, fit0$fitted.values, design, disp.mat, ave, weights)
  fit <- edgeR::glmFit(y, design = design, dispersion = disp / aqd, offset = offset, lib.size = NULL, weights = weights)
  adj <- .Call(edgeR:::.cxx_compute_adj_vec, y, fit$fitted.values, design, disp.mat, aqd, weights)
  s2 <- adj$s2
  s2_in <- s2; s2_in[adj$df == 0] <- 0
  fd <- limma:::fitFDistUnequalDF1(s2_in, df1 = adj$df, covariate = ave, span = NULL, robust = FALSE)
  sv <- limma::squeezeVar(s2, df = adj$df, covariate = ave, robust = FALSE, legacy = FALSE)

  ids <- rownames(y)
  d$vec("edger_ql_fit", ids, ave_logcpm = ave, deviance_first = fit0$deviance,
        deviance = fit$deviance, df_residual = fit$df.residual,
        s2 = s2, df_residual_adj = adj$df, deviance_adj = adj$deviance,
        s2_prior = sv$var.prior, s2_post = sv$var.post, df_prior = sv$df.prior,
        fdist_scale = fd$scale, fdist_df2 = fd$df2,
        fdist_df2_shrunk = if (is.null(fd$df2.shrunk)) NA_real_ else fd$df2.shrunk)
  d$matrix("edger_ql_coefficients", fit$coefficients, ids)
  d$scalar("edger_ql_top_n", ntop)
  d$scalar("edger_ql_dispersion_uncapped", disp_raw)
  d$scalar("edger_ql_dispersion", disp)
  d$scalar("edger_ql_dispersion_capped", disp_raw > 4)
  d$scalar("edger_ql_ave_ql_dispersion", aqd)
  list(fit = fit, disp = disp, aqd = aqd, df_adj = adj$df, s2_post = sv$var.post,
       df_prior = sv$df.prior, s2_prior = sv$var.prior)
}

# glmLRT reparametrisation + glmQLFTest F, for one contrast matrix (1 or more columns).
ref_ql_test <- function(y, design, offset, ql, contrast, label, d) {
  fit <- ql$fit
  nlibs <- ncol(y)
  contrast <- as.matrix(contrast)
  qrc <- qr(contrast)
  coef <- seq_len(qrc$rank)
  logFC <- (fit$coefficients %*% contrast) / log(2)
  Dvec <- rep_len(1, nlibs)
  Dvec[coef] <- diag(qrc$qr)[coef]
  Q <- qr.Q(qrc, complete = TRUE, Dvec = Dvec)
  design0 <- (design %*% Q)[, -coef, drop = FALSE]
  null <- edgeR::glmFit(y, design = design0, offset = offset, weights = fit$weights,
                        dispersion = ql$disp / ql$aqd, prior.count = 0)
  LR <- null$deviance - fit$deviance
  df.test <- null$df.residual - fit$df.residual
  F <- LR / df.test / ql$s2_post
  df.total <- pmin(ql$df_prior + ql$df_adj, sum(fit$df.residual))
  p <- pf(F, df1 = df.test, df2 = df.total, lower.tail = FALSE)
  fdr <- plain_bh(p)
  if (ncol(logFC) == 1) logFC <- drop(logFC)
  ids <- rownames(y)
  tab <- data.table(id = ids, deviance_null = null$deviance, LR = LR, df_test = df.test,
                    F = F, df_total = df.total, PValue = p, FDR = fdr)
  if (is.matrix(logFC)) for (k in seq_len(ncol(logFC))) tab[[paste0("logFC_", k)]] <- logFC[, k] else tab$logFC <- logFC
  fwrite(tab, file.path(d$dir, paste0("edger_test_", label, ".csv")), na = "NA")
  list(logFC = logFC, F = F, PValue = p, FDR = fdr, df_total = df.total)
}

ref_edger <- function(inp, d) {
  si <- inp$sampleInfo; cc <- inp$conditionCol; ctl <- inp$controlCols
  method <- inp$params$edger_norm_method
  f <- if (length(ctl) > 0) paste("~ 0 +", cc, "+", paste(ctl, collapse = " + ")) else paste("~ 0 +", cc)
  design <- model.matrix(as.formula(f), data = si)
  checkMatrixRank(designMat = design, predictors = c(cc, ctl))
  y0 <- inp$countMatrix[, rownames(si), drop = FALSE]
  d$matrix("edger_design", design)

  # filterByExpr
  fb <- plain_filter_by_expr(y0, design)
  d$check("filterByExpr.keep", fb$keep, edgeR::filterByExpr(edgeR::DGEList(counts = y0), design = design))
  d$vec("edger_filter", rownames(y0), n_above_cutoff = fb$n_above, total = fb$total, keep = fb$keep)
  d$scalar("edger_filter_min_sample_size", fb$min_sample_size)
  d$scalar("edger_filter_cpm_cutoff", fb$cpm_cutoff)
  if (sum(fb$keep) == 0)
    stop(md_error("filterByExpr removed every gene. Check input count matrix and sample-size per condition."))

  dge <- edgeR::DGEList(counts = y0)[fb$keep, , keep.lib.sizes = FALSE]
  y <- dge$counts
  lib <- dge$samples$lib.size
  d$check("lib_size", lib, colSums(y0[fb$keep, , drop = FALSE]))

  # calcNormFactors
  nf <- if (method == "TMMwsp") list(norm_factors = unname(edgeR::calcNormFactors(y, lib.size = lib, method = method)), ref_column = NA)
        else plain_norm_factors(y, lib, method)
  dge <- edgeR::calcNormFactors(dge, method = method)
  d$check("calcNormFactors", nf$norm_factors, dge$samples$norm.factors)
  d$vec("edger_norm", colnames(y), lib_size = lib, norm_factor = nf$norm_factors)
  d$scalar("edger_norm_method", method)
  d$scalar("edger_norm_ref_column", nf$ref_column)
  lib_eff <- lib * dge$samples$norm.factors
  offset <- edgeR:::.compressOffsets(y, lib.size = lib_eff, offset = edgeR::getOffset(dge))

  if (ncol(design) >= ncol(y)) {
    # No residual df. estimateDisp warns and returns NA dispersions, and
    # production then fails inside glmQLFit; reproduce that failure as is.
    dge <- edgeR::estimateDisp(dge, design = design)
    edgeR::glmQLFit(dge, design)
    stop("reference: expected glmQLFit to fail with no residual df")
  }

  disp <- ref_estimate_disp(y, design, lib_eff, offset, d)
  dge_pkg <- edgeR::estimateDisp(dge, design = design)
  d$check("estimateDisp.common", disp$common, dge_pkg$common.dispersion)
  d$check("estimateDisp.trended", disp$trended, dge_pkg$trended.dispersion)
  d$check("estimateDisp.tagwise", disp$tagwise, dge_pkg$tagwise.dispersion)
  d$check("estimateDisp.prior_df", disp$prior.df, dge_pkg$prior.df)
  d$check("estimateDisp.AveLogCPM", disp$ave_common, dge_pkg$AveLogCPM)

  ql <- ref_ql_fit(y, design, offset, disp$ave_common, disp$trended, d)
  fit_pkg <- edgeR::glmQLFit(dge_pkg, design)
  d$check("glmQLFit.coefficients", ql$fit$coefficients, fit_pkg$coefficients)
  d$check("glmQLFit.df_prior", ql$df_prior, fit_pkg$df.prior)
  d$check("glmQLFit.s2_post", ql$s2_post, fit_pkg$s2.post)
  d$check("glmQLFit.s2_prior", ql$s2_prior, fit_pkg$s2.prior)
  d$check("glmQLFit.df_residual_adj", ql$df_adj, fit_pkg$df.residual.adj)
  d$check("glmQLFit.ave_ql_dispersion", ql$aqd, fit_pkg$average.ql.dispersion)

  ids <- rownames(y)
  ave <- disp$ave_common
  # Omnibus: K-1 treatment contrasts against the first level.
  lv <- levels(si[[cc]])
  if (length(lv) < 2) stop(md_error("edgeR requires at least 2 condition levels."))
  omni_c <- limma::makeContrasts(contrasts = paste0(cc, lv[-1], " - ", cc, lv[1]), levels = design)
  d$matrix("edger_contrast_omnibus", omni_c)
  om <- ref_ql_test(y, design, offset, ql, omni_c, "omnibus", d)
  om_pkg <- edgeR::glmQLFTest(fit_pkg, contrast = omni_c)
  d$check("glmQLFTest.omnibus.F", om$F, om_pkg$table$F)
  d$check("glmQLFTest.omnibus.PValue", om$PValue, om_pkg$table$PValue)
  stats <- data.table(GroupId = ids, AveExpr = ave, F = om$F, PValue = om$PValue, AdjPValue = om$FDR)

  # Pairwise. The CI df is the production one, df.prior + df.residual (unadjusted),
  # which is not the df.total the p-value uses: replicated defect.
  df_ci <- fit_pkg$df.prior + if (!is.null(fit_pkg$df.residual.zeros)) fit_pkg$df.residual.zeros else fit_pkg$df.residual
  d$vec("edger_ci_df", ids, df_ci = df_ci)
  for (i in seq_len(nrow(inp$encoded))) {
    le <- inp$encoded$left[i]; re <- inp$encoded$right[i]
    lab <- paste0(inp$comparisonDF$left[i], " - ", inp$comparisonDF$right[i])
    con <- limma::makeContrasts(contrasts = paste0(cc, le, " - ", cc, re), levels = design)
    pw <- ref_ql_test(y, design, offset, ql, con, paste0("pair", i), d)
    pw_pkg <- edgeR::glmQLFTest(fit_pkg, contrast = con)
    d$check(paste0("glmQLFTest.pair", i, ".F"), pw$F, pw_pkg$table$F)
    d$check(paste0("glmQLFTest.pair", i, ".logFC"), pw$logFC, pw_pkg$table$logFC)
    F_vec <- pw$F
    usable <- is.finite(F_vec) & F_vec > 0
    t_stat <- suppressWarnings(ifelse(usable, sqrt(F_vec) * sign(pw$logFC), NA_real_))
    se_val <- ifelse(usable, pw$logFC / t_stat, NA_real_)
    ci_half <- qt(0.975, df = df_ci) * se_val
    nm <- function(x) paste0(x, " ", lab)
    stats[[nm("Log2FC")]]    <- pw$logFC
    stats[[nm("stat")]]      <- t_stat
    stats[[nm("SE")]]        <- se_val
    stats[[nm("CILeft")]]    <- ifelse(usable, pw$logFC - ci_half, NA_real_)
    stats[[nm("CIRight")]]   <- ifelse(usable, pw$logFC + ci_half, NA_real_)
    stats[[nm("F")]]         <- pw$F
    stats[[nm("PValue")]]    <- pw$PValue
    stats[[nm("AdjPValue")]] <- pw$FDR
  }
  stats
}
