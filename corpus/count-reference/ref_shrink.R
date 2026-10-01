# DESeq2 1.50.2 lfcShrink(type = "apeglm" | "ashr") rebuilt step by step, for the
# shrink-core port (apeglm 1.32.0, ashr 2.2-63, mixsqp 0.3-54).
#
# For every comparison of a shrink run it rebuilds ddsShrink exactly as production
# does (deseq2StatsFun.R: DESeq, relevel + nbinomWaldTest, results(name = coef)),
# dumps the plain inputs the Rust API takes (counts, size factors, dispersions,
# post-relevel design, coef index, MLE lfc / SE), then replays the package internals
# in order and dumps each intermediate:
#
#   apeglm  priorVar (Efron-Morris + uniroot), the nbinomCR prefit (cnst, both LBFGSpp
#           fits, delta, conv), the per-row R pass (cnst2, optimHess, var.est, the
#           optim(BFGS) fallback), map / sd / interval / fsr / svalue / diag, plus the
#           LBFGSpp evaluation path for a subset of genes (nbinom_trace.cpp).
#   ashr    exclusions, mixsd grid, likelihood matrix, mixsqp (tsvd decision, 20 EM
#           steps, SQP result, progress, gradient and Hessian at the solution), fitted
#           pi, pruned pi, loglik, PosteriorMean / SD, NegativeProb, lfsr, svalue.
#
# Every replayed intermediate is checked against the package function and the final
# shrunk lfc / SE against lfcShrink() itself.


# Every shrink dump is written at 17 significant digits (fwrite's default is 15),
# so the Rust port reads its inputs bit-exactly and gaps below 1e-15 are visible.
fw17 <- function(dt, path, ...) {
  dt <- as.data.table(dt)
  for (cc in names(dt)) if (is.double(dt[[cc]]))
    set(dt, j = cc, value = ifelse(is.na(dt[[cc]]), NA_character_, sprintf("%.17g", dt[[cc]])))
  fwrite(dt, path, na = "NA", quote = FALSE)
}
vec17 <- function(d, name, ids, ...) fw17(data.table(id = ids, ...), file.path(d$dir, paste0(name, ".csv")))
mat17 <- function(d, name, m, ids = rownames(m)) {
  dt <- as.data.table(m)
  if (!is.null(ids)) dt <- cbind(data.table(id = ids), dt)
  fw17(dt, file.path(d$dir, paste0(name, ".csv")))
}

SHRINK_TRACE_N <- 40L  # genes per comparison whose full optimizer path is dumped

Rcpp::sourceCpp(file.path(here, "nbinom_trace.cpp"), cacheDir = file.path(tempdir(), "nbt"))

shrink_inputs_dds <- function(inp) {
  cond <- inp$conditionCol; ctl <- inp$controlCols; si <- inp$sampleInfo
  has_ctl <- !is.null(ctl) && length(ctl) > 0
  full <- as.formula(paste("~", cond, if (has_ctl) paste("+", paste(ctl, collapse = " + ")) else ""))
  design <- model.matrix(full, data = si)
  y0 <- inp$countMatrix[, rownames(si), drop = FALSE]
  flt <- plain_filter_by_expr(y0, design)
  y <- inp$countMatrix[flt$keep, , drop = FALSE]
  dds0 <- DESeqDataSetFromMatrix(countData = y, colData = si, design = full)
  DESeq(dds0, quiet = TRUE)
}

ref_shrink <- function(inp, d) {
  withr::local_preserve_seed()
  p <- inp$params; cond <- inp$conditionCol
  shrink <- if (is.null(p$deseq2_lfc_shrinkage)) "none" else p$deseq2_lfc_shrinkage
  apeglm_seed <- if (is.null(p$apeglm_seed)) 1L else p$apeglm_seed
  d$scalar("shrink_type", shrink)
  d$scalar("shrink_apeglm_seed", apeglm_seed)
  if (identical(inp$mode, "anova") || shrink == "none") return(invisible(NULL))
  dds <- shrink_inputs_dds(inp)
  ids <- rownames(dds)
  mat17(d, "shrink_counts", counts(dds))
  vec17(d, "shrink_size_factors", colnames(dds), size_factor = unname(sizeFactors(dds)))
  enc <- inp$encoded; cdf <- inp$comparisonDF
  out <- list()
  for (i in seq_len(nrow(enc))) {
    left <- enc$left[i]; right <- enc$right[i]
    tag <- sprintf("shrink_cmp_%02d", i)
    ddsShrink <- dds
    if (levels(colData(ddsShrink)[[cond]])[1] != right) {
      colData(ddsShrink)[[cond]] <- relevel(colData(ddsShrink)[[cond]], ref = right)
      ddsShrink <- nbinomWaldTest(ddsShrink, quiet = TRUE)
    }
    coefName <- paste0(cond, "_", left, "_vs_", right)
    coefNum <- which(resultsNames(ddsShrink) == coefName)
    res <- results(ddsShrink, name = coefName)
    X <- model.matrix(design(ddsShrink), data = colData(ddsShrink))
    mat17(d, paste0(tag, "_design"), X, ids = rownames(X))
    vec17(d, paste0(tag, "_input"), ids, dispersion = unname(dispersions(ddsShrink)),
          lfc_mle = res$log2FoldChange, lfc_se = res$lfcSE)
    d$scalar(paste0(tag, "_coef_index"), coefNum)
    d$scalar(paste0(tag, "_coef_name"), coefName)
    d$scalar(paste0(tag, "_label"), paste0(cdf$left[i], " - ", cdf$right[i]))
    d$scalar(paste0(tag, "_design_colnames"), colnames(X))
    if (shrink == "apeglm") {
      s <- withr::with_seed(apeglm_seed, lfcShrink(ddsShrink, coef = coefName, type = "apeglm",
                                                   returnList = TRUE, quiet = TRUE))
      out[[i]] <- ref_apeglm(d, tag, ddsShrink, X, res, coefNum, s)
    } else if (shrink == "ashr") {
      s <- lfcShrink(ddsShrink, coef = coefName, type = "ashr", returnList = TRUE, quiet = TRUE)
      out[[i]] <- ref_ashr(d, tag, res, s)
    } else {
      out[[i]] <- NULL
    }
  }
  invisible(out)
}

# ---- apeglm (method = "nbinomCR", apeAdapt = TRUE, interval laplace) -------------
ref_apeglm <- function(d, tag, dds, X, res, coef, s) {
  ids <- rownames(dds)
  Y <- counts(dds); G <- nrow(Y); n <- ncol(Y); p <- ncol(X)
  disps <- dispersions(dds)
  offset <- matrix(log(sizeFactors(dds)), nrow = G, ncol = n, byrow = TRUE)
  weights <- matrix(1, nrow = G, ncol = n)
  mle <- log(2) * cbind(res$log2FoldChange, res$lfcSE)

  # priorVar(mle), written out.
  keep <- !is.na(mle[, 1]); Xm <- mle[keep, 1]; D <- mle[keep, 2]^2; S <- Xm^2
  I <- function(A) 1 / (2 * (A + D)^2)
  Ahat <- function(A) sum((S - D) * I(A)) / sum(I(A))
  objective <- function(A) Ahat(A) - A
  obj_min <- objective(.001^2)
  if (obj_min < 0) {
    pv <- .001^2; ur <- list(iter = 0L, estim.prec = NA_real_, f.root = NA_real_)
  } else {
    ur <- uniroot(objective, interval = c(.001^2, 20^2)); pv <- ur$root
  }
  d$check(paste0(tag, "_apeglm_prior_var"), pv, apeglm:::priorVar(mle))
  d$check(paste0(tag, "_apeglm_prior_var_pkg"), pv, s$fit$prior.control$prior.var)
  prior.scale <- min(sqrt(pv), 1)
  d$check(paste0(tag, "_apeglm_prior_scale"), prior.scale, s$fit$prior.control$prior.scale)
  d$scalar(paste0(tag, "_apeglm_prior_var"), pv)
  d$scalar(paste0(tag, "_apeglm_prior_scale"), prior.scale)
  d$scalar(paste0(tag, "_apeglm_prior_objective_at_min"), obj_min)
  d$scalar(paste0(tag, "_apeglm_uniroot"), list(iter = ur$iter, estim_prec = ur$estim.prec, f_root = ur$f.root))
  d$scalar(paste0(tag, "_apeglm_mle_kept"), sum(keep))
  no.shrink <- setdiff(seq_len(p), coef); shrink <- setdiff(seq_len(p), no.shrink)
  pc <- list(no.shrink = no.shrink, prior.mean = 0, prior.scale = prior.scale, prior.df = 1,
             prior.no.shrink.mean = 0, prior.no.shrink.scale = 15)
  sigma <- 15; Sc <- prior.scale
  d$scalar(paste0(tag, "_apeglm_no_shrink"), no.shrink)

  # nbinomCppRoutine, written out.
  nonzero <- rowSums(Y) > 0
  YNZ <- t(Y[nonzero, , drop = FALSE]); wNZ <- t(weights[nonzero, , drop = FALSE])
  oNZ <- t(offset[nonzero, , drop = FALSE]); size <- 1 / disps[nonzero]
  nnz <- sum(nonzero)
  cnst_raw <- vapply(seq_len(nnz), function(i)
    apeglm:::nbinomFn(rep(0, p), x = X, y = YNZ[, i], size = size[i], weights = wNZ[, i],
                      offset = oNZ[, i], sigma = sigma, S = Sc, no.shrink = no.shrink,
                      shrink = shrink, cnst = 0), 0)
  cnst <- ifelse(cnst_raw > 1, cnst_raw, 1)
  init1 <- rep(c(.1, -.1), length.out = p); init2 <- rep(c(-.1, .1), length.out = p)
  glm <- function(init) apeglm:::nbinomGLM(x = X, Y = YNZ, size = size, weights = wNZ, offset = oNZ,
                                           sigma2 = sigma^2, S2 = Sc^2, no_shrink = no.shrink,
                                           shrink = shrink, init = init, cnst = cnst)
  out1 <- glm(init1); out2 <- glm(init2)
  delta <- apply(abs(out1$betas - out2$betas), 2, max)
  conv <- out1$convergence; conv[delta > .01] <- -1L

  # The optimizer path from the instrumented copy, checked bit-identical first.
  traced <- seq_len(nnz) <= SHRINK_TRACE_N | conv != 0
  tglm <- function(init) nbinomGLMTrace(x = X, Y = YNZ, size = size, weights = wNZ, offset = oNZ,
                                        sigma2 = sigma^2, S2 = Sc^2, no_shrink = no.shrink,
                                        shrink = shrink, init = init, cnst = cnst, trace_cols = traced)
  t1 <- tglm(init1); t2 <- tglm(init2)
  d$check(paste0(tag, "_apeglm_trace_fit1_betas"), t1$betas, out1$betas)
  d$check(paste0(tag, "_apeglm_trace_fit1_value"), t1$value, out1$value)
  d$check(paste0(tag, "_apeglm_trace_fit2_betas"), t2$betas, out2$betas)
  d$check(paste0(tag, "_apeglm_trace_fit2_value"), t2$value, out2$value)
  w <- 2 * p + 2
  nzid <- ids[nonzero]
  for (k in 1:2) {
    tr <- matrix(if (k == 1) t1$trace else t2$trace, ncol = w, byrow = TRUE)
    tdt <- data.table(id = nzid[tr[, 1] + 1], fit = k)
    for (j in seq_len(p)) tdt[[paste0("beta", j)]] <- tr[, 1 + j]
    tdt$f <- tr[, p + 2]
    for (j in seq_len(p)) tdt[[paste0("grad", j)]] <- tr[, p + 2 + j]
    tdt[["eval"]] <- ave(seq_along(tdt$id), tdt$id, FUN = seq_along)
    fw17(tdt, file.path(d$dir, paste0(tag, "_apeglm_path_fit", k, ".csv")))
  }

  # apeglm() row loop with nbinomCR, written out.
  intercept.idx <- rowSums(X == 0) == p - 1
  basemean <- if (sum(intercept.idx) > 0) rowMeans(Y[, intercept.idx, drop = FALSE]) else rowMeans(Y)
  map0 <- matrix(NA_real_, G, p); map0[nonzero, ] <- t(out1$betas)
  conv0 <- rep(NA_integer_, G); conv0[nonzero] <- conv
  NAp <- rep(NA_real_, p)
  rows <- vector("list", G)
  for (i in seq_len(G)) {
    y <- Y[i, ]; size_i <- 1 / disps[i]; off <- offset[i, ]; wt <- weights[i, ]
    r <- list(cnst2 = NA_real_, hess = rep(NA_real_, p * p), var_est = NAp, fallback = NA,
              nan_prefit = NA, fb_fn = NA_integer_, fb_gr = NA_integer_, fb_conv = NA_integer_,
              fb_value = NA_real_, map = NAp, sd = NAp, lo = NA_real_, hi = NA_real_, fsr = NA_real_,
              dconv = NA_real_, dcount = NA_real_, dvalue = NA_real_, final_hess = rep(NA_real_, p * p))
    if (all(y == 0)) { rows[[i]] <- r; next }
    prefit <- map0[i, ]
    r$nan_prefit <- any(is.nan(prefit))
    if (r$nan_prefit) {
      init <- rep(c(1, -1), length.out = p)
      init[1] <- if (basemean[i] == 0) 0 else log(basemean[i])
    } else init <- prefit
    prefit.conv <- conv0[i]
    r$cnst2 <- -apeglm:::nbinomFn(init, X, y, size_i, wt, off, sigma, Sc, no.shrink, shrink, 0) - 1
    var.est <- NULL
    if (prefit.conv == 0) {
      H <- -1 * optimHess(par = init, fn = apeglm:::nbinomFn, gr = apeglm:::nbinomGr, x = X, y = y,
                          size = size_i, weights = wt, offset = off, sigma = sigma, S = Sc,
                          no.shrink = no.shrink, shrink = shrink, cnst = r$cnst2)
      var.est <- diag(-solve(H))
      r$hess <- as.vector(H); r$var_est <- var.est
    }
    r$fallback <- prefit.conv != 0 || any(var.est <= 0)
    o <- apeglm:::optimNbinomHess(init = init, y = y, x = X, param = disps[i], weights = wt, offset = off,
                                  prior.control = pc, bounds = c(-Inf, Inf), optim.method = "BFGS",
                                  prefit.conv = prefit.conv)
    if (r$fallback) {
      r$fb_fn <- o$counts[[1]]; r$fb_gr <- o$counts[[2]]; r$fb_conv <- o$convergence
      fo <- optim(par = init, fn = apeglm:::nbinomFn, gr = apeglm:::nbinomGr, x = X, y = y, size = size_i,
                  weights = wt, offset = off, sigma = sigma, S = Sc, no.shrink = no.shrink, shrink = shrink,
                  cnst = r$cnst2, method = "BFGS")
      r$fb_value <- fo$value
    }
    r$final_hess <- as.vector(o$hessian)
    map <- o$par
    cov.mat <- -solve(o$hessian)
    r$map <- map
    if (any(diag(cov.mat) <= 0)) { rows[[i]] <- r; next }
    sd <- sqrt(diag(cov.mat))
    qn <- qnorm((1 - 0.95) / 2, lower.tail = FALSE)
    r$sd <- sd; r$lo <- map[coef] - qn * sd[coef]; r$hi <- map[coef] + qn * sd[coef]
    r$fsr <- pnorm(-abs(map[coef]), 0, sd[coef])
    r$dconv <- o$convergence; r$dcount <- o$counts[1]; r$dvalue <- o$value
    rows[[i]] <- r
  }
  gm <- function(nm, len) matrix(unlist(lapply(rows, `[[`, nm)), ncol = len, byrow = TRUE)
  map <- gm("map", p); sdm <- gm("sd", p); fsr <- vapply(rows, `[[`, 0, "fsr")
  svalue <- apeglm::svalue(fsr)
  fit <- s$fit
  d$check(paste0(tag, "_apeglm_map"), map, fit$map)
  d$check(paste0(tag, "_apeglm_sd"), sdm, fit$sd)
  d$check(paste0(tag, "_apeglm_interval_lo"), vapply(rows, `[[`, 0, "lo"), fit$interval[, 1])
  d$check(paste0(tag, "_apeglm_interval_hi"), vapply(rows, `[[`, 0, "hi"), fit$interval[, 2])
  d$check(paste0(tag, "_apeglm_fsr"), fsr, fit$fsr[, 1])
  d$check(paste0(tag, "_apeglm_svalue"), svalue, fit$svalue[, 1])
  d$check(paste0(tag, "_apeglm_diag_conv"), vapply(rows, `[[`, 0, "dconv"), fit$diag[, "conv"])
  d$check(paste0(tag, "_apeglm_diag_count"), vapply(rows, `[[`, 0, "dcount"), fit$diag[, "count"])
  d$check(paste0(tag, "_apeglm_lfcShrink_lfc"), log2(exp(1)) * map[, coef], s$res$log2FoldChange)
  d$check(paste0(tag, "_apeglm_lfcShrink_se"), log2(exp(1)) * sdm[, coef], s$res$lfcSE)

  # Per-gene dumps.
  pre <- data.table(id = ids, nonzero = nonzero, basemean = basemean)
  pre$cnst_raw <- NA_real_; pre$cnst_raw[nonzero] <- cnst_raw
  pre$cnst <- NA_real_; pre$cnst[nonzero] <- cnst
  for (k in 1:2) {
    o <- if (k == 1) out1 else out2; tk <- if (k == 1) t1 else t2
    for (j in seq_len(p)) { v <- rep(NA_real_, G); v[nonzero] <- o$betas[j, ]; pre[[sprintf("fit%d_beta%d", k, j)]] <- v }
    v <- rep(NA_real_, G); v[nonzero] <- o$value; pre[[sprintf("fit%d_value", k)]] <- v
    v <- rep(NA_integer_, G); v[nonzero] <- o$convergence; pre[[sprintf("fit%d_status", k)]] <- v
    v <- rep(NA_integer_, G); v[nonzero] <- tk$nevals; pre[[sprintf("fit%d_nevals", k)]] <- v
  }
  pre$delta <- NA_real_; pre$delta[nonzero] <- delta
  pre$prefit_conv <- conv0
  pre$traced <- FALSE; pre$traced[nonzero] <- traced
  fw17(pre, file.path(d$dir, paste0(tag, "_apeglm_prefit.csv")), na = "NA")

  rp <- data.table(id = ids, nan_prefit = vapply(rows, function(r) r$nan_prefit, NA),
                   cnst2 = vapply(rows, `[[`, 0, "cnst2"), fallback = vapply(rows, function(r) r$fallback, NA),
                   fb_fn = vapply(rows, function(r) as.integer(r$fb_fn), 0L),
                   fb_gr = vapply(rows, function(r) as.integer(r$fb_gr), 0L),
                   fb_conv = vapply(rows, function(r) as.integer(r$fb_conv), 0L),
                   fb_value = vapply(rows, `[[`, 0, "fb_value"))
  hm <- gm("hess", p * p); fh <- gm("final_hess", p * p); ve <- gm("var_est", p)
  for (a in seq_len(p)) for (b in seq_len(p)) rp[[sprintf("hess_%d_%d", a, b)]] <- hm[, (b - 1) * p + a]
  for (a in seq_len(p)) rp[[sprintf("var_est_%d", a)]] <- ve[, a]
  for (a in seq_len(p)) for (b in seq_len(p)) rp[[sprintf("final_hess_%d_%d", a, b)]] <- fh[, (b - 1) * p + a]
  fw17(rp, file.path(d$dir, paste0(tag, "_apeglm_rowpass.csv")), na = "NA")

  fo <- data.table(id = ids)
  for (j in seq_len(p)) fo[[paste0("map", j)]] <- map[, j]
  for (j in seq_len(p)) fo[[paste0("sd", j)]] <- sdm[, j]
  fo$interval_lo <- fit$interval[, 1]; fo$interval_hi <- fit$interval[, 2]
  fo$fsr <- fit$fsr[, 1]; fo$svalue <- fit$svalue[, 1]
  fo$diag_conv <- fit$diag[, "conv"]; fo$diag_count <- fit$diag[, "count"]; fo$diag_value <- fit$diag[, "value"]
  fo$log2FoldChange <- s$res$log2FoldChange; fo$lfcSE <- s$res$lfcSE
  fo$CrILeft <- log2(exp(1)) * fit$interval[, 1]; fo$CrIRight <- log2(exp(1)) * fit$interval[, 2]
  # Gradient certificate: R's nbinomGr (unscaled, cnst-free) at the MAP.
  gr <- matrix(NA_real_, G, p)
  for (i in which(nonzero)) if (!any(is.na(map[i, ])))
    gr[i, ] <- apeglm:::nbinomGr(map[i, ], X, Y[i, ], 1 / disps[i], weights[i, ], offset[i, ], sigma, Sc, no.shrink, shrink, 0)
  for (j in seq_len(p)) fo[[paste0("grad_at_map", j)]] <- gr[, j]
  fw17(fo, file.path(d$dir, paste0(tag, "_apeglm_final.csv")), na = "NA")
  d$scalar(paste0(tag, "_apeglm_n_fallback"), sum(rp$fallback, na.rm = TRUE))
  d$scalar(paste0(tag, "_apeglm_n_conv_flagged"), sum(conv != 0))
  d$scalar(paste0(tag, "_apeglm_n_nan_prefit"), sum(rp$nan_prefit, na.rm = TRUE))
  invisible(fo)
}

# ---- ashr (mixcompdist = "normal", method = "shrink", optmethod mixSQP) -----------
ref_ashr <- function(d, tag, res, s) {
  ids <- rownames(res)
  betahat <- res$log2FoldChange; sebetahat <- res$lfcSE
  data <- ashr::set_data(betahat, sebetahat)
  excl <- ashr:::get_exclusions(data)
  mixsd <- ashr:::autoselect.mixsd(data, sqrt(2), 0, c(-Inf, Inf), "normal")
  k <- length(mixsd)
  g0 <- ashr::normalmix(rep(1, k), rep(0, k), mixsd)
  llik <- t(ashr:::log_comp_dens_conv.normalmix(g0, data))[!excl, , drop = FALSE]
  lnorm <- apply(llik, 1, max)
  Lmat <- exp(llik - lnorm)
  nzc <- apply(Lmat, 2, max) > 0
  Lm <- Lmat[, nzc, drop = FALSE]
  nn <- nrow(Lm); m <- ncol(Lm)
  vec17(d, paste0(tag, "_ashr_data"), ids, x = data$x, s = data$s, excluded = excl)
  fw17(data.table(mixsd = mixsd, nonzero_col = nzc), file.path(d$dir, paste0(tag, "_ashr_grid.csv")))
  mat17(d, paste0(tag, "_ashr_L"), Lm, ids = ids[!excl])
  vec17(d, paste0(tag, "_ashr_lnorm"), ids[!excl], lnorm = lnorm)

  # mixsqp() preprocessing and its EM phase, written out.
  w <- rep(1, nn) / nn; x0 <- rep(1, m) / m
  nr <- mixsqp:::normalize.rows(Lm); Ln <- nr$A; z <- log(nr$z)
  d$check(paste0(tag, "_ashr_rowmax_is_one"), nr$z, rep(1, nn))
  sv <- svd(Ln, 0, 0)$d
  set.seed(1); ts <- if (m > 4) mixsqp:::tsvd(Ln, 1e-6) else NULL
  use_svd <- !is.null(ts) && ncol(ts$U) < m
  eps <- rep(1e-6 - min(0, min(Ln)), nn)
  em <- mixsqp:::mixem_rcpp(Ln, w, z, x0, eps, 20L, 1e-8, FALSE)
  set.seed(1)
  sq <- mixsqp::mixsqp(Lm, rep(1, nn), rep(1, m), control = list(verbose = FALSE, eps = 1e-6, numiter.em = 20))
  d$scalar(paste0(tag, "_ashr_mixsqp"), list(n = nn, m = m, use_svd = use_svd,
           tsvd_null = is.null(ts), status = sq$status, value = sq$value,
           niter = nrow(sq$progress), eps = eps[1]))
  fw17(data.table(sv = sv), file.path(d$dir, paste0(tag, "_ashr_singular_values.csv")))
  fw17(data.table(em_x = drop(em$x), x = drop(sq$x), grad = drop(sq$grad)),
         file.path(d$dir, paste0(tag, "_ashr_mixsqp_x.csv")))
  fw17(data.table(em_objective = drop(em$objective), em_max_diff = drop(em$max.diff)),
         file.path(d$dir, paste0(tag, "_ashr_em_progress.csv")))
  fw17(as.data.table(sq$progress), file.path(d$dir, paste0(tag, "_ashr_mixsqp_progress.csv")), na = "NA")
  mat17(d, paste0(tag, "_ashr_mixsqp_hessian"), sq$hessian, ids = NULL)

  pihat <- pmax(sq$x / sum(sq$x), 0)
  pi_full <- rep(0, k); pi_full[nzc] <- pihat
  ghat <- ashr::normalmix(pi_full, rep(0, k), mixsd)
  fit <- s$fit
  d$check(paste0(tag, "_ashr_mixsd"), mixsd, fit$fitted_g$sd)
  d$check(paste0(tag, "_ashr_pi"), pi_full, fit$fitted_g$pi)
  ll <- ashr::calc_loglik(ghat, data)
  d$check(paste0(tag, "_ashr_loglik"), ll, fit$loglik)
  gp <- ashr:::prune.default(ghat, 1e-10)
  fw17(data.table(pi = pi_full, kept = pi_full > 1e-10), file.path(d$dir, paste0(tag, "_ashr_pi.csv")))
  pm <- ashr:::calc_pm(gp, data); psd <- ashr:::calc_psd(gp, data)
  np <- ashr:::calc_np(gp, data); lfdr <- ashr:::calc_lfdr(gp, data); lfsr <- ashr:::calc_lfsr(gp, data)
  d$check(paste0(tag, "_ashr_PosteriorMean"), pm, fit$result$PosteriorMean)
  d$check(paste0(tag, "_ashr_PosteriorSD"), psd, fit$result$PosteriorSD)
  d$check(paste0(tag, "_ashr_lfsr"), lfsr, fit$result$lfsr)
  d$check(paste0(tag, "_ashr_NegativeProb"), np, fit$result$NegativeProb)
  d$check(paste0(tag, "_ashr_lfcShrink_lfc"), pm, s$res$log2FoldChange)
  d$check(paste0(tag, "_ashr_lfcShrink_se"), psd, s$res$lfcSE)
  d$scalar(paste0(tag, "_ashr_loglik"), ll)
  qn <- qnorm(0.975)
  fo <- data.table(id = ids, PosteriorMean = pm, PosteriorSD = psd, NegativeProb = np, ZeroProb = lfdr,
                   lfsr = lfsr, svalue = fit$result$svalue, log2FoldChange = s$res$log2FoldChange,
                   lfcSE = s$res$lfcSE, CrILeft = pm - qn * psd, CrIRight = pm + qn * psd)
  fw17(fo, file.path(d$dir, paste0(tag, "_ashr_final.csv")), na = "NA")
  invisible(fo)
}
