# R reference over the truth corpus: production's edgeR / DESeq2 / shrinkage calls, plus the
# diagnostics the truth certificates need (size factors, dispersions, coefficients, Cook's,
# apeglm MAP, ashr mixture).
#
#   sh corpus/truth/run_r.sh [scenario ...]        (runs this inside md-flexi-r45-local)
#
# The engine calls are MDFlexiComparisons' own internals (sourced from R/*.R): .fitEdgeRModel,
# .extractEdgeROmnibusStat, .extractEdgeRPairwiseStats, .applyFilterByExpr, .fitDESeq2Model,
# .extractDESeq2PairwiseResults and .extractDESeq2LogFCsOnly, so the tables are production's.
# The diagnostics are read from the same fitted objects. Shrinkage internals (apeglm MAP,
# ashr fitted_g) come from a second lfcShrink call with production's arguments; the script
# stops if its shrunk LFC differs from production's table.
#
# Writes <scenario>/r_<engine>.csv and <scenario>/r_<engine>.diag/.

suppressPackageStartupMessages({
  library(data.table); library(Biobase); library(SummarizedExperiment); library(limma)
  library(stringr); library(glue); library(log4r); library(dplyr); library(jsonlite)
  library(foreach); library(S4Vectors); library(edgeR); library(DESeq2)
})
FLEXI_DIR <- Sys.getenv("MD_FLEXI_DIR")
TRUTH_DIR <- Sys.getenv("MD_TRUTH_DIR", "/truth")
for (f in list.files(file.path(FLEXI_DIR, "R"), pattern = "\\.R$", full.names = TRUE)) source(f)
# Silence production's info logging.
MDFlexiComparisonsLogger <- function() log4r::logger(threshold = "ERROR")

wr <- function(dt, path) fwrite(as.data.table(dt), path, na = "NA")
wj <- function(x, path) write_json(x, path, auto_unbox = TRUE, digits = NA, pretty = TRUE)

read_scenario <- function(dir) {
  cnt <- fread(file.path(dir, "counts.csv"))
  ids <- as.character(cnt$id)
  m <- as.matrix(cnt[, -1]); rownames(m) <- ids
  storage.mode(m) <- "integer"
  si <- as.data.frame(fread(file.path(dir, "sample_info.csv")))
  rownames(si) <- si$replicate
  p <- fromJSON(file.path(dir, "params.json"))
  ctl <- if (length(p$control_cols)) p$control_cols else NULL
  for (cc in ctl) if (is.character(si[[cc]])) si[[cc]] <- factor(si[[cc]])
  si$condition <- factor(si$condition)
  si <- si[colnames(m), , drop = FALSE]
  cmp <- as.data.frame(fread(file.path(dir, "comparisons.csv")))
  list(counts = m, si = si, params = p, ctl = ctl,
       comparisonDF = data.frame(left = cmp$left, right = cmp$right, stringsAsFactors = FALSE),
       encoded = data.frame(left = cmp$encoded_left, right = cmp$encoded_right,
                            stringsAsFactors = FALSE))
}

left_join_all <- function(stats, ids) {
  allDT <- data.table(GroupId = ids); stats <- as.data.table(stats); stats$GroupId <- as.character(stats$GroupId)
  out <- merge(allDT, stats, by = "GroupId", all.x = TRUE)
  out$GroupId <- as.integer(out$GroupId)
  setorder(out, GroupId)
  out
}

# ---- edgeR -------------------------------------------------------------------------
run_edger <- function(s, dir) {
  fr <- .fitEdgeRModel(s$counts, s$si, "condition", s$ctl, normMethod = s$params$edger_norm_method)
  fit <- fr$fit; design <- fr$designMat
  anova <- .extractEdgeROmnibusStat(fit, design, s$si, "condition", "GroupId")
  pw <- .extractEdgeRPairwiseStats(fit, design, s$encoded, s$comparisonDF, "condition",
                                   s$params$condition_separator, "GroupId")
  stats <- merge(anova, pw, by = "GroupId", all = TRUE)
  wr(left_join_all(stats, rownames(s$counts)), file.path(dir, "r_edger.csv"))

  d <- file.path(dir, "r_edger.diag"); dir.create(d, showWarnings = FALSE)
  smp <- fit$samples
  wr(data.table(replicate = rownames(smp), lib_size = smp$lib.size, norm_factor = smp$norm.factors),
     file.path(d, "samples.csv"))
  uc <- fit$unshrunk.coefficients; colnames(uc) <- paste0("coef_", colnames(design))
  sc <- fit$coefficients; colnames(sc) <- paste0("coefshr_", colnames(design))
  # The DGEList kept by glmQLFit is not stored on the fit, so the trend is recomputed the same
  # way (estimateDisp on the filtered, normalised DGEList) for the diagnostics only.
  wr(data.table(id = rownames(fit$counts), ave_log_cpm = fit$AveLogCPM,
                s2_post = fit$s2.post, s2_prior = fit$s2.prior,
                df_residual_adj = fit$df.residual.adj, df_residual = fit$df.residual,
                df_prior = rep_len(fit$df.prior, nrow(fit$counts)),
                uc, sc), file.path(d, "genes.csv"))
  wr(as.data.table(design, keep.rownames = "replicate"), file.path(d, "design.csv"))
  wj(list(fit_dispersion = fit$dispersion, ave_ql_dispersion = fit$average.ql.dispersion,
          df_residual_total = sum(fit$df.residual), top_proportion = fit$top.proportion,
          design_columns = colnames(design)), file.path(d, "scalars.json"))
  invisible(fr)
}

# The trend and common dispersion are not on the QL fit object; recompute them exactly as
# .fitEdgeRModel does to record them.
edger_disp <- function(s) {
  fml <- if (length(s$ctl)) as.formula(paste("~ 0 + condition +", paste(s$ctl, collapse = " + "))) else ~ 0 + condition
  design <- model.matrix(fml, data = s$si)
  dge <- DGEList(counts = s$counts[, rownames(s$si), drop = FALSE])
  keep <- filterByExpr(dge, design = design)
  dge <- dge[keep, , keep.lib.sizes = FALSE]
  dge <- calcNormFactors(dge, method = s$params$edger_norm_method)
  dge <- estimateDisp(dge, design = design)
  list(ids = rownames(dge), trended = dge$trended.dispersion, tagwise = dge$tagwise.dispersion,
       common = dge$common.dispersion, prior_df = dge$prior.df)
}

# ---- DESeq2 ------------------------------------------------------------------------
deseq2_fit <- function(s, test = "Wald") {
  cm <- .applyFilterByExpr(s$counts, s$si, "condition", s$ctl, engine = "DESeq2")
  dds <- .fitDESeq2Model(cm, s$si, "condition", s$ctl, test = test)
  dds
}

dump_deseq2_base <- function(dds, d) {
  dir.create(d, showWarnings = FALSE)
  mm <- model.matrix(design(dds), data = colData(dds))
  wr(data.table(replicate = colnames(dds), size_factor = sizeFactors(dds)), file.path(d, "samples.csv"))
  mc <- mcols(dds)
  rn <- resultsNames(dds)
  coefs <- as.data.table(as.data.frame(mc[, rn, drop = FALSE])); setnames(coefs, paste0("coef_", rn))
  ses <- as.data.table(as.data.frame(mc[, paste0("SE_", rn), drop = FALSE]))
  cooks <- assays(dds)[["cooks"]]
  maxc <- DESeq2:::recordMaxCooks(design(dds), colData(dds), mm, cooks, nrow(dds))
  rep <- if (!is.null(mc$replace)) mc$replace else rep(FALSE, nrow(dds))
  wr(data.table(id = rownames(dds), baseMean = mc$baseMean, dispGeneEst = mc$dispGeneEst,
                dispFit = mc$dispFit, dispersion = mc$dispersion, dispOutlier = mc$dispOutlier,
                betaConv = if (!is.null(mc$betaConv)) mc$betaConv else mc$fullBetaConv, betaIter = if (!is.null(mc$betaIter)) mc$betaIter else NA, maxCooks = maxc, replace = rep, coefs, ses),
     file.path(d, "genes.csv"))
  wr(data.table(id = rownames(dds), as.data.table(cooks)), file.path(d, "cooks.csv"))
  if ("replaceCounts" %in% assayNames(dds) && any(rep %in% TRUE)) {
    rc <- assays(dds)[["replaceCounts"]][which(rep %in% TRUE), , drop = FALSE]
    wr(data.table(id = rownames(rc), as.data.table(rc)), file.path(d, "replaced_counts.csv"))
  }
  wr(as.data.table(mm, keep.rownames = "replicate"), file.path(d, "design.csv"))
  m <- ncol(dds); p <- ncol(mm)
  wj(list(dispPriorVar = attr(dispersionFunction(dds), "dispPriorVar"),
          varLogDispEsts = attr(dispersionFunction(dds), "varLogDispEsts"),
          trend_coefficients = as.numeric(attr(dispersionFunction(dds), "coefficients")),
          fit_type = attr(dispersionFunction(dds), "fitType"),
          cooks_cutoff = if (m > p) qf(0.99, p, m - p) else NA,
          results_names = rn, m = m, p = p), file.path(d, "scalars.json"))
}

relevel_dds <- function(dds, right) {
  lv <- levels(colData(dds)$condition)
  if (lv[1] != right) {
    colData(dds)$condition <- relevel(colData(dds)$condition, ref = right)
    dds <- nbinomWaldTest(dds, quiet = TRUE)
  }
  dds
}

run_deseq2 <- function(s, dir, shrinks) {
  dds <- deseq2_fit(s, "Wald")
  aveExpr <- results(dds, independentFiltering = FALSE)$baseMean
  names(aveExpr) <- rownames(dds)
  dump_deseq2_base(dds, file.path(dir, "r_deseq2.diag"))
  for (sh in shrinks) {
    eng <- if (sh == "none") "deseq2" else paste0("deseq2_", sh)
    stats <- withr::with_preserve_seed(
      .extractDESeq2PairwiseResults(dds, s$encoded, s$comparisonDF, "condition",
                                    s$params$condition_separator, "GroupId",
                                    lfcShrinkage = sh, alpha = s$params$deseq2_alpha,
                                    apeglm_seed = s$params$apeglm_seed))
    stats$AveExpr <- aveExpr[match(stats$GroupId, names(aveExpr))]
    wr(left_join_all(stats, rownames(s$counts)), file.path(dir, paste0("r_", eng, ".csv")))
    if (sh %in% c("apeglm", "ashr")) {
      d <- file.path(dir, paste0("r_", eng, ".diag")); dir.create(d, showWarnings = FALSE)
      for (k in seq_len(nrow(s$encoded))) {
        L <- s$encoded$left[k]; R <- s$encoded$right[k]
        lab <- paste0(s$comparisonDF$left[k], s$params$condition_separator, s$comparisonDF$right[k])
        ddsS <- relevel_dds(dds, R)
        coefName <- paste0("condition_", L, "_vs_", R)
        if (sh == "apeglm") {
          r <- withr::with_seed(s$params$apeglm_seed,
                 lfcShrink(ddsS, coef = coefName, type = "apeglm", returnList = TRUE, quiet = TRUE))
          mm <- model.matrix(design(ddsS), data = colData(ddsS))
          ci <- match(coefName, resultsNames(ddsS))
          mp <- r$fit$map; colnames(mp) <- paste0("map_", resultsNames(ddsS))
          chk <- log2(exp(1)) * r$fit$map[, ci]
          prod <- stats[[paste0("Log2FC ", lab)]][match(rownames(ddsS), stats$GroupId)]
          if (!isTRUE(all.equal(unname(chk), unname(prod)))) stop("apeglm diag rerun differs from production")
          wr(data.table(id = rownames(ddsS), as.data.table(mp), sd = r$fit$sd[, ci],
                        conv = r$fit$diag[, "conv"]), file.path(d, sprintf("cmp%d.csv", k)))
          wr(as.data.table(mm, keep.rownames = "replicate"), file.path(d, sprintf("cmp%d_design.csv", k)))
          wj(list(label = lab, coef = coefName, coef_index = ci,
                  prior_scale = r$fit$prior.control$prior.scale,
                  prior_var = r$fit$prior.control$prior.var,
                  no_shrink_scale = r$fit$prior.control$prior.no.shrink.scale),
             file.path(d, sprintf("cmp%d.json", k)))
        } else {
          res <- results(ddsS, name = coefName)
          fit <- ashr::ash(res$log2FoldChange, res$lfcSE, mixcompdist = "normal", method = "shrink")
          prod <- stats[[paste0("Log2FC ", lab)]][match(rownames(ddsS), stats$GroupId)]
          if (!isTRUE(all.equal(fit$result$PosteriorMean, unname(prod)))) stop("ashr diag rerun differs from production")
          g <- fit$fitted_g
          wr(data.table(pi = g$pi, mean = g$mean, sd = g$sd), file.path(d, sprintf("cmp%d_g.csv", k)))
          wr(data.table(id = rownames(ddsS), betahat = res$log2FoldChange, sebetahat = res$lfcSE,
                        PosteriorMean = fit$result$PosteriorMean, PosteriorSD = fit$result$PosteriorSD,
                        lfsr = fit$result$lfsr, lfdr = fit$result$lfdr), file.path(d, sprintf("cmp%d.csv", k)))
          wj(list(label = lab, loglik = fit$loglik), file.path(d, sprintf("cmp%d.json", k)))
        }
      }
    }
  }
}

run_deseq2_anova <- function(s, dir) {
  dds <- deseq2_fit(s, "LRT")
  lfc <- .extractDESeq2LogFCsOnly(dds, s$encoded, s$comparisonDF, "condition",
                                  s$params$condition_separator, "GroupId")
  lrt <- as.data.frame(results(dds, independentFiltering = TRUE, alpha = s$params$deseq2_alpha))
  an <- data.table(GroupId = rownames(lrt), AveExpr = lrt$baseMean, LRT = lrt$stat,
                   PValue = lrt$pvalue, AdjPValue = lrt$padj)
  lfc <- as.data.table(lfc); lfc$GroupId <- as.character(lfc$GroupId)
  stats <- merge(an, lfc, by = "GroupId", all = TRUE)
  wr(left_join_all(stats, rownames(s$counts)), file.path(dir, "r_deseq2_anova.csv"))
  d <- file.path(dir, "r_deseq2_anova.diag")
  dump_deseq2_base(dds, d)
}

run_scenario <- function(name) {
  dir <- file.path(TRUTH_DIR, name)
  s <- read_scenario(dir)
  t0 <- proc.time()[["elapsed"]]
  if (identical(s$params$mode, "anova")) {
    run_deseq2_anova(s, dir)
    shr <- character(0)
  } else {
    shr <- if (grepl("^null_", name)) "none" else c("none", "normal", "apeglm", "ashr")
    run_deseq2(s, dir, shr)
  }
  run_edger(s, dir)
  ed <- edger_disp(s)
  wr(data.table(id = ed$ids, trended_disp = ed$trended, tagwise_disp = ed$tagwise),
     file.path(dir, "r_edger.diag", "disp.csv"))
  sj <- fromJSON(file.path(dir, "r_edger.diag", "scalars.json"))
  sj$common_disp <- ed$common; sj$disp_prior_df <- ed$prior_df
  wj(sj, file.path(dir, "r_edger.diag", "scalars.json"))
  cat(sprintf("%-16s ok %.1fs\n", name, proc.time()[["elapsed"]] - t0))
}

ids <- commandArgs(trailingOnly = TRUE)
if (length(ids) == 0) {
  ids <- list.dirs(TRUTH_DIR, full.names = FALSE, recursive = FALSE)
}
ok <- vapply(ids, function(id) tryCatch(withCallingHandlers({ run_scenario(id); TRUE },
  error = function(e) {
    calls <- vapply(sys.calls(), function(cl) paste(deparse(cl, nlines = 1L), collapse = ""), "")
    cat(paste("  at", tail(calls, 12)), sep = "\n")
  }), error = function(e) {
  cat(sprintf("%-16s FAIL %s\n", id, conditionMessage(e))); FALSE }), TRUE)
cat(sprintf("r reference: %d / %d scenarios ok\n", sum(ok), length(ok)))
quit(status = if (all(ok)) 0 else 1)
