#!/usr/bin/env Rscript
# audit_count_corpus.R -- sanity audit of the edgeR / DESeq2 count corpus (PR-E0/E1).
# Run inside the same image as gen_count.sh, with the corpus mounted at /corpus:
#   docker run ... -v ~/wd/md-count-golden-corpus:/corpus <image> Rscript data-raw/golden-corpus/audit_count_corpus.R
# Prints one line per check and a final FAIL count; writes /corpus/audit.json.

suppressPackageStartupMessages(library(jsonlite))
`%||%` <- function(a, b) if (is.null(a)) b else a
C <- Sys.getenv("MD_GOLDEN_CORPUS_DIR", "/corpus")
ids <- sort(list.files(file.path(C, "runs")))
res <- list()
check <- function(id, what, ok, detail = "") {
  res[[length(res) + 1]] <<- list(id = id, check = what, ok = isTRUE(ok), detail = detail)
  cat(sprintf("%-4s %-48s %-34s %s\n", if (isTRUE(ok)) "ok" else "FAIL", id, what, detail))
}
load_run <- function(id) {
  m <- fromJSON(file.path(C, "runs", id, "manifest.json"), simplifyVector = FALSE)
  r <- if (identical(m$status, "ok")) readRDS(file.path(C, "runs", id, "results.rds")) else NULL
  # runANOVA returns every column as character with "" for filtered genes; coerce the numeric ones.
  if (!is.null(r)) for (k in setdiff(names(r), c("GroupId", "GroupLabel", "GroupLabelType", "GeneNames",
                                                 "Description", "MaxLog2FCPair")))
    if (is.character(r[[k]])) r[[k]] <- suppressWarnings(as.numeric(r[[k]]))
  list(m = m, r = r)
}
# Discovery columns are "<Stat> <comparison>"; ANOVA has the bare statistic name.
G0_R <- "4.5.0"
G0_VERSIONS <- list(edgeR = "4.8.2", DESeq2 = "1.50.2", apeglm = "1.32.0", ashr = "2.2.63", bioconductor = "3.22")
cols <- function(r, prefix) grep(paste0("^", prefix, "( |$)"), names(r), value = TRUE)

for (id in ids) {
  x <- load_run(id); m <- x$m; r <- x$r
  pv <- m$package_versions
  # Equality to the G0 set, not just presence: the newest BioC 3.22 / CRAN versions on
  # 2026-10-01, which production's unpinned build installs (plan § Versions).
  check(id, "versions equal G0 set", identical(unlist(pv[names(G0_VERSIONS)]), unlist(G0_VERSIONS)) &&
          startsWith(m$r_version, paste("R version", G0_R)),
        sprintf("edgeR %s DESeq2 %s apeglm %s ashr %s BioC %s R %s", pv$edgeR, pv$DESeq2, pv$apeglm, pv$ashr,
                pv$bioconductor, sub("R version ", "", m$r_version)))
  if (!is.null(m$expected_error)) {
    check(id, "expected error matched", identical(m$status, "error") && isTRUE(m$error_matches_expected),
          substr(m$actual_error %||% "(succeeded)", 1, 90))
    next
  }
  check(id, "status ok", identical(m$status, "ok"), substr(m$actual_error %||% "", 1, 120))
  if (is.null(r)) next
  if (!is.null(m$warning_text)) check(id, "warnings (info)", TRUE, substr(m$warning_text, 1, 120))
  eng <- m$params$de_method
  pcols <- cols(r, "PValue"); acols <- cols(r, "AdjPValue")
  pvals <- unlist(r[, pcols, drop = FALSE])
  check(id, "p-values in [0,1]", length(pcols) > 0 && all(pvals[!is.na(pvals)] >= 0 & pvals[!is.na(pvals)] <= 1),
        sprintf("%d p-value columns", length(pcols)))
  # filterByExpr-dropped genes are NA in every statistic column, kept genes have a p-value.
  stat_cols <- c(pcols, cols(r, "Log2FC"), cols(r, "MaxLog2FC"))
  all_na <- rowSums(!is.na(r[, stat_cols, drop = FALSE])) == 0
  # Every input gene comes back exactly once (dropped genes are left-joined back as NA).
  n_in <- c(airway = 8000L, count_synth = 2000L, count_synth_cooks = 2000L)[[m$dataset]]
  check(id, "rows: every input gene once", nrow(r) == n_in && !anyDuplicated(r$GroupId),
        sprintf("%d rows (input %d), %d all-NA (filterByExpr), %d kept", nrow(r), n_in, sum(all_na), sum(!all_na)))
  # edgeR: a kept gene has every p-value. DESeq2: a kept gene may lose its p-value to the Cook's
  # flag, which is gene-level, so it is NA in every pair or in none.
  n_na_p <- rowSums(is.na(r[, pcols, drop = FALSE]))
  partial <- !all_na & n_na_p > 0 & (eng == "edgeR" | n_na_p < length(pcols))
  check(id, "kept rows: NA p-values gene-level", sum(partial) == 0,
        sprintf("%d kept rows with a partial NA p-value, %d with none (Cook's)", sum(partial),
                sum(!all_na & n_na_p == length(pcols))))
  # BH recomputed: over the non-NA p (edgeR, LRT) or over the IF-passing set (DESeq2 Wald).
  bh <- vapply(seq_along(pcols), function(k) {
    p <- r[[pcols[k]]]; a <- r[[acols[k]]]; s <- !is.na(a)
    if (!any(s)) return(0)
    max(abs(p.adjust(p[s], "BH") - a[s]) / pmax(a[s], .Machine$double.xmin))
  }, 0)
  check(id, "AdjPValue = BH(PValue)", all(bh < 1e-12), sprintf("max rel %.2g", max(bh)))
  if (eng == "DESeq2" && m$mode == "discovery") {
    # Independent filtering: padj NA where pvalue is not NA, and those rows have the lowest baseMean.
    for (k in seq_along(pcols)) {
      p <- r[[pcols[k]]]; a <- r[[acols[k]]]
      ifl <- !is.na(p) & is.na(a)
      thr_ok <- if (any(ifl) && any(!is.na(a))) max(r$AveExpr[ifl]) <= min(r$AveExpr[!is.na(a)]) + 1e-9 else TRUE
      check(id, sprintf("IF threshold monotone [%d]", k), thr_ok,
            sprintf("%d IF-filtered, %d Cook/NA p", sum(ifl), sum(!all_na & is.na(p))))
    }
    sh <- m$params$deseq2_lfc_shrinkage
    if (sh == "none") {
      # Wald CI on the unshrunk MLE; with shrinkage Log2FC is the posterior, so the CI is off-centre.
      d <- abs(unlist(r[, cols(r, "Log2FC")]) - qnorm(0.975) * unlist(r[, cols(r, "SE")]) -
                 unlist(r[, cols(r, "CILeft")]))
      check(id, "CILeft = Log2FC - z*SE", all(d[!is.na(d)] < 1e-12), sprintf("max abs %.2g", max(c(0, d), na.rm = TRUE)))
    }
    cr <- unlist(r[, cols(r, "CrILeft"), drop = FALSE])
    check(id, "CrI NA iff shrinkage none", if (sh == "none") all(is.na(cr)) else any(!is.na(cr)), sh)
  }
  if (eng == "edgeR" && m$mode == "discovery") {
    # Per-comparison columns only; 3+ condition runs also carry an omnibus "F" / "PValue".
    lfc <- cols(r, "Log2FC"); se <- cols(r, "SE"); f <- grep("^F ", names(r), value = TRUE)
    d <- abs(abs(unlist(r[, lfc])) / sqrt(unlist(r[, f])) - unlist(r[, se]))
    check(id, "SE = |logFC|/sqrt(F)", all(d[is.finite(d)] < 1e-10))
  }
}

# Cross-run invariants.
pair <- function(a, b, what, f) {
  if (!all(file.exists(file.path(C, "runs", c(a, b), "results.rds")))) return(check(paste(a, "~", b), what, FALSE, "missing"))
  ra <- readRDS(file.path(C, "runs", a, "results.rds")); rb <- readRDS(file.path(C, "runs", b, "results.rds"))
  ra <- ra[order(ra$GroupId), ]; rb <- rb[order(rb$GroupId), ]
  f(ra, rb)
}
same <- function(x, y) isTRUE(all.equal(x, y, tolerance = 0, check.attributes = FALSE))
for (sh in c("normal", "apeglm", "ashr")) {
  pair("count_deseq2_count_synth_all_ctlnone", sprintf("count_deseq2_count_synth_shrink_%s", sh),
       sprintf("CI/PValue invariant to shrink=%s", sh), function(a, b) {
         cc <- c(cols(a, "CILeft"), cols(a, "CIRight"), cols(a, "PValue"), cols(a, "AdjPValue"))
         check("count_synth none~shrink", sprintf("CI/P invariant (%s)", sh), same(a[, cc], b[, cc]))
       })
}
# Named edge case (round-1 review M3): airway ENSG00000119698 has a round-off negative edgeR F
# under the cell covariate, so stat / SE / CI are NA while Log2FC and PValue are not.
for (sh in c("all", "custom")) {
  id <- sprintf("count_edger_airway_%s_ctlfactor", sh)
  if (!file.exists(file.path(C, "runs", id, "results.rds"))) { check(id, "negative-F gene pinned", FALSE, "missing"); next }
  r <- readRDS(file.path(C, "runs", id, "results.rds")); g <- r[r$GeneNames == "ENSG00000119698", ]
  f <- g[[grep("^F ", names(g), value = TRUE)[1]]]
  check(id, "negative-F gene pinned", nrow(g) == 1 && f < 0 && abs(f) < 1e-8 &&
          is.na(g[[cols(g, "SE")[1]]]) && is.na(g[[cols(g, "CILeft")[1]]]) && !is.na(g[[cols(g, "PValue")[1]]]),
        sprintf("F = %.3g", f))
}
pair("count_edger_count_synth_all_ctlnone", "count_deseq2_count_synth_all_ctlnone", "same filterByExpr set", function(a, b) {
  check("edgeR~DESeq2 count_synth", "same filterByExpr kept set",
        same(is.na(a[[cols(a, "Log2FC")[1]]]), is.na(b[[cols(b, "Log2FC")[1]]])))
})

nfail <- sum(!vapply(res, `[[`, NA, "ok"))
cat(sprintf("\n%d checks, %d FAIL\n", length(res), nfail))
writeLines(toJSON(list(n_checks = length(res), n_fail = nfail, checks = res), auto_unbox = TRUE, pretty = TRUE),
           file.path(C, "audit.json"))
