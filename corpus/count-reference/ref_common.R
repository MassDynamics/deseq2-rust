# Shared plumbing for the standalone edgeR / DESeq2 reference (PR-E2).
#
# The reference rebuilds each count engine step by step from its internals,
# dumps every intermediate as its own golden, and then reassembles the
# production output table from those intermediates. Its self-check is that
# reassembled table against the end-to-end corpus run (results.rds).
#
# Input preparation (safe column names, long table, count matrix, sample info,
# comparison encoding) reuses the MDFlexiComparisons helpers, because that part
# is the Python router's job in the port, not the engine's.

suppressPackageStartupMessages({
  library(data.table); library(Biobase); library(SummarizedExperiment); library(limma)
  library(stringr); library(glue); library(log4r); library(dplyr); library(jsonlite)
  library(foreach); library(S4Vectors); library(edgeR); library(DESeq2)
})

FLEXI_DIR  <- Sys.getenv("MD_FLEXI_DIR")
CORPUS_DIR <- Sys.getenv("MD_COUNT_CORPUS_DIR", "/corpus")
OUT_ROOT   <- file.path(CORPUS_DIR, "reference")

# Attach and source the production package the same way the corpus harness does.
for (f in list.files(file.path(FLEXI_DIR, "R"), pattern = "\\.R$", full.names = TRUE)) source(f)

# ---- dumping -----------------------------------------------------------------
# Every intermediate is written as CSV with shortest round-trip doubles, keyed
# by gene id where it is per-gene. Scalars go to reference.json.
new_dumper <- function(dir) {
  dir.create(dir, recursive = TRUE, showWarnings = FALSE)
  env <- new.env()
  env$dir <- dir
  env$scalars <- list()
  env$checks <- list()
  env$vec <- function(name, ids, ...) {
    dt <- data.table(id = ids, ...)
    fwrite(dt, file.path(dir, paste0(name, ".csv")), na = "NA")
  }
  env$matrix <- function(name, m, ids = rownames(m)) {
    dt <- as.data.table(m)
    if (!is.null(ids)) dt <- cbind(data.table(id = ids), dt)
    fwrite(dt, file.path(dir, paste0(name, ".csv")), na = "NA")
  }
  env$scalar <- function(name, value) env$scalars[[name]] <- value
  # Internal consistency: the plain-R recompute against the package function.
  env$check <- function(name, ours, pkg) {
    r <- compare_vec(ours, pkg)
    env$checks[[name]] <- r
    invisible(r)
  }
  env
}

compare_vec <- function(a, b) {
  a <- as.numeric(unlist(a)); b <- as.numeric(unlist(b))
  if (length(a) != length(b))
    return(list(pass = FALSE, detail = sprintf("length %d vs %d", length(a), length(b))))
  na_mismatch <- sum(xor(is.na(a), is.na(b)))
  both <- !is.na(a) & !is.na(b)
  inf_mismatch <- sum(both & (is.infinite(a) | is.infinite(b)) & a != b)
  fin <- both & is.finite(a) & is.finite(b)
  rel <- abs(a[fin] - b[fin]) / pmax(abs(b[fin]), 1e-300)
  rel[a[fin] == b[fin]] <- 0
  max_rel <- if (any(fin)) max(rel) else 0
  list(pass = na_mismatch == 0 && inf_mismatch == 0 && max_rel <= 1e-8,
       identical = na_mismatch == 0 && inf_mismatch == 0 && max_rel == 0,
       max_rel = max_rel, na_mismatch = na_mismatch, n = length(a))
}

# ---- inputs ------------------------------------------------------------------
# Mirrors runDiscovery()/runANOVA() up to the engine call, for a corpus run.
prepare_inputs <- function(manifest) {
  p <- manifest$params
  shared <- file.path(CORPUS_DIR, manifest$shared_inputs_dir)
  ed  <- readRDS(file.path(shared, "experiment_design.rds"))
  int <- readRDS(file.path(shared, "intensity.rds"))
  run_dir <- file.path(CORPUS_DIR, "runs", manifest$id)
  ctl_df  <- if (file.exists(f <- file.path(run_dir, "control_cols_df.rds"))) readRDS(f) else NULL
  cust_df <- if (file.exists(f <- file.path(run_dir, "custom_comparisons_df.rds"))) readRDS(f) else NULL

  if (p$de_method %in% c("edgeR", "DESeq2") && manifest$entity_type != "gene")
    stop(md_error(glue::glue("de_method '{p$de_method}' is only supported for gene entity type (count data). Use de_method = 'limma' for protein, peptide, metabolite, or PTM data.")))
  intensityColumn <- if ("NormalisedIntensity" %in% colnames(int)) "NormalisedIntensity" else "Intensity"
  if (sum(int[[intensityColumn]] < 0, na.rm = TRUE) > 0)
    stop(md_error("The data contains negative intensities. Please check if your data was log-transformed before starting the analysis."))

  safe <- makeSafeColumns(experimentDesign = ed, conditionCol = p$condition_col, controlColsDF = ctl_df)
  int[["condition"]] <- NULL
  imported <- importFlexiData(experimentDesign = safe$experimentDesign, intensitiesTable = int,
                              conditionCol = safe$conditionCol, controlColsDF = safe$controlColsDF,
                              groupIdCol = "GroupId", intensityColumn = intensityColumn, skipLog2 = TRUE)
  conditionCol <- safe$conditionCol
  comparisonDF <- getComparisonsDF(experimentDesign = safe$experimentDesign, conditionCol = conditionCol,
                                   customComparisonsDF = cust_df, comparisonType = p$comparison_type)
  conditionsDict <- imported$conditionsDict[[conditionCol]]
  encoded <- encodeCustomComparisonsDF(allCondLevels = conditionsDict$original,
                                       DFToEncode = comparisonDF, conditionsDict = conditionsDict)

  countMatrix <- .buildCountMatrixFromLongDT(imported$longIntensityDT, "GroupId")
  sampleInfo  <- .buildSampleInfoDF(imported$longIntensityDT, conditionCol, imported$controlCols)
  sampleInfo  <- sampleInfo[colnames(countMatrix), , drop = FALSE]
  if (anyNA(sampleInfo[[conditionCol]]))
    stop(md_error(sprintf("Condition column '%s' contains missing values. Fix the sample metadata before running DE.", conditionCol)))
  sampleInfo[[conditionCol]] <- factor(sampleInfo[[conditionCol]])

  list(countMatrix = countMatrix, sampleInfo = sampleInfo, conditionCol = conditionCol,
       controlCols = imported$controlCols, comparisonDF = comparisonDF, encoded = encoded,
       mode = manifest$mode, params = p)
}

dump_inputs <- function(inp, d) {
  d$matrix("input_counts", inp$countMatrix)
  si <- as.data.frame(inp$sampleInfo)
  for (cc in names(si)) if (is.factor(si[[cc]])) si[[cc]] <- as.character(si[[cc]])
  fwrite(si, file.path(d$dir, "input_sample_info.csv"))
  fwrite(cbind(as.data.table(inp$comparisonDF), encoded_left = inp$encoded$left,
               encoded_right = inp$encoded$right), file.path(d$dir, "input_comparisons.csv"))
  d$scalar("condition_levels", levels(inp$sampleInfo[[inp$conditionCol]]))
}

# ---- plain-R building blocks shared by both engines ----------------------------
# edgeR::filterByExpr.default with a design, min.count 10, min.total.count 15,
# large.n 10, min.prop 0.7. Returns the keep vector and its ingredients.
plain_filter_by_expr <- function(y, design) {
  lib <- colSums(y)
  mss <- 1 / max(hat(design))
  if (mss > 10) mss <- 10 + (mss - 10) * 0.7
  cutoff <- 10 / median(lib) * 1e6
  cpm <- y / rep(lib, each = nrow(y)) * 1e6
  n_above <- rowSums(cpm >= cutoff)
  total <- rowSums(y)
  keep <- n_above >= (mss - 1e-14) & total >= (15 - 1e-14)
  list(keep = keep, min_sample_size = mss, cpm_cutoff = cutoff, lib_size = lib,
       n_above = n_above, total = total)
}

# p.adjust(method = "BH") written out, for the Rust port to mirror.
plain_bh <- function(p) {
  out <- rep(NA_real_, length(p))
  ok <- !is.na(p)
  pp <- p[ok]; n <- length(pp)
  if (n == 0) return(out)
  o <- order(pp, decreasing = TRUE)
  ro <- order(o)
  out[ok] <- pmin(1, cummin(n / (n:1) * pp[o]))[ro]
  out
}

# Output columns production adds outside the engine. The router owns them in the
# port (PR-E7), so the self-check skips them; any other column one side lacks fails.
NON_ENGINE_COLUMNS <- "^(GeneNames|Description|GroupLabel|GroupLabelType|NImputed: .+|NReplicates: .+)$"

# Compare a reassembled production table against the end-to-end corpus output.
self_check <- function(ours, e2e, key = "GroupId") {
  e2e <- as.data.frame(e2e); ours <- as.data.frame(ours)
  e2e[[key]] <- as.character(e2e[[key]]); ours[[key]] <- as.character(ours[[key]])
  cols <- setdiff(intersect(names(ours), names(e2e)), key)
  report <- list(); all_pass <- TRUE
  if (!setequal(ours[[key]], e2e[[key]])) {
    return(list(passed = FALSE, detail = "row keys differ", columns = list()))
  }
  missing <- grep(NON_ENGINE_COLUMNS, setdiff(names(e2e), c(key, cols)), invert = TRUE, value = TRUE)
  extra <- setdiff(names(ours), c(key, cols))
  if (length(missing) || length(extra)) {
    all_pass <- FALSE
    report$column_sets <- list(pass = FALSE, engine_columns_missing = missing, reference_only = extra)
  }
  m <- match(e2e[[key]], ours[[key]])
  for (cc in cols) {
    a <- ours[[cc]][m]; b <- e2e[[cc]]
    if (is.character(b) && !is.character(a)) {
      # ANOVA outputs arrive as character with NA written as "".
      b[b == ""] <- NA
      if (all(is.na(suppressWarnings(as.numeric(b[!is.na(b)])))) && any(!is.na(b))) {
        ok <- identical(as.character(a), b)
        report[[cc]] <- list(pass = ok, kind = "string")
        if (!ok) all_pass <- FALSE
        next
      }
      b <- as.numeric(b)
    }
    if (is.character(a)) {
      a[a == ""] <- NA; b[b == ""] <- NA
      ok <- identical(a, b)
      report[[cc]] <- list(pass = ok, kind = "string")
      if (!ok) all_pass <- FALSE
      next
    }
    r <- compare_vec(a, b)
    report[[cc]] <- r
    if (!isTRUE(r$pass)) all_pass <- FALSE
  }
  list(passed = all_pass, columns_compared = cols, detail = report,
       max_rel = max(c(0, unlist(lapply(report, function(r) r$max_rel)))))
}
