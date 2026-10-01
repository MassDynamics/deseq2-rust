#!/usr/bin/env Rscript
# generate_golden_corpus.R
#
# Standalone generator for the limma->Rust rewrite golden corpus (PR-A1).
# Run with:
#   cd /Users/giuseppeinfusini/wd/md-repos/MDFlexiComparisons
#   source .claude-r-env
#   "$RSCRIPT_EXEC" data-raw/golden-corpus/generate_golden_corpus.R
#
# See data-raw/golden-corpus/README.md for prerequisites and layout. Paths
# below are hardcoded per instruction -- edit them if your checkout differs.
# This script makes NO changes to the MDFlexiComparisons package tree: it
# only reads from it and writes under MD_GOLDEN_CORPUS_DIR.

REPO_ROOT <- "/Users/giuseppeinfusini/wd/md-repos/MDFlexiComparisons"
# MD_CORPUS_KIND=count selects run_matrix.R's `count_runs` (edgeR / DESeq2, phase 2);
# anything else is the limma corpus. Each kind has its own default corpus dir.
CORPUS_KIND <- Sys.getenv("MD_CORPUS_KIND", "limma")
CORPUS_DIR <- Sys.getenv("MD_GOLDEN_CORPUS_DIR",
                         if (CORPUS_KIND == "count") "/Users/giuseppeinfusini/wd/md-count-golden-corpus"
                         else "/Users/giuseppeinfusini/wd/md-limma-golden-corpus")

# ---- Locale / determinism ------------------------------------------------
# The plan requires LC_COLLATE=C so string sorts (condition levels, GroupId
# ordering) are byte-order deterministic across machines.
Sys.setlocale("LC_COLLATE", "C")
if (Sys.getenv("LC_COLLATE") != "C") Sys.setenv(LC_COLLATE = "C")

suppressPackageStartupMessages({
  library(data.table); library(Biobase); library(SummarizedExperiment); library(limma)
  library(stringr); library(glue); library(log4r); library(dplyr); library(jsonlite)
  library(foreach); library(S4Vectors)
})

# The package has no compiled code (its only src/ content is the Python
# sibling package, not R C/C++), so it can be exercised by sourcing R/*.R
# directly rather than installing/devtools::load_all() (which needs Xcode
# build tools that are not present on this machine). This also means every
# function -- including dot-prefixed internal ones like .runLimmaDiscovery --
# is directly callable, which the self-check below relies on.
rfiles <- list.files(file.path(REPO_ROOT, "R"), pattern = "[.]R$", full.names = TRUE)
for (f in rfiles) source(f)
stopifnot(exists("runDiscovery"), exists("runANOVA"), exists("md_error"),
          exists("fitOneConditionWithContrasts"), exists("importFlexiData"),
          exists("getComparisonsDF"))

# The R image installs limma unpinned through BiocManager (dependencies.R), so MD's eBayes
# calls (R/limmaStatsFun.R:305 and :437) run with limma's default legacy = NULL.

# createRunMetadata() (R/runDiscovery.R:550) calls packageVersion("MDFlexiComparisons"),
# which errors because the package is sourced directly rather than installed
# (see README.md). Shadow base::packageVersion in globalenv -- where every
# sourced function's closure lives, since source() with the default envir
# puts them there -- so that one call resolves to a fixed sentinel version
# instead of throwing, without editing any file under R/.
packageVersion <- function(pkg, ...) {
  if (identical(pkg, "MDFlexiComparisons")) {
    return(package_version("0.0.0.9000")) # sentinel: package sourced directly, not installed -- see README.md
  }
  utils::packageVersion(pkg, ...)
}

HAS_ARROW <- requireNamespace("arrow", quietly = TRUE)
save_table <- function(obj, path_noext) {
  # Deliverable spec: parquet via {arrow} where available, else .rds with a
  # note. arrow is NOT installed in this renv library (confirmed via
  # requireNamespace) and renv.lock must not be touched to add it, so every
  # run in this corpus generation is written as .rds. The hook is kept so a
  # future run with arrow installed produces parquet without code changes.
  if (HAS_ARROW) {
    arrow::write_parquet(obj, paste0(path_noext, ".parquet"))
    paste0(basename(path_noext), ".parquet")
  } else {
    saveRDS(obj, paste0(path_noext, ".rds"))
    paste0(basename(path_noext), ".rds")
  }
}

dir.create(CORPUS_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(CORPUS_DIR, "shared"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(CORPUS_DIR, "runs"), recursive = TRUE, showWarnings = FALSE)

# ---- Provenance (fix plan Step 6) ------------------------------------------
# The p-values of the unequal-df prior fit depend on optimize() and libm, so an exact replay
# needs the platform too. The container has no git and cannot see its own digest: the caller
# passes them in (MD_FLEXI_SHA, MD_IMAGE_DIGEST, MD_HARNESS_SHA256); "unknown" says the caller did not.
PROVENANCE <- list(
  md_flexi_comparisons_sha = Sys.getenv("MD_FLEXI_SHA", "unknown"),
  image_digest = Sys.getenv("MD_IMAGE_DIGEST", "unknown"),
  harness_sha256 = Sys.getenv("MD_HARNESS_SHA256", "unknown"),
  limma_legacy = FALSE,
  platform = R.version$platform,
  lc_ctype = Sys.getlocale("LC_CTYPE"),
  libc = tryCatch(system2("ldd", "--version", stdout = TRUE, stderr = TRUE)[1],
                  error = function(e) "unknown"),
  lapack = La_version()
)
if (CORPUS_KIND == "count") {
  # limma's eBayes is not on the count path. G0: production pins nothing (Dockerfile on
  # md_dataset_package_r_base:latest, dependencies.R installs unversioned), so the golden versions
  # are the newest of the BioC release that R 4.5.0 selects (3.22). Confirmed 2026-10-01: every
  # package in the image equals the newest available. Rebuild the image and set
  # MD_VERSIONS_STATUS=provisional if a newer patch release appears.
  PROVENANCE$limma_legacy <- NULL
  PROVENANCE$versions_status <- Sys.getenv("MD_VERSIONS_STATUS", "confirmed")
}

# ---- Tolerance policy (plan Phase 3, copied verbatim into every manifest) --
TOLERANCE_POLICY <- list(
  logFC_CI_coef   = list(kind = "abs", value = 1e-10,
                          columns = "Log2FC, CILeft, CIRight and other lmFit/contrasts.fit coefficients"),
  se_s2_t_F       = list(kind = "rel", value = 1e-8,
                          columns = "SE, s2.post, stat (moderated t), F"),
  pvalues         = list(kind = "rel", value = 1e-8, abs_floor = 1e-12,
                          columns = "PValue, AdjPValue"),
  df_s2_prior     = list(kind = "rel", value = 1e-6,
                          columns = "df.prior, s2.prior (eBayes empirical priors)"),
  decide_columns  = list(kind = "exact_int",
                          columns = "decide_* (limma::decideTests), except boundary_sensitive runs",
                          note = "boundary_sensitive=TRUE runs may legitimately flip near a 0.5-threshold decision under a different limma point release/BLAS; compare with a wider band and inspect by eye."),
  counts_ids_cols = list(kind = "exact", columns = "row counts, GroupId values, column names")
)

# The count corpus (edgeR / DESeq2) has its own policy: rel 1e-8 on continuous outputs, exact on
# NA pattern, the independent-filtering threshold and significance flags (plan § Decisions), with
# one floor for round-off. edgeR's QL F can come out a tiny negative number for a gene with
# logFC ~ 0 (airway ENSG00000119698, F = -4.2e-10, in the *_ctlfactor runs); production takes
# sqrt(F), so stat / SE / CI are NA there. The sign of such an F is round-off, so neither rel 1e-8
# on F nor the exact NA rule on its derived columns can hold for it.
COUNT_TOLERANCE_POLICY <- list(
  continuous      = list(kind = "rel", value = 1e-8,
                          columns = "Log2FC, SE, stat, F, CI, CrI, AveExpr, PValue, AdjPValue"),
  edger_f_floor   = list(kind = "abs", value = 1e-8,
                          columns = "F where |F_golden| < 1e-8; then stat, SE, CILeft, CIRight NA-ness is exempt from na_pattern",
                          note = "Port reproduces production: F < 0 gives NA stat/SE/CI (known_production_behaviour)."),
  na_pattern      = list(kind = "exact", columns = "NA positions in every column, except the edger_f_floor rows"),
  if_threshold    = list(kind = "exact", columns = "DESeq2 independent-filtering filtered set",
                          note = "Enforced now as the exact set of rows with PValue present and AdjPValue NA; the numeric threshold and theta get their own golden in PR-E2."),
  significance    = list(kind = "exact", columns = "AdjPValue < deseq2_alpha (DESeq2) or 0.05 (edgeR)",
                          note = "Derived, not stored: no output carries a flag column, so the port's flag computed from its AdjPValue must equal the flag computed from the golden AdjPValue."),
  edger_ci_df     = list(kind = "note", columns = "edgeR CILeft, CIRight",
                          note = "known_production_behaviour on every edgeR discovery run: the CI uses qt(0.975, df.prior + df.residual) (edgeRStatsFun.R:227-231), not glmQLFTest's df.total. The port reproduces it."),
  counts_ids_cols = list(kind = "exact", columns = "row counts, GroupId values, column names")
)

# ---- Dataset loading ------------------------------------------------------
load_rda_as_list <- function(name) {
  e <- new.env()
  load(file.path(REPO_ROOT, "data", paste0(name, ".rda")), envir = e)
  get(name, envir = e)
}

get_intensity_key <- function(entity_type) paste0(entity_type, "_intensity")
get_metadata_key  <- function(entity_type) paste0(entity_type, "_metadata")

build_gene_synth <- function() {
  # Small deterministic synthetic gene-entity dataset. Real bundled datasets
  # (bojkova2020/demichev2021/monkeyPox/panCancer) are all protein or
  # peptide entity; the corpus needs exactly one gene-entity run (the
  # process_r.py shim-override run), so a minimal synthetic fixture is
  # built here rather than fabricating a large one.
  set.seed(20260925)
  n_genes <- 40
  conditions <- c("treated", "control")
  reps <- 3
  exp_design <- data.frame(
    sample_name = paste0(rep(conditions, each = reps), "_", rep(seq_len(reps), times = length(conditions))),
    condition = rep(conditions, each = reps),
    stringsAsFactors = FALSE
  )
  gene_ids <- sprintf("GENE%04d", seq_len(n_genes))
  base_expr <- rnorm(n_genes, mean = 8, sd = 1.5)
  effect <- c(rep(0, n_genes * 0.7), rnorm(n_genes * 0.3, mean = 1.5, sd = 0.5))
  effect <- sample(effect) # shuffle so DE genes aren't just the first rows
  intensity_rows <- list()
  k <- 1
  for (i in seq_len(n_genes)) {
    for (s in seq_len(nrow(exp_design))) {
      cond <- exp_design$condition[s]
      mu <- base_expr[i] + ifelse(cond == "treated", effect[i], 0)
      val <- 2^(rnorm(1, mean = mu, sd = 0.3)) # linear scale; runDiscovery log2-transforms internally
      intensity_rows[[k]] <- data.frame(
        GroupId = i, replicate = exp_design$sample_name[s],
        NormalisedIntensity = val, Imputed = 0
      )
      k <- k + 1
    }
  }
  gene_intensity <- do.call(rbind, intensity_rows)
  gene_metadata <- data.frame(GroupId = seq_len(n_genes), GeneNames = gene_ids,
                               GroupLabel = gene_ids, GroupLabelType = "Gene",
                               stringsAsFactors = FALSE)
  list(experiment_design = exp_design, gene_intensity = gene_intensity, gene_metadata = gene_metadata)
}

# Round-4 review (2026-09-29): regimes no bundled or public set covers.
synth_protein <- function(ed, n, seed, effect_cols = NULL, mnar = 0.10, mar = 0.03) {
  # Log2 intensities with a scaled-inverse-chi2 variance per feature (prior df 4),
  # 10% of features differential between conditions, optional additive effects
  # (effect_cols: named list of per-sample vectors, one coefficient per feature),
  # then per sample the lowest `mnar` fraction plus `mar` at random go missing
  # as Intensity=NA, Imputed=1. Linear scale out, like the bundled sets.
  set.seed(seed)
  conds <- sort(unique(ed$condition), method = "radix")
  ns <- nrow(ed)
  mu <- rnorm(n, 22, 2)
  s2 <- 0.3 * 4 / rchisq(n, 4)
  eff <- matrix(0, n, length(conds))
  de <- sample(n, n %/% 10)
  eff[de, -1] <- rnorm(length(de) * (length(conds) - 1), 0, 1)
  y <- mu + eff[, match(ed$condition, conds)] + matrix(rnorm(n * ns), n) * sqrt(s2)
  for (v in effect_cols) y <- y + outer(rnorm(n, 0, 0.5), v)
  for (j in seq_len(ns)) {
    miss <- y[, j] < quantile(y[, j], mnar, names = FALSE) | runif(n) < mar
    y[miss, j] <- NA
  }
  list(
    experiment_design = ed,
    protein_intensity = data.frame(GroupId = rep(seq_len(n), times = ns),
                                   replicate = rep(ed$sample_name, each = n),
                                   NormalisedIntensity = 2^as.vector(y),
                                   Imputed = as.integer(is.na(as.vector(y))),
                                   stringsAsFactors = FALSE),
    protein_metadata = data.frame(GroupId = seq_len(n), ProteinIds = sprintf("SYN%05d", seq_len(n)),
                                  GroupLabel = sprintf("SYN%05d", seq_len(n)), stringsAsFactors = FALSE))
}

build_synth_scale <- function() {
  # 50,000 features x 100 samples (4 conditions x 25).
  ed <- data.frame(sample_name = sprintf("S%03d", 1:100), condition = rep(c("A", "B", "C", "D"), each = 25),
                   stringsAsFactors = FALSE)
  synth_protein(ed, 50000, 20260930)
}

build_synth_blocking <- function() {
  # 3,000 features, unbalanced conditions (24/18/12), an 18-level batch of 3
  # samples each assigned across conditions, and a numeric age.
  set.seed(20261001)
  cond <- rep(c("ctrl", "doseA", "doseB"), times = c(24, 18, 12))
  ed <- data.frame(sample_name = sprintf("B%02d", seq_along(cond)), condition = cond,
                   batch = sprintf("batch%02d", sample(rep(1:18, each = 3))),
                   age = round(runif(length(cond), 30, 70)), stringsAsFactors = FALSE)
  batch_effect <- lapply(sort(unique(ed$batch)), function(b) as.numeric(ed$batch == b))
  synth_protein(ed, 3000, 20261002, effect_cols = c(batch_effect, list((ed$age - 50) / 10)))
}

# ---- Count fixtures (edgeR / DESeq2 corpus) ---------------------------------
count_long <- function(counts, ed) {
  # Gene x sample integer matrix -> the long gene_intensity / gene_metadata shape.
  ids <- rownames(counts)
  list(experiment_design = ed,
       gene_intensity = data.frame(GroupId = rep(seq_along(ids), times = ncol(counts)),
                                   replicate = rep(colnames(counts), each = nrow(counts)),
                                   NormalisedIntensity = as.numeric(counts),
                                   # Upstream marks every zero Imputed = 1; importFlexiData
                                   # stops on (0, Imputed = 0) even with skipLog2 (importFlexiData.R:254-264).
                                   Imputed = as.integer(as.numeric(counts) == 0),
                                   stringsAsFactors = FALSE),
       gene_metadata = data.frame(GroupId = seq_along(ids), GeneNames = ids, GroupLabel = ids,
                                  GroupLabelType = "Gene", stringsAsFactors = FALSE))
}

build_airway <- function() {
  # All 8 samples; the first 8000 Ensembl ids in sorted order, zeros included.
  e <- new.env(); data("airway", package = "airway", envir = e)
  se <- e$airway
  counts <- SummarizedExperiment::assay(se, "counts")
  counts <- counts[sort(rownames(counts), method = "radix")[1:8000], , drop = FALSE]
  cd <- as.data.frame(SummarizedExperiment::colData(se))
  ed <- data.frame(sample_name = colnames(counts), condition = as.character(cd$dex),
                   cell = as.character(cd$cell), avgLength = as.numeric(cd$avgLength),
                   stringsAsFactors = FALSE)
  count_long(counts, ed)
}

synth_counts <- function(reps, seed, n = 2000, n_outlier = 0) {
  # NB counts: log-normal gene means, dispersion 0.04 + 2/mu, 15% DE genes with
  # log2FC ~ N(0, 1.2) for doseA and doseB, library-size factors 0.5-2, a crossed
  # 2-level batch (+0.3 log2 on 30% of genes) and a numeric rin covariate. Gene 1
  # is all zero. n_outlier genes get one sample multiplied by 50 (Cook's case).
  set.seed(seed)
  conds <- c("ctrl", "doseA", "doseB")
  ed <- data.frame(sample_name = sprintf("S%02d", seq_len(3 * reps)), condition = rep(conds, each = reps),
                   batch = rep(c("b1", "b2"), length.out = 3 * reps),
                   rin = round(runif(3 * reps, 6, 10), 1), stringsAsFactors = FALSE)
  mu0 <- exp(rnorm(n, log(80), 2))
  lfc <- matrix(0, n, 3)
  de <- sample(n, round(0.15 * n))
  lfc[de, 2:3] <- rnorm(2 * length(de), 0, 1.2)
  bat <- ifelse(runif(n) < 0.3, 0.3, 0)
  sf <- exp(runif(nrow(ed), log(0.5), log(2)))
  mu <- mu0 * 2^(lfc[, match(ed$condition, conds)] + outer(bat, as.numeric(ed$batch == "b2"))) *
    rep(sf, each = n)
  disp <- 0.04 + 2 / mu0
  counts <- matrix(rnbinom(length(mu), mu = mu, size = 1 / disp), n)
  counts[1, ] <- 0L
  if (n_outlier > 0) {
    og <- 1 + sample(n - 1, n_outlier)
    for (g in og) { j <- sample(ncol(counts), 1); counts[g, j] <- counts[g, j] * 50L + 50L }
  }
  dimnames(counts) <- list(sprintf("SYNG%05d", seq_len(n)), ed$sample_name)
  count_long(counts, ed)
}

load_dataset <- function(dataset) {
  if (dataset == "airway") return(build_airway())
  if (dataset == "count_synth") return(synth_counts(4, 20261001))
  if (dataset == "count_synth_cooks") return(synth_counts(7, 20261002, n_outlier = 20))
  if (dataset == "gene_synth") return(build_gene_synth())
  if (dataset == "synth_scale") return(build_synth_scale())
  if (dataset == "synth_blocking") return(build_synth_blocking())
  # Public PTM / metabolite datasets converted by build_public_datasets.R.
  public_rds <- file.path(REPO_ROOT, "data-raw", "golden-corpus", "public", paste0(dataset, ".rds"))
  if (file.exists(public_rds)) return(readRDS(public_rds))
  load_rda_as_list(dataset)
}

# ---- Transform functions --------------------------------------------------
# Each takes (ds, entity_type) -- the raw dataset list -- and returns a
# modified dataset list of the same shape. Deterministic, no RNG unless
# seeded. Used for edge cases and blocking/covariate one-at-a-time runs that
# need data the bundled sets don't provide as-is.

apply_transform <- function(ds, entity_type, transform_name) {
  if (is.null(transform_name)) return(ds)
  fn <- get(paste0("transform_", transform_name), mode = "function")
  fn(ds, entity_type)
}

transform_subset_2reps <- function(ds, entity_type) {
  ik <- get_intensity_key(entity_type)
  ed <- ds$experiment_design
  ed <- ed[ed$condition %in% c("2h", "6h"), ]
  keep_samples <- unlist(lapply(split(ed$sample_name, ed$condition), function(s) sort(s)[1:2]))
  ed <- ed[ed$sample_name %in% keep_samples, ]
  it <- ds[[ik]]
  it <- it[it$replicate %in% keep_samples, ]
  ds$experiment_design <- ed
  ds[[ik]] <- it
  ds
}

transform_single_rep_condition <- function(ds, entity_type) {
  # Start from the 2-reps-per-condition subset, then drop one of the "2h"
  # replicates so that condition has exactly 1 replicate left.
  ds <- transform_subset_2reps(ds, entity_type)
  ik <- get_intensity_key(entity_type)
  ed <- ds$experiment_design
  drop_sample <- sort(ed$sample_name[ed$condition == "2h"])[1]
  ed <- ed[ed$sample_name != drop_sample, ]
  it <- ds[[ik]]
  it <- it[it$replicate != drop_sample, ]
  ds$experiment_design <- ed
  ds[[ik]] <- it
  ds
}

transform_collinear_covariate <- function(ds, entity_type) {
  ed <- ds$experiment_design
  ed$condition_dup <- ed$condition # perfectly collinear with condition -> rank-deficient design
  ds$experiment_design <- ed
  ds
}

transform_near_collinear_numeric <- function(ds, entity_type) {
  # Review B4 / fix-plan Step 3: a numeric covariate equal to one condition's
  # indicator plus 1e-9 noise. qr() at tol 1e-7 calls this rank-deficient;
  # numpy.linalg.matrix_rank does not.
  ed <- ds$experiment_design
  first <- sort(unique(ed$condition), method = "radix")[1]
  ed$near_dup <- as.numeric(ed$condition == first) + 1e-9 * seq_len(nrow(ed))
  ds$experiment_design <- ed
  ds
}

keep_first_features <- function(ds, entity_type, n) {
  ik <- get_intensity_key(entity_type); mk <- get_metadata_key(entity_type)
  keep <- sort(unique(ds[[ik]]$GroupId))[seq_len(n)]
  ds[[ik]] <- ds[[ik]][ds[[ik]]$GroupId %in% keep, ]
  ds[[mk]] <- ds[[mk]][ds[[mk]]$GroupId %in% keep, ]
  ds
}
# Fix-plan Step 4: pin the "> 3 features" gates (limmaStatsFun.R) at 3 and 4.
transform_first3_features <- function(ds, entity_type) keep_first_features(ds, entity_type, 3)
transform_first4_features <- function(ds, entity_type) keep_first_features(ds, entity_type, 4)

transform_single_rep_pair <- function(ds, entity_type) {
  # Fix-plan Step 4: 2h keeps 1 replicate and 6h keeps 2; each feature has one
  # of the two 6h values set to Intensity=NA/Imputed=1 (alternating by GroupId,
  # so neither sample is dropped). The pair passes the 3-sample and 50% filters,
  # but every feature has 2 observations for 2 coefficients (df 0), so the
  # separate 2h-vs-6h eBayes fails inside limmaStatsFun's tryCatch. 24h keeps
  # all replicates, so the omnibus fit has residual df.
  ik <- get_intensity_key(entity_type)
  ed <- ds$experiment_design
  s2h <- sort(ed$sample_name[ed$condition == "2h"]); s6h <- sort(ed$sample_name[ed$condition == "6h"])
  drop <- c(s2h[-1], s6h[-(1:2)])
  ds$experiment_design <- ed[!ed$sample_name %in% drop, ]
  it <- ds[[ik]][!ds[[ik]]$replicate %in% drop, ]
  blank <- ifelse(it$GroupId %% 2 == 0, it$replicate == s6h[1], it$replicate == s6h[2])
  it$NormalisedIntensity[blank] <- NA
  it$Imputed[blank] <- 1
  ds[[ik]] <- it
  ds
}

transform_condition_time_point <- function(ds, entity_type) {
  # Fix-plan Step 4: a non-syntactic condition column name (make.names).
  ed <- ds$experiment_design
  names(ed)[names(ed) == "condition"] <- "Time point"
  ds$experiment_design <- ed
  ds
}

transform_all_na_feature <- function(ds, entity_type) transform_all_na_gene(ds, entity_type)

transform_add_numeric_covariate <- function(ds, entity_type) {
  ed <- ds$experiment_design
  # Deterministic numeric covariate derived from sample order, not random.
  ed <- ed[order(ed$sample_name), ]
  ed$numeric_batch <- (seq_len(nrow(ed)) %% 3) + 1
  ds$experiment_design <- ed
  ds
}

transform_high_missingness <- function(ds, entity_type) {
  ik <- get_intensity_key(entity_type)
  it <- ds[[ik]]
  ed <- ds$experiment_design
  cond_target <- sort(unique(ed$condition))[1]
  target_samples <- sort(ed$sample_name[ed$condition == cond_target])
  is_target_row <- it$replicate %in% target_samples & it$Imputed == 0
  target_idx <- which(is_target_row)
  # Deterministically flip ~85% of the non-imputed rows in one condition to
  # Imputed=1 (every row except every 7th), simulating high missingness for
  # a filter-threshold boundary test.
  flip_idx <- target_idx[seq_along(target_idx) %% 7 != 0]
  it$Imputed[flip_idx] <- 1
  ds[[ik]] <- it
  ds
}

transform_restore_missing <- function(ds, entity_type) {
  # Production shape for un-imputed data (fix-plan Step 5): every Imputed==1 row
  # carries Intensity=NA, so the bundled imputed values are dropped.
  ik <- get_intensity_key(entity_type)
  it <- ds[[ik]]
  it$NormalisedIntensity[it$Imputed == 1] <- NA
  ds[[ik]] <- it
  ds
}

transform_all_na_gene <- function(ds, entity_type) {
  # Review B3: one gene with every row Intensity=NA, Imputed=1. The gene shim
  # (threshold 0) keeps it, and its NaN Amean reaches the trend covariate.
  ik <- get_intensity_key(entity_type)
  it <- ds[[ik]]
  it$NormalisedIntensity[it$GroupId == 1] <- NA
  it$Imputed[it$GroupId == 1] <- 1
  ds[[ik]] <- it
  ds
}

transform_mnar_low_abundance <- function(ds, entity_type) {
  # bojkova2020 is almost fully observed, so missingness is simulated the way it
  # arises in DDA/DIA data: in each sample the lowest 15% of observed intensities
  # become Intensity=NA, Imputed=1. Deterministic (no RNG).
  ik <- get_intensity_key(entity_type)
  it <- ds[[ik]]
  for (s in unique(it$replicate)) {
    rows <- which(it$replicate == s & it$Imputed == 0)
    cut <- quantile(it$NormalisedIntensity[rows], 0.15, names = FALSE, type = 7)
    drop <- rows[it$NormalisedIntensity[rows] < cut]
    it$NormalisedIntensity[drop] <- NA
    it$Imputed[drop] <- 1
  }
  it$NormalisedIntensity[it$Imputed == 1] <- NA
  ds[[ik]] <- it
  ds
}

transform_unicode_conditions <- function(ds, entity_type) {
  ik <- get_intensity_key(entity_type)
  relabel <- c("2h" = "2h_µg/mL", "6h" = "6h_°C", "10h" = "10h_café", "24h" = "24h")
  ed <- ds$experiment_design
  ed$condition <- unname(relabel[ed$condition])
  it <- ds[[ik]]
  it$condition <- unname(relabel[it$condition])
  ds$experiment_design <- ed
  ds[[ik]] <- it
  ds
}

transform_skip_branch_subset <- function(ds, entity_type) {
  # Curated small protein subset spanning conditions 2h/6h/24h so that, under
  # a count-filter threshold of 3, the 2h-vs-6h contrast has <=3 quantifiable
  # features (most marked Imputed in one of those two conditions) while the
  # 6h-vs-24h contrast retains enough features to fit. Exercises
  # limmaFitSeparateModels' per-contrast >3-feature skip.
  ik <- get_intensity_key(entity_type)
  ed <- ds$experiment_design
  ed <- ed[ed$condition %in% c("2h", "6h", "24h"), ]
  it <- ds[[ik]]
  it <- it[it$replicate %in% ed$sample_name, ]
  group_ids <- sort(unique(it$GroupId))[1:20]
  it <- it[it$GroupId %in% group_ids, ]
  samples_2h <- sort(ed$sample_name[ed$condition == "2h"])
  # Under one_condition logic a feature valid in EITHER condition passes, so
  # to starve the 2h-vs-6h contrast the flipped GroupIds must fall below the
  # count threshold (3) in BOTH conditions: impute every 2h sample and all but
  # 2 of the 6h samples. 24h stays untouched, so 6h-vs-24h keeps all 20
  # features quantifiable and still fits.
  samples_6h <- sort(ed$sample_name[ed$condition == "6h"])
  samples_6h_flip <- samples_6h[-(1:2)]
  keep_ids_2h <- group_ids[1:2]
  flip <- it$replicate %in% c(samples_2h, samples_6h_flip) &
    !(it$GroupId %in% keep_ids_2h)
  it$Imputed[flip] <- 1
  # Zero the intensity too: importFlexiData asserts Imputed==1 <=> intensity==0
  # and the validity filter keys off that pairing, so Imputed alone is inert.
  it$NormalisedIntensity[flip] <- 0
  ds$experiment_design <- ed
  ds[[ik]] <- it
  ds
}

transform_negative_intensity <- function(ds, entity_type) {
  ik <- get_intensity_key(entity_type)
  it <- ds[[ik]]
  idx <- which(it$Imputed == 0)[1]
  it$NormalisedIntensity[idx] <- -abs(it$NormalisedIntensity[idx]) - 1
  ds[[ik]] <- it
  ds
}

transform_single_condition_level <- function(ds, entity_type) {
  ik <- get_intensity_key(entity_type)
  ed <- ds$experiment_design
  keep_cond <- sort(unique(ed$condition))[1]
  ed <- ed[ed$condition == keep_cond, ]
  it <- ds[[ik]]
  it <- it[it$replicate %in% ed$sample_name, ]
  ds$experiment_design <- ed
  ds[[ik]] <- it
  ds
}

transform_blocking_split <- function(ds, entity_type) {
  # Mirrors data-raw/test-data-blocking.R: bojkova2020's sample_name is
  # "<group>_<condition>_<subject>"; split it out so `subject` is available
  # as a categorical covariate.
  ed <- ds$experiment_design
  parts <- do.call(rbind, strsplit(ed$sample_name, "_", fixed = TRUE))
  ed$group <- parts[, 1]
  ed$condition <- parts[, 2]
  ed$subject <- parts[, 3]
  ds$experiment_design <- ed
  ds
}

# ---- Round-3 review transforms (2026-09-29) ---------------------------------
blank_rows <- function(ds, entity_type, rows) {
  ik <- get_intensity_key(entity_type)
  ds[[ik]]$NormalisedIntensity[rows] <- NA
  ds[[ik]]$Imputed[rows] <- 1
  ds
}

transform_two_informative_pair <- function(ds, entity_type) {
  # Round-3 must-fix 1: four features in the 2h-vs-6h separate model. Feature 1
  # is complete (df 10), feature 2 loses one 2h value (df 9), features 3-4 keep
  # one value per condition (df 0). That leaves exactly two informative
  # variances with unequal df, the n.informative == 2 branch of
  # fitFDistUnequalDF1. 10h and 24h stay complete, so the other fits have df.
  ds <- keep_first_features(ds, entity_type, 4)
  ik <- get_intensity_key(entity_type)
  it <- ds[[ik]]
  ed <- ds$experiment_design
  s2h <- sort(ed$sample_name[ed$condition == "2h"]); s6h <- sort(ed$sample_name[ed$condition == "6h"])
  ids <- sort(unique(it$GroupId))
  rows <- (it$GroupId == ids[2] & it$replicate == s2h[1]) |
    (it$GroupId %in% ids[3:4] & it$replicate %in% c(s2h[-1], s6h[-1]))
  blank_rows(ds, entity_type, which(rows))
}

transform_mnar_blocking <- function(ds, entity_type) {
  # Round-3 must-fix 2: NA intensities plus a categorical covariate, so
  # contrasts.fit takes its non-orthogonal approximation.
  transform_mnar_low_abundance(transform_blocking_split(ds, entity_type), entity_type)
}

transform_restore_missing_numeric_cov <- function(ds, entity_type) {
  transform_add_numeric_covariate(transform_restore_missing(ds, entity_type), entity_type)
}

transform_soft_mnar <- function(ds, entity_type) {
  # Round-3 must-fix 4: missingness that mixes MNAR and MAR at a higher rate.
  # Per sample, values below the 30th percentile go missing with probability
  # 0.6 and every other value with probability 0.05. Seeded.
  set.seed(20260929)
  ik <- get_intensity_key(entity_type)
  it <- ds[[ik]]
  for (s in sort(unique(it$replicate))) {
    rows <- which(it$replicate == s & it$Imputed == 0)
    x <- it$NormalisedIntensity[rows]
    p <- ifelse(x < quantile(x, 0.30, names = FALSE, type = 7), 0.6, 0.05)
    drop <- rows[runif(length(rows)) < p]
    it$NormalisedIntensity[drop] <- NA
    it$Imputed[drop] <- 1
  }
  it$NormalisedIntensity[it$Imputed == 1] <- NA
  ds[[ik]] <- it
  ds
}

transform_mnar_all_na <- function(ds, entity_type) {
  # Unequal-df path plus one feature with no observations at all.
  transform_all_na_feature(transform_mnar_low_abundance(ds, entity_type), entity_type)
}

skip_branch_keep <- function(ds, entity_type, n_keep) {
  # Same starvation as skip_branch_subset, but n_keep GroupIds stay valid in
  # the 2h-vs-6h pair, pinning the per-pair "> 3" gate at exactly 3 and 4.
  ik <- get_intensity_key(entity_type)
  ed <- ds$experiment_design
  ed <- ed[ed$condition %in% c("2h", "6h", "24h"), ]
  it <- ds[[ik]]
  it <- it[it$replicate %in% ed$sample_name, ]
  group_ids <- sort(unique(it$GroupId))[1:20]
  it <- it[it$GroupId %in% group_ids, ]
  s2h <- sort(ed$sample_name[ed$condition == "2h"])
  s6h <- sort(ed$sample_name[ed$condition == "6h"])
  flip <- it$replicate %in% c(s2h, s6h[-(1:2)]) & !(it$GroupId %in% group_ids[seq_len(n_keep)])
  it$Imputed[flip] <- 1
  it$NormalisedIntensity[flip] <- 0
  ds$experiment_design <- ed
  ds[[ik]] <- it
  ds
}
transform_all_pairs_starved <- function(ds, entity_type) {
  # GroupIds 1-2 are valid only in 2h and 3-4 only in 24h; the rest keep two
  # valid 6h values. Under a count threshold of 3 the omnibus keeps 4 features
  # (it passes the > 3 gate) while 2h-vs-6h and 6h-vs-24h keep 2 each.
  ik <- get_intensity_key(entity_type)
  ed <- ds$experiment_design
  ed <- ed[ed$condition %in% c("2h", "6h", "24h"), ]
  it <- ds[[ik]]
  it <- it[it$replicate %in% ed$sample_name, ]
  ids <- sort(unique(it$GroupId))[1:20]
  it <- it[it$GroupId %in% ids, ]
  s2h <- ed$sample_name[ed$condition == "2h"]; s24h <- ed$sample_name[ed$condition == "24h"]
  s6h_keep <- sort(ed$sample_name[ed$condition == "6h"])[1:2]
  valid <- (it$GroupId %in% ids[1:2] & it$replicate %in% s2h) |
    (it$GroupId %in% ids[3:4] & it$replicate %in% s24h) |
    (it$GroupId %in% ids[5:20] & it$replicate %in% s6h_keep)
  it$Imputed[!valid] <- 1
  it$NormalisedIntensity[!valid] <- 0
  ds$experiment_design <- ed
  ds[[ik]] <- it
  ds
}
transform_skip_branch_keep3 <- function(ds, entity_type) skip_branch_keep(ds, entity_type, 3)
transform_skip_branch_keep4 <- function(ds, entity_type) skip_branch_keep(ds, entity_type, 4)

relabel_conditions <- function(ds, entity_type, relabel) {
  ik <- get_intensity_key(entity_type)
  ds$experiment_design$condition <- unname(relabel[ds$experiment_design$condition])
  ds[[ik]]$condition <- unname(relabel[ds[[ik]]$condition])
  ds
}

transform_reserved_word_conditions <- function(ds, entity_type) {
  # make.names keeps non-ASCII letters and appends "." to reserved words.
  relabel_conditions(ds, entity_type, c("2h" = "TRUE", "6h" = "été6h", "10h" = "function", "24h" = "24h"))
}

relabel_entity <- function(ds, entity_type) {
  # bojkova2020 served as another entity type, so importFeaturesMetadataTable
  # takes that entity's branch. The numbers are protein numbers.
  md <- ds$protein_metadata
  if (entity_type == "ptm") md$PTMProtein <- md$ProteinIds
  if (entity_type == "metabolite") {
    md$MetaboliteId <- paste0("MET", md$GroupId)
    md$ProteinIds <- NULL
  }
  ds[[get_intensity_key(entity_type)]] <- ds$protein_intensity
  ds[[get_metadata_key(entity_type)]] <- md
  ds$protein_intensity <- NULL
  ds$protein_metadata <- NULL
  ds
}
transform_as_ptm <- function(ds, entity_type) relabel_entity(ds, entity_type)
transform_as_metabolite <- function(ds, entity_type) relabel_entity(ds, entity_type)

transform_one_level_in_pair <- function(ds, entity_type) {
  # A categorical covariate that is constant inside the 2h-vs-6h pair and
  # varies elsewhere.
  ed <- ds$experiment_design
  ed$batch <- ifelse(ed$condition %in% c("2h", "6h"), "A", ifelse(ed$group == "Control", "A", "B"))
  ds$experiment_design <- ed
  ds
}

transform_empty_string_covariate <- function(ds, entity_type) {
  # The third Control replicate of each condition carries "" in the covariate.
  ds <- transform_blocking_split(ds, entity_type)
  ed <- ds$experiment_design
  ed$subject[ed$subject == "3" & ed$group == "Control"] <- ""
  ds$experiment_design <- ed
  ds
}

transform_raw_intensity_only <- function(ds, entity_type) {
  ik <- get_intensity_key(entity_type)
  names(ds[[ik]])[names(ds[[ik]]) == "NormalisedIntensity"] <- "Intensity"
  ds
}

transform_two_samples <- function(ds, entity_type) {
  # One 2h and one 6h sample only: fewer than 3 samples across the contrast.
  ik <- get_intensity_key(entity_type)
  ed <- ds$experiment_design
  keep <- c(sort(ed$sample_name[ed$condition == "2h"])[1], sort(ed$sample_name[ed$condition == "6h"])[1])
  ds$experiment_design <- ed[ed$sample_name %in% keep, ]
  ds[[ik]] <- ds[[ik]][ds[[ik]]$replicate %in% keep, ]
  ds
}

transform_gene_libsize <- function(ds, entity_type) {
  # One sample scaled 5x on the linear scale: library sizes differ by > 3x.
  ik <- get_intensity_key(entity_type)
  s <- sort(unique(ds[[ik]]$replicate))[1]
  rows <- ds[[ik]]$replicate == s
  ds[[ik]]$NormalisedIntensity[rows] <- ds[[ik]]$NormalisedIntensity[rows] * 5
  ds
}

# ---- Round-4 review transforms (2026-09-29) ---------------------------------
transform_first10_features <- function(ds, entity_type) keep_first_features(ds, entity_type, 10)
transform_first30_features <- function(ds, entity_type) keep_first_features(ds, entity_type, 30)

transform_drop_missing_rows <- function(ds, entity_type) {
  # Sparse long table: missing values are absent rows, not NA/Imputed=1 rows.
  ik <- get_intensity_key(entity_type)
  ds[[ik]] <- ds[[ik]][!is.na(ds[[ik]]$NormalisedIntensity), ]
  ds
}

transform_na_not_imputed <- function(ds, entity_type) {
  # Intensity NA with Imputed = 0 (spec 10 Q6): missing but not flagged imputed.
  ik <- get_intensity_key(entity_type)
  ds[[ik]]$Imputed[is.na(ds[[ik]]$NormalisedIntensity)] <- 0L
  ds
}

# ---- Count-corpus transforms ---------------------------------------------------
transform_tiny_counts <- function(ds, entity_type) {
  ik <- get_intensity_key(entity_type)
  ds[[ik]]$NormalisedIntensity <- pmin(ds[[ik]]$NormalisedIntensity, 1)
  ds[[ik]]$Imputed <- as.integer(ds[[ik]]$NormalisedIntensity == 0)
  ds
}
transform_one_rep_per_condition <- function(ds, entity_type) {
  ik <- get_intensity_key(entity_type)
  ed <- ds$experiment_design
  keep <- vapply(split(ed$sample_name, ed$condition), function(s) sort(s)[1], "")
  ds$experiment_design <- ed[ed$sample_name %in% keep, ]
  ds[[ik]] <- ds[[ik]][ds[[ik]]$replicate %in% keep, ]
  ds
}
transform_drop_count_rows <- function(ds, entity_type) {
  # Every 19th long-table row is absent; the engines see those cells as NA -> 0. The table is
  # sample-major, so a stride dividing the gene count (20 into 2000) would drop whole genes;
  # 19 is coprime to 2000 and exceeds the 12 samples, so each gene loses at most one cell.
  ik <- get_intensity_key(entity_type)
  ds[[ik]] <- ds[[ik]][seq_len(nrow(ds[[ik]])) %% 19 != 0, ]
  ds
}
transform_unicode_count_conditions <- function(ds, entity_type) {
  relabel <- c(ctrl = "0 µM", doseA = "dose-A (5°C)", doseB = "TRUE")
  ds$experiment_design$condition <- unname(relabel[ds$experiment_design$condition])
  ds
}
transform_cooks_rescue <- function(ds, entity_type) {
  # DESeq2's two-group Cook's rescue (results(), cooksCutoff): a flagged count that at least 3
  # other samples exceed keeps its p-value. Airway genes with every count >= 500, in id order:
  # in the first 6 the trt counts are divided by 20 and one trt sample is set to half the smallest
  # untrt count, an outlier in its own group that all 4 untrt samples exceed (rescued); in the
  # next 3 one untrt count is multiplied by 50, which nothing exceeds (flagged, p-value NA).
  ik <- get_intensity_key(entity_type)
  it <- ds[[ik]]
  ed <- ds$experiment_design
  trt <- sort(ed$sample_name[ed$condition == "trt"]); untrt <- sort(ed$sample_name[ed$condition == "untrt"])
  mins <- tapply(it$NormalisedIntensity, it$GroupId, min)
  genes <- as.integer(names(mins)[mins >= 500])[1:9]
  for (k in seq_along(genes)) {
    g <- it$GroupId == genes[k]
    if (k <= 6) {
      t_rows <- which(g & it$replicate %in% trt)
      it$NormalisedIntensity[t_rows] <- round(it$NormalisedIntensity[t_rows] / 20)
      out <- which(g & it$replicate == trt[(k - 1) %% length(trt) + 1])
      it$NormalisedIntensity[out] <- round(min(it$NormalisedIntensity[g & it$replicate %in% untrt]) / 2)
    } else {
      out <- which(g & it$replicate == untrt[(k - 1) %% length(untrt) + 1])
      it$NormalisedIntensity[out] <- it$NormalisedIntensity[out] * 50
    }
  }
  it$Imputed <- as.integer(it$NormalisedIntensity == 0)
  ds[[ik]] <- it
  ds
}
transform_non_integer_counts <- function(ds, entity_type) {
  ik <- get_intensity_key(entity_type)
  ds[[ik]]$NormalisedIntensity <- ds[[ik]]$NormalisedIntensity + 0.25
  ds
}

# ---- Gene-entity shim override --------------------------------------------
# Replicates src/md_flexi_comparisons/process_r.py:733-740 verbatim. The run
# matrix stores the internal R-facing filter_logic/filter_criteria keys
# (post the process.R label map), so this override sets those keys directly
# rather than re-implementing the English-label -> internal-key dictionary.
apply_gene_shim <- function(run) {
  if (!isTRUE(run$gene_shim)) return(run)
  run$filter_threshold <- 0
  run$filter_criteria <- "percentage"
  run$filter_logic <- "one_condition"
  run$fit_separate_models <- FALSE
  run$limma_trend <- TRUE
  run
}

# ---- Builders for controlColsDF / customComparisonsDF ---------------------
build_control_cols_df <- function(control_cols) {
  if (is.null(control_cols)) return(NULL)
  data.frame(Column = control_cols$Column, Type = control_cols$Type, stringsAsFactors = FALSE)
}
build_custom_comparisons_df <- function(custom_comparisons) {
  if (is.null(custom_comparisons)) return(NULL)
  data.frame(left = custom_comparisons$left, right = custom_comparisons$right, stringsAsFactors = FALSE)
}

# ---- Self-check ------------------------------------------------------------
# Independently rebuilds the limma fit via the same exported building blocks
# runDiscovery calls internally (importFlexiData -> getComparisonsDF ->
# fitOneConditionWithContrasts), then diffs its numeric columns against the
# end-to-end runDiscovery()/runANOVA() output on the tolerance policy above.
# This only targets de_method="limma". The edgeR / DESeq2 self-check is the
# standalone R reference (PR-E2), so count runs record performed = FALSE here.
self_check_run <- function(experiment_design, intensity, metadata, entity_type,
                            condition_col, control_cols_df, custom_comparisons_df,
                            comparison_type, fit_separate_models, return_decide_test_column,
                            filter_logic, filter_criteria, filter_threshold,
                            limma_trend, limma_robust, end_to_end_df) {
  out <- list(performed = FALSE, passed = NA, detail = NULL, columns_compared = character(0))
  result <- tryCatch({
    intensity_col <- if ("NormalisedIntensity" %in% colnames(intensity)) "NormalisedIntensity" else "Intensity"
    imported <- importFlexiData(experimentDesign = experiment_design, intensitiesTable = intensity,
                                 conditionCol = condition_col, controlColsDF = control_cols_df,
                                 groupIdCol = "GroupId", intensityColumn = intensity_col, skipLog2 = FALSE)
    comparisonDF <- getComparisonsDF(experimentDesign = imported$experimentDesign, conditionCol = condition_col,
                                      customComparisonsDF = custom_comparisons_df, comparisonType = comparison_type)
    fitted <- fitOneConditionWithContrasts(
      intensitiesTable = imported$longIntensityDT, experimentDesign = imported$experimentDesign,
      customComparisonsCond = comparisonDF, conditionCol = condition_col, controlCols = imported$controlCols,
      intensityCol = intensityColumnInternal(), groupIdCol = "GroupId", conditionsDict = imported$conditionsDict[[condition_col]],
      fitSeparateModels = fit_separate_models, returnDecideTestColumn = return_decide_test_column,
      conditionSeparator = " - ", filterValidValuesLogic = filter_logic, filterValidValuesCriteria = filter_criteria,
      filterValidValuesThreshold = filter_threshold, limmaTrend = limma_trend, limmaRobust = limma_robust)
    building_block_df <- as.data.frame(fitted$limmaStats) # avoid data.table `[.data.table` NSE surprises below
    building_block_df$GroupId <- as.character(building_block_df$GroupId)
    e2e <- end_to_end_df
    e2e$GroupId <- as.character(e2e$GroupId)

    numeric_cols_bb <- names(building_block_df)[vapply(building_block_df, is.numeric, logical(1))]
    common_cols <- intersect(numeric_cols_bb, names(e2e))
    common_cols <- setdiff(common_cols, "GroupId")
    if (length(common_cols) == 0) {
      return(list(performed = TRUE, passed = NA,
                   detail = "No common numeric columns found between building-block and end-to-end output; skipping numeric diff.",
                   columns_compared = character(0)))
    }
    merged <- merge(building_block_df[, c("GroupId", common_cols)], e2e[, c("GroupId", common_cols)],
                     by = "GroupId", suffixes = c("_bb", "_e2e"))
    col_report <- list()
    all_pass <- TRUE
    for (cc in common_cols) {
      bb_v <- merged[[paste0(cc, "_bb")]]
      e2e_v <- merged[[paste0(cc, "_e2e")]]
      ok <- is.numeric(bb_v) && is.numeric(e2e_v)
      if (!ok) next
      # NA/NaN must sit in the same cells; the diff runs over the cells both
      # sides have. An all-NaN column (edge_two_informative_*) then passes.
      na_mismatch <- sum(xor(is.na(bb_v), is.na(e2e_v)))
      both <- !is.na(bb_v) & !is.na(e2e_v)
      abs_diff <- abs(bb_v[both] - e2e_v[both])
      rel_diff <- abs_diff / pmax(abs(e2e_v[both]), 1e-12)
      max_abs <- if (any(both)) max(abs_diff) else 0
      max_rel <- if (any(both)) max(rel_diff) else 0
      # Loose generic gate (1e-6 abs OR 1e-6 rel); per-column tolerance
      # policy in TOLERANCE_POLICY is the authority consumers should apply.
      pass <- na_mismatch == 0 && is.finite(max_abs) && (max_abs < 1e-6 || max_rel < 1e-6)
      if (!isTRUE(pass)) all_pass <- FALSE
      col_report[[cc]] <- list(max_abs_diff = max_abs, max_rel_diff = max_rel,
                               na_mismatch = na_mismatch, pass = isTRUE(pass))
    }
    list(performed = TRUE, passed = all_pass, detail = col_report, columns_compared = common_cols)
  }, error = function(e) {
    list(performed = FALSE, passed = NA, detail = paste("Self-check raised:", conditionMessage(e)),
         columns_compared = character(0))
  })
  result
}

# ---- Per-run execution ------------------------------------------------------
run_one <- function(run) {
  message(sprintf("[%s] %s", run$id, run$description))
  run <- apply_gene_shim(run)

  run_dir <- file.path(CORPUS_DIR, "runs", run$id)
  dir.create(run_dir, recursive = TRUE, showWarnings = FALSE)

  raw_ds <- load_dataset(run$dataset)
  ds <- apply_transform(raw_ds, run$entity_type, run$transform)

  dataset_key <- if (is.null(run$transform)) run$dataset else paste0(run$dataset, "__", run$transform)
  shared_dir <- file.path(CORPUS_DIR, "shared", dataset_key)
  shared_written <- file.exists(file.path(shared_dir, "experiment_design.rds"))
  if (!shared_written) {
    dir.create(shared_dir, recursive = TRUE, showWarnings = FALSE)
    saveRDS(ds$experiment_design, file.path(shared_dir, "experiment_design.rds"))
    saveRDS(ds[[get_intensity_key(run$entity_type)]], file.path(shared_dir, "intensity.rds"))
    saveRDS(ds[[get_metadata_key(run$entity_type)]], file.path(shared_dir, "metadata.rds"))
  }

  control_cols_df <- build_control_cols_df(run$control_cols)
  custom_comparisons_df <- build_custom_comparisons_df(run$custom_comparisons)
  if (!is.null(control_cols_df)) saveRDS(control_cols_df, file.path(run_dir, "control_cols_df.rds"))
  if (!is.null(custom_comparisons_df)) saveRDS(custom_comparisons_df, file.path(run_dir, "custom_comparisons_df.rds"))

  common_args <- list(
    experimentDesign = ds$experiment_design,
    intensitiesTable = ds[[get_intensity_key(run$entity_type)]],
    featuresMetadataTable = ds[[get_metadata_key(run$entity_type)]],
    conditionCol = run$condition_col,
    entityType = run$entity_type,
    controlColsDF = control_cols_df,
    groupIdCol = "GroupId",
    comparisonType = run$comparison_type,
    customComparisonsDF = custom_comparisons_df,
    returnDecideTestColumn = run$return_decide_test_column,
    conditionSeparator = " - ",
    filterValidValuesLogic = run$filter_logic,
    filterValidValuesCriteria = run$filter_criteria,
    filterValidValuesThreshold = run$filter_threshold,
    limmaTrend = run$limma_trend,
    limmaRobust = run$limma_robust,
    de_method = run$de_method,
    edger_norm_method = run$edger_norm_method,
    deseq2_lfc_shrinkage = run$deseq2_lfc_shrinkage,
    deseq2_alpha = run$deseq2_alpha,
    apeglm_seed = run$apeglm_seed,
    returnRunMetadata = TRUE
  )

  manifest <- list(
    id = run$id, description = run$description, dataset = run$dataset, entity_type = run$entity_type,
    mode = run$mode, params = run[c("comparison_type", "custom_comparisons", "control_cols",
                                     "fit_separate_models", "filter_logic", "filter_criteria",
                                     "filter_threshold", "limma_trend", "limma_robust",
                                     "return_decide_test_column", "transform", "gene_shim", "condition_col",
                                     "de_method", "edger_norm_method", "deseq2_lfc_shrinkage",
                                     "deseq2_alpha", "apeglm_seed")],
    expected_error = run$expected_error, boundary_sensitive = run$boundary_sensitive, notes = run$notes,
    shared_inputs_dir = file.path("shared", dataset_key),
    lc_collate = Sys.getenv("LC_COLLATE"),
    r_version = R.version.string,
    package_versions = list(
      MDFlexiComparisons = "sourced directly from R/*.R (not installed as a package); no DESCRIPTION Version field consulted",
      limma = as.character(packageVersion("limma")),
      Biobase = as.character(packageVersion("Biobase")),
      SummarizedExperiment = as.character(packageVersion("SummarizedExperiment")),
      data.table = as.character(packageVersion("data.table")),
      edgeR = pkg_version_or_na("edgeR"),
      DESeq2 = pkg_version_or_na("DESeq2"),
      apeglm = pkg_version_or_na("apeglm"),
      ashr = pkg_version_or_na("ashr"),
      mixsqp = pkg_version_or_na("mixsqp"),
      statmod = pkg_version_or_na("statmod"),
      locfit = pkg_version_or_na("locfit"),
      bioconductor = if (requireNamespace("BiocManager", quietly = TRUE)) as.character(BiocManager::version()) else NA_character_
    ),
    blas = La_library(),
    provenance = PROVENANCE,
    tolerance_policy = if (CORPUS_KIND == "count") COUNT_TOLERANCE_POLICY else TOLERANCE_POLICY,
    arrow_available = HAS_ARROW,
    output_format = if (HAS_ARROW) "parquet" else "rds"
  )
  if (!is.null(run$known_production_behaviour)) manifest$known_production_behaviour <- run$known_production_behaviour

  # Warnings (e.g. "Not enough replicates per level...") must not abort the
  # run or unwind past a later real error: collect+muffle them with
  # withCallingHandlers (which does not unwind the stack), and let a genuine
  # error propagate to the outer tryCatch regardless of whether a warning
  # fired first.
  warnings_collected <- character(0)
  caught <- tryCatch({
    val <- withCallingHandlers({
      if (run$mode == "discovery") {
        do.call(runDiscovery, c(common_args, list(fitSeparateModels = run$fit_separate_models, outputType = "df")))
      } else if (run$mode == "anova") {
        anova_args <- common_args
        anova_args$fitSeparateModels <- NULL
        do.call(runANOVA, c(anova_args, list(outputType = "df")))
      } else {
        stop(sprintf("Unknown run mode: %s", run$mode))
      }
    }, warning = function(w) {
      warnings_collected[[length(warnings_collected) + 1]] <<- conditionMessage(w)
      invokeRestart("muffleWarning")
    })
    list(ok = TRUE, value = val)
  }, error = function(e) list(ok = FALSE, error = conditionMessage(e)))
  if (length(warnings_collected) > 0) caught$warning_text <- paste(warnings_collected, collapse = " | ")

  if (!isTRUE(caught$ok)) {
    manifest$status <- "error"
    manifest$actual_error <- caught$error
    manifest$warning_text <- caught$warning_text %||% NULL
    manifest$error_matches_expected <- if (is.null(run$expected_error)) FALSE else grepl(run$expected_error, caught$error, fixed = TRUE)
    if (is.null(run$expected_error)) {
      warning(sprintf("[%s] UNEXPECTED ERROR: %s", run$id, caught$error))
    } else if (!manifest$error_matches_expected) {
      warning(sprintf("[%s] error text does not match expected substring.\n  expected: %s\n  actual:   %s",
                       run$id, run$expected_error, caught$error))
    }
    writeLines(toJSON(manifest, auto_unbox = TRUE, null = "null", na = "null", pretty = TRUE),
               file.path(run_dir, "manifest.json"))
    return(invisible(manifest))
  }

  if (!is.null(run$expected_error)) {
    warning(sprintf("[%s] expected an error containing '%s' but the run succeeded.", run$id, run$expected_error))
  }

  out_df <- caught$value$out
  run_metadata_df <- caught$value$runMetadata

  save_table(out_df, file.path(run_dir, "results"))
  saveRDS(run_metadata_df, file.path(run_dir, "run_metadata.rds"))

  sc <- if (run$de_method != "limma") {
    list(performed = FALSE, passed = NA, columns_compared = character(0),
         detail = "edgeR/DESeq2: self-check is the standalone R reference (PR-E2), not this harness.")
  } else self_check_run(
    experiment_design = ds$experiment_design, intensity = ds[[get_intensity_key(run$entity_type)]],
    metadata = ds[[get_metadata_key(run$entity_type)]], entity_type = run$entity_type,
    condition_col = run$condition_col, control_cols_df = control_cols_df, custom_comparisons_df = custom_comparisons_df,
    comparison_type = run$comparison_type, fit_separate_models = if (run$mode == "anova") FALSE else run$fit_separate_models,
    return_decide_test_column = run$return_decide_test_column, filter_logic = run$filter_logic,
    filter_criteria = run$filter_criteria, filter_threshold = run$filter_threshold,
    limma_trend = run$limma_trend, limma_robust = run$limma_robust, end_to_end_df = out_df
  )

  manifest$status <- "ok"
  manifest$n_rows <- nrow(out_df)
  manifest$n_cols <- ncol(out_df)
  manifest$columns <- colnames(out_df)
  manifest$warning_text <- caught$warning_text %||% NULL
  manifest$self_check <- sc

  writeLines(toJSON(manifest, auto_unbox = TRUE, null = "null", na = "null", pretty = TRUE),
             file.path(run_dir, "manifest.json"))
  invisible(manifest)
}

`%||%` <- function(a, b) if (is.null(a)) b else a
pkg_version_or_na <- function(p) tryCatch(as.character(utils::packageVersion(p)), error = function(e) NA_character_)

# ---- Entry point ------------------------------------------------------------
run_corpus <- function(run_ids = NULL) {
  source(file.path(REPO_ROOT, "data-raw", "golden-corpus", "run_matrix.R"), local = (env <- new.env()))
  all_runs <- get(if (CORPUS_KIND == "count") "count_runs" else "runs", envir = env)
  if (!is.null(run_ids)) {
    unknown <- setdiff(run_ids, names(all_runs))
    if (length(unknown) > 0) stop("Unknown run id(s): ", paste(unknown, collapse = ", "))
    all_runs <- all_runs[run_ids]
  }

  summary_rows <- list()
  for (id in names(all_runs)) {
    m <- tryCatch(run_one(all_runs[[id]]), error = function(e) {
      warning(sprintf("[%s] generator-level failure: %s", id, conditionMessage(e)))
      list(id = id, status = "generator_error", error = conditionMessage(e))
    })
    summary_rows[[id]] <- list(id = id, status = m$status %||% "unknown",
                                error_matches_expected = m$error_matches_expected %||% NA,
                                self_check_passed = if (!is.null(m$self_check)) m$self_check$passed else NA)
  }
  # A subset run adds to the existing index rather than replacing it.
  index_path <- file.path(CORPUS_DIR, "index.json")
  if (!is.null(run_ids) && file.exists(index_path)) {
    previous <- fromJSON(index_path, simplifyVector = FALSE)$runs
    previous[names(summary_rows)] <- summary_rows
    summary_rows <- previous
  }
  index <- list(generated_at = as.character(Sys.time()), n_runs = length(summary_rows),
                corpus_dir = CORPUS_DIR, arrow_available = HAS_ARROW, runs = summary_rows)
  if (CORPUS_KIND == "count") index <- c(list(kind = "count", versions_status = PROVENANCE$versions_status), index)
  writeLines(toJSON(index, auto_unbox = TRUE, null = "null", na = "null", pretty = TRUE),
             file.path(CORPUS_DIR, "index.json"))
  invisible(index)
}

if (sys.nframe() == 0) {
  args <- commandArgs(trailingOnly = TRUE)
  if (length(args) > 0) {
    run_corpus(run_ids = args)
  } else {
    run_corpus()
  }
}
