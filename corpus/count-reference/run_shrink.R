# Shrinkage goldens for shrink-core: the DESeq2 reference plus ref_shrink.R.
#
#   Rscript count-reference/run_shrink.R [run_id ...]
#
# Writes <corpus>/reference-shrink/<run_id>/ only; <corpus>/reference/ is never
# touched. With no ids it runs every count_deseq2_*shrink* run. Each run directory
# carries the full DESeq2 reference dump (so new shrink runs also get deseq2-core
# goldens) plus the shrink_* intermediates, and one reference.json for both.

args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("^--file=", "", grep("^--file=", args, value = TRUE))))
source(file.path(here, "ref_common.R"))
source(file.path(here, "ref_edger.R"))
source(file.path(here, "ref_deseq2.R"))
source(file.path(here, "ref_shrink.R"))
OUT_ROOT <- file.path(CORPUS_DIR, "reference-shrink")

# Production post-processing: the engine table becomes the run output by the
# same left join to all gene ids and integer GroupId, then the ANOVA shaping.
finish_table <- function(stats, inp) {
  allDT <- data.table(GroupId = rownames(inp$countMatrix))
  out <- merge(allDT, stats, by = "GroupId", all.x = TRUE)
  out <- type_convert(out, "integer", "GroupId")
  out <- as.data.frame(out)
  if (identical(inp$mode, "anova"))
    out <- .packageANOVAOutput(out, runMetadata = NULL, returnRunMetadata = FALSE)
  out
}

run_one_reference <- function(run_id) {
  manifest <- fromJSON(file.path(CORPUS_DIR, "runs", run_id, "manifest.json"), simplifyVector = TRUE)
  d <- new_dumper(file.path(OUT_ROOT, run_id))
  set.seed(NULL)
  res <- tryCatch({
    inp <- prepare_inputs(manifest)
    dump_inputs(inp, d)
    stats <- switch(manifest$params$de_method,
      edgeR  = ref_edger(inp, d),
      DESeq2 = { st <- ref_deseq2(inp, d); ref_shrink(inp, d); st },
      stop("reference: unsupported de_method ", manifest$params$de_method))
    list(ok = TRUE, out = finish_table(stats, inp))
  }, error = function(e) list(ok = FALSE, message = conditionMessage(e)))

  if (identical(manifest$status, "error")) {
    expected <- manifest$expected_error
    passed <- !res$ok && grepl(expected, res$message, fixed = TRUE)
    sc <- list(passed = passed, kind = "expected_error",
               expected = expected, got = if (res$ok) "no error" else res$message)
  } else if (!res$ok) {
    sc <- list(passed = FALSE, kind = "unexpected_error", got = res$message)
  } else {
    e2e <- readRDS(file.path(CORPUS_DIR, "runs", run_id, "results.rds"))
    sc <- self_check(res$out, e2e)
    sc$kind <- "table"
    sc$e2e_columns_not_compared <- setdiff(names(e2e), c("GroupId", sc$columns_compared))
    fwrite(as.data.table(res$out), file.path(d$dir, "reference_output.csv"), na = "NA")
  }

  checks_ok <- all(vapply(d$checks, function(r) isTRUE(r$pass), TRUE))
  not_identical <- names(Filter(function(r) !isTRUE(r$identical), d$checks))
  write_json(list(run_id = run_id, de_method = manifest$params$de_method, mode = manifest$mode,
                  self_check = sc, internal_checks_pass = checks_ok,
                  internal_checks = d$checks, scalars = d$scalars,
                  # What made these goldens: a stale reference shows as a hash mismatch.
                  provenance = list(md_limma_sha = Sys.getenv("MD_LIMMA_SHA", "unknown"),
                                    reference_sha256 = Sys.getenv("MD_REFERENCE_SHA256", "unknown"),
                                    corpus_harness_sha256 = manifest$provenance$harness_sha256,
                                    md_flexi_comparisons_sha = manifest$provenance$md_flexi_comparisons_sha,
                                    image_digest = Sys.getenv("MD_IMAGE_DIGEST", "unknown")),
                  versions = list(R = R.version.string,
                                  edgeR = as.character(packageVersion("edgeR")),
                                  DESeq2 = as.character(packageVersion("DESeq2")),
                                  limma = as.character(packageVersion("limma")),
                                  apeglm = as.character(packageVersion("apeglm")),
                                  ashr = as.character(packageVersion("ashr")),
                                  mixsqp = as.character(packageVersion("mixsqp")),
                                  irlba = as.character(packageVersion("irlba")))),
             file.path(d$dir, "reference.json"), auto_unbox = TRUE, digits = NA, pretty = TRUE,
             null = "null", na = "string")
  cat(sprintf("%-50s self_check=%s internal=%s max_rel=%s not_identical=[%s]%s\n", run_id,
              if (isTRUE(sc$passed)) "PASS" else "FAIL", if (checks_ok) "PASS" else "FAIL",
              format(if (is.null(sc$max_rel)) 0 else sc$max_rel, digits = 3),
              paste(not_identical, collapse = ","),
              if (!isTRUE(sc$passed) && !is.null(sc$got)) paste0(" got: ", substr(sc$got, 1, 160)) else ""))
  invisible(isTRUE(sc$passed) && checks_ok)
}

ids <- commandArgs(trailingOnly = TRUE)
if (length(ids) == 0) {
  ids <- list.dirs(file.path(CORPUS_DIR, "runs"), full.names = FALSE, recursive = FALSE)
  ids <- ids[grepl("^count_deseq2_.*shrink", ids)]
}
ok <- vapply(ids, function(id) tryCatch(run_one_reference(id), error = function(e) {
  cat(sprintf("%-50s CRASH %s\n", id, conditionMessage(e))); FALSE
}), TRUE)
cat(sprintf("\nreference: %d / %d runs pass self-check and internal checks\n", sum(ok), length(ok)))

# reference-shrink/index.json: every run directory present, its verdict and the sha256
# of each file, rebuilt over the whole directory on every invocation.
index <- lapply(sort(list.dirs(OUT_ROOT, full.names = FALSE, recursive = FALSE)), function(id) {
  files <- sort(list.files(file.path(OUT_ROOT, id)))
  j <- fromJSON(file.path(OUT_ROOT, id, "reference.json"), simplifyVector = FALSE)
  list(run_id = id, self_check_passed = isTRUE(j$self_check$passed),
       internal_checks_pass = isTRUE(j$internal_checks_pass), provenance = j$provenance,
       files = as.list(setNames(unname(tools::sha256sum(file.path(OUT_ROOT, id, files))), files)))
})
write_json(list(reference_sha256 = Sys.getenv("MD_REFERENCE_SHA256", "unknown"),
                md_limma_sha = Sys.getenv("MD_LIMMA_SHA", "unknown"), n_runs = length(index), runs = index),
           file.path(OUT_ROOT, "index.json"), auto_unbox = TRUE, pretty = TRUE, null = "null")
quit(status = if (all(ok)) 0 else 1)
