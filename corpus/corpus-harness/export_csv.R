# Export the golden corpus from .rds to CSV so a non-R consumer (the Rust/Python
# port) can read it. data.table::fwrite writes doubles with the shortest
# round-trip representation, so no precision is lost. Run:
#
#   cd /Users/giuseppeinfusini/wd/md-repos/MDFlexiComparisons
#   source .claude-r-env
#   "$RSCRIPT_EXEC" data-raw/golden-corpus/export_csv.R
#
# Writes <run>/results.csv, <run>/run_metadata.csv, shared/<ds>/*.csv next to the .rds.

suppressPackageStartupMessages(library(data.table))

corpus_dir <- Sys.getenv("MD_GOLDEN_CORPUS_DIR", "/Users/giuseppeinfusini/wd/md-limma-golden-corpus")

export_one <- function(rds_path) {
  csv_path <- sub("\\.rds$", ".csv", rds_path)
  obj <- readRDS(rds_path)
  obj <- as.data.frame(obj, check.names = FALSE, stringsAsFactors = FALSE)
  fwrite(obj, csv_path, na = "NA", quote = "auto", bom = FALSE)
  c(file = csv_path, rows = nrow(obj), cols = ncol(obj))
}

# Optional arguments: corpus subdirectories to export (e.g. runs/<id> shared/<ds>); default all.
subdirs <- commandArgs(trailingOnly = TRUE)
roots <- if (length(subdirs) > 0) file.path(corpus_dir, subdirs) else corpus_dir

written <- list()
for (rds in list.files(roots, pattern = "\\.rds$", recursive = TRUE, full.names = TRUE)) {
  res <- tryCatch(export_one(rds), error = function(e) c(file = rds, error = conditionMessage(e)))
  written[[length(written) + 1]] <- res
}

ok <- Filter(function(r) is.na(r["error"]), written)
bad <- Filter(function(r) !is.na(r["error"]), written)
cat(sprintf("exported %d files, %d failures\n", length(ok), length(bad)))
for (b in bad) cat("FAILED:", b["file"], "-", b["error"], "\n")
