# build_public_datasets.R
#
# Converts two public datasets into the harness's dataset-list shape
# (experiment_design, <entity>_intensity, <entity>_metadata) and writes
# public/<name>.rds, which load_dataset() picks up. Raw files are in public/:
#
#   phospho.cells.Ins.pe.rda  PhosR (github.com/PYangLab/PhosR, master, data/), GPL-3.
#     Humphrey et al. phosphoproteomics, FL83B and Hepa1.6 mouse cell lines,
#     Control vs Insulin, 6 replicates each; 5000 sites, log2 intensities, 76% NA.
#     Kept as published: 35 REV__ decoys and 2 CON__ contaminants (6 decoys pass
#     the filter in ptm_phosr_all). Parity data, not clean biology.
#     git blob 7faa9833f98518141fb462444787612609ba6ffd (find the PhosR commit
#     with `git log --find-object=<blob>` in a PhosR clone).
#   ST000001_AN000001.tsv     Metabolomics Workbench ST000001 (FatBIE), CC BY 4.0,
#     from https://www.metabolomicsworkbench.org/rest/study/analysis_id/AN000001/datatable/
#     Arabidopsis GC-MS, Ws vs fatb knock-down (source Class "fatb-ko KD;
#     At1g08510", level named "fatb") x Control vs Wounded, 6 replicates each;
#     102 named metabolites, raw peak heights, no missing values.
#
# Both raw files are checked against RAW_SHA256 before conversion.
#
# PhosR's PhosphoExperiment class is not installed in the image, so its slots
# are read as plain attributes. Run inside md-flexi-r45-local from the repo root:
#   Rscript data-raw/golden-corpus/build_public_datasets.R

suppressPackageStartupMessages(library(SummarizedExperiment))
out_dir <- "data-raw/golden-corpus/public"

RAW_SHA256 <- c(
  "phospho.cells.Ins.pe.rda" = "ab577077fc8f1f0fad33f0fdd6b925c2b0c337f5a0bf827b4d36f5dc7832dc09",
  "ST000001_AN000001.tsv" = "fe250e51fe7c04b27cc772e2069b1c3ab6932f0e019d3b471ff831ff160a3f13")
for (f in names(RAW_SHA256)) {
  got <- strsplit(system2("sha256sum", file.path(out_dir, f), stdout = TRUE), " ")[[1]][1]
  if (!identical(got, RAW_SHA256[[f]])) stop(sprintf("%s: sha256 %s, expected %s", f, got, RAW_SHA256[[f]]))
}

long_intensity <- function(mat, samples) {
  # mat: features x samples on the raw scale; NA stays NA with Imputed = 1,
  # the convention the harness's MNAR transforms use.
  data.frame(GroupId = rep(seq_len(nrow(mat)), times = ncol(mat)),
             replicate = rep(samples, each = nrow(mat)),
             NormalisedIntensity = as.vector(mat),
             Imputed = as.integer(is.na(as.vector(mat))),
             stringsAsFactors = FALSE)
}

# ---- PTM: PhosR insulin phosphosites --------------------------------------
e <- new.env()
load(file.path(out_dir, "phospho.cells.Ins.pe.rda"), envir = e)
at <- attributes(e$phospho.cells.Ins.pe)
log2_mat <- at$assays@data@listData$Quantification
samples <- sub("^Intensity\\.", "", colnames(log2_mat))
parts <- do.call(rbind, strsplit(samples, "_"))
ed <- data.frame(sample_name = samples, condition = paste(parts[, 1], parts[, 2], sep = "_"),
                 treatment = parts[, 2], cellline = parts[, 1], stringsAsFactors = FALSE)
ptm <- list(
  experiment_design = ed,
  ptm_intensity = long_intensity(2^log2_mat, samples),
  ptm_metadata = data.frame(GroupId = seq_len(nrow(log2_mat)), PTMProtein = at$UniprotID,
                            GeneNames = at$GeneSymbol,
                            GroupLabel = paste(at$GeneSymbol, at$Site, sep = "_"),
                            stringsAsFactors = FALSE))
saveRDS(ptm, file.path(out_dir, "phosr_insulin.rds"))
cat(sprintf("phosr_insulin: %d sites x %d samples, %.1f%% NA, conditions %s\n", nrow(log2_mat),
            ncol(log2_mat), 100 * mean(is.na(log2_mat)), paste(unique(ed$condition), collapse = ",")))

# ---- Metabolite: Metabolomics Workbench ST000001 --------------------------
tab <- read.delim(file.path(out_dir, "ST000001_AN000001.tsv"), check.names = FALSE,
                  stringsAsFactors = FALSE)
genotype <- ifelse(grepl("fatb-ko", tab$Class), "fatb", "Ws")
treatment <- ifelse(grepl("Non-Wounded", tab$Class), "Control", "Wounded")
ed <- data.frame(sample_name = tab$Samples, condition = paste(genotype, treatment, sep = "_"),
                 treatment = treatment, genotype = genotype, stringsAsFactors = FALSE)
mets <- colnames(tab)[-(1:2)]
mat <- t(as.matrix(tab[, mets]))
met <- list(
  experiment_design = ed,
  metabolite_intensity = long_intensity(mat, tab$Samples),
  metabolite_metadata = data.frame(GroupId = seq_along(mets), MetaboliteId = mets,
                                   GroupLabel = mets, stringsAsFactors = FALSE))
saveRDS(met, file.path(out_dir, "st000001_fatbie.rds"))
cat(sprintf("st000001_fatbie: %d metabolites x %d samples, %d NA, conditions %s\n", nrow(mat),
            ncol(mat), sum(is.na(mat)), paste(table(ed$condition), names(table(ed$condition)), collapse = ",")))
