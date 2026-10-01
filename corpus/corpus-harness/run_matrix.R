# run_matrix.R
#
# Declarative table of every golden-corpus run for PR-A1 (limma -> Rust
# rewrite). Consumed by generate_golden_corpus.R. Nothing in this file talks
# to R packages beyond base -- it just builds a list of plain lists.
#
# Scope: two run lists live here.
#   `runs`       -- the limma corpus (projects/2026-09-25-limma-rust-rewrite/plan.md):
#                   limma + protein entity, plus the gene-entity runs that exercise the
#                   Python shim's override (process_r.py:733-740).
#   `count_runs` -- the edgeR / DESeq2 gene-entity corpus (phase 2,
#                   projects/2026-10-01-edger-deseq2-rust-port/plan.md, PR-E0/E1).
#                   Selected with MD_CORPUS_KIND=count; written to its own corpus dir.
#                   Section 4 at the bottom of this file.
#
# Trim vs. the plan's nominal ~98 runs (48 core + ~35 one-at-a-time + ~15
# edge): this file has 48 core + 20 one-at-a-time + 9 edge = 77 runs. Every
# category the plan names is represented at least once; what's cut is
# additional repetition within a category (e.g. sweeping filter_logic across
# every bundled dataset instead of just the two core ones). Documented as a
# deliberate trim in pr-a1-harness-status.md.
#
# ---- Run schema -------------------------------------------------------
# id                    unique character id; also the output subdirectory name
# description           one-line human description
# dataset               "bojkova2020" / "demichev2021" / "monkeyPox" /
#                        "panCancer" / "gene_synth"
# entity_type           "protein" / "peptide" / "gene"
# mode                  "discovery" (runDiscovery) or "anova" (runANOVA)
# comparison_type       "all" or "custom" (runDiscovery's comparisonType)
# custom_comparisons    NULL, or list(left = character(), right = character())
# control_cols          NULL, or list(Column = character(), Type = character())
# fit_separate_models   logical
# filter_logic          "one_condition" / "all_conditions" / "full"
#                        (already the internal R-facing key, i.e. AFTER the
#                        process.R label map -- see generate_golden_corpus.R
#                        header comment on why runs are "post-shim")
# filter_criteria       "percentage" / "count"
# filter_threshold      numeric
# limma_trend           logical
# limma_robust          logical
# return_decide_test_column logical
# transform             NULL or name of a transform function applied to the
#                        raw dataset before the run (see
#                        generate_golden_corpus.R::apply_transform())
# gene_shim             logical; if TRUE, apply the Python process_r.py
#                        gene-entity override on top of whatever this row
#                        specifies (mirrors process_r.py:733-740)
# expected_error        NULL or a substring expected in the raised error
# boundary_sensitive    logical; decide_* / near-threshold values may differ
#                        across limma point releases or BLAS -- see tolerance
#                        policy in generate_golden_corpus.R
# notes                 free text
# de_method             "limma" / "edgeR" / "DESeq2" (count_runs only use the last two)
# edger_norm_method     calcNormFactors method (edgeR only; form default "TMM")
# deseq2_lfc_shrinkage  "none" / "apeglm" / "ashr" / "normal" (DESeq2 pairwise only)
# deseq2_alpha          results(alpha=) for independent filtering (DESeq2 only)
# apeglm_seed           withr::with_seed around lfcShrink(type="apeglm")
#
# ------------------------------------------------------------------------

mk_run <- function(id, description, dataset, entity_type, mode = "discovery",
                    comparison_type = "all", custom_comparisons = NULL,
                    control_cols = NULL, fit_separate_models = TRUE,
                    filter_logic = "one_condition", filter_criteria = "percentage",
                    filter_threshold = 0.5, limma_trend = TRUE, limma_robust = TRUE,
                    return_decide_test_column = TRUE, transform = NULL,
                    gene_shim = FALSE, expected_error = NULL,
                    boundary_sensitive = FALSE, notes = "", condition_col = "condition",
                    de_method = "limma", edger_norm_method = "TMM",
                    deseq2_lfc_shrinkage = "none", deseq2_alpha = 0.05, apeglm_seed = 1L) {
  list(id = id, description = description, dataset = dataset,
       de_method = de_method, edger_norm_method = edger_norm_method,
       deseq2_lfc_shrinkage = deseq2_lfc_shrinkage, deseq2_alpha = deseq2_alpha,
       apeglm_seed = apeglm_seed,
       entity_type = entity_type, mode = mode, comparison_type = comparison_type,
       custom_comparisons = custom_comparisons, control_cols = control_cols,
       fit_separate_models = fit_separate_models, filter_logic = filter_logic,
       filter_criteria = filter_criteria, filter_threshold = filter_threshold,
       limma_trend = limma_trend, limma_robust = limma_robust,
       return_decide_test_column = return_decide_test_column, transform = transform,
       gene_shim = gene_shim, expected_error = expected_error,
       boundary_sensitive = boundary_sensitive, notes = notes, condition_col = condition_col)
}

runs <- list()

# ---- 1. Fully-crossed core (48 runs) -----------------------------------
# bojkova2020 + demichev2021, protein entity, comparisonType="all",
# filterValidValuesCriteria="percentage" @ default threshold 0.5,
# crossed over fit_separate_models x limma_robust x limma_trend x filter_logic.
core_grid <- expand.grid(
  dataset = c("bojkova2020", "demichev2021"),
  fit_separate_models = c(TRUE, FALSE),
  limma_robust = c(TRUE, FALSE),
  limma_trend = c(TRUE, FALSE),
  filter_logic = c("one_condition", "all_conditions", "full"),
  stringsAsFactors = FALSE
)
for (i in seq_len(nrow(core_grid))) {
  r <- core_grid[i, ]
  id <- sprintf("core_%s_sep%s_rob%s_trend%s_%s",
                r$dataset, r$fit_separate_models, r$limma_robust, r$limma_trend, r$filter_logic)
  runs[[id]] <- mk_run(
    id = id,
    description = sprintf("Core cross: %s, sepModels=%s robust=%s trend=%s logic=%s",
                           r$dataset, r$fit_separate_models, r$limma_robust, r$limma_trend, r$filter_logic),
    dataset = r$dataset, entity_type = "protein", comparison_type = "all",
    fit_separate_models = r$fit_separate_models, filter_logic = r$filter_logic,
    filter_criteria = "percentage", filter_threshold = 0.5,
    limma_trend = r$limma_trend, limma_robust = r$limma_robust,
    notes = "Part of the 2x2x2x2x3 fully-crossed core matrix."
  )
}

# ---- 2. One-at-a-time from defaults (20 runs) --------------------------
# Baseline defaults: bojkova2020, protein, comparisonType="all",
# fitSeparateModels=TRUE, filter_logic=one_condition, filter_criteria=percentage,
# threshold=0.5, trend=TRUE, robust=TRUE. Each row below varies knobs that are
# NOT already swept by the core-48 cross (control cols, filter criteria,
# comparison_type, mode, entity_type, dataset variety, comparison count).

runs[["oat_filter_count_default"]] <- mk_run(
  "oat_filter_count_default", "Count-based filter, default threshold=3",
  "bojkova2020", "protein", filter_criteria = "count", filter_threshold = 3)

runs[["oat_filter_count_eq_nreps"]] <- mk_run(
  "oat_filter_count_eq_nreps", "Count filter threshold == max replicates per condition (6)",
  "bojkova2020", "protein", filter_criteria = "count", filter_threshold = 6,
  boundary_sensitive = TRUE)

runs[["oat_filter_pct_zero"]] <- mk_run(
  "oat_filter_pct_zero", "Percentage filter threshold = 0.0 (accept anything)",
  "bojkova2020", "protein", filter_criteria = "percentage", filter_threshold = 0.0)

runs[["oat_filter_pct_one"]] <- mk_run(
  "oat_filter_pct_one", "Percentage filter threshold = 1.0 (require complete data)",
  "bojkova2020", "protein", filter_criteria = "percentage", filter_threshold = 1.0,
  boundary_sensitive = TRUE)

runs[["oat_control_categorical"]] <- mk_run(
  "oat_control_categorical", "Categorical covariate (group: Control/Virus)",
  "bojkova2020", "protein",
  control_cols = list(Column = "group", Type = "categorical"))

runs[["oat_control_numerical"]] <- mk_run(
  "oat_control_numerical", "Numerical covariate (monkeyPox's native control_var)",
  "monkeyPox", "peptide",
  control_cols = list(Column = "control_var", Type = "numerical"))

runs[["oat_control_both"]] <- mk_run(
  "oat_control_both", "Categorical + numerical covariates together",
  "bojkova2020", "protein", transform = "add_numeric_covariate",
  control_cols = list(Column = c("group", "numeric_batch"), Type = c("categorical", "numerical")))

runs[["oat_peptide_default"]] <- mk_run(
  "oat_peptide_default", "Peptide entity, defaults, no covariate",
  "monkeyPox", "peptide")

runs[["oat_gene_shim"]] <- mk_run(
  "oat_gene_shim", "Gene entity with the Python process_r.py override applied",
  "gene_synth", "gene", gene_shim = TRUE,
  notes = "Deferred scope EXCEPT this one run -- see process_r.py:733-740.")

runs[["oat_anova_all"]] <- mk_run(
  "oat_anova_all", "ANOVA (omnibus), comparisonType=all, no covariate",
  "bojkova2020", "protein", mode = "anova")

runs[["oat_anova_custom"]] <- mk_run(
  "oat_anova_custom", "ANOVA (omnibus), comparisonType=custom (2 pairs)",
  "bojkova2020", "protein", mode = "anova", comparison_type = "custom",
  custom_comparisons = list(left = c("2h", "6h"), right = c("6h", "24h")))

runs[["oat_anova_covariate"]] <- mk_run(
  "oat_anova_covariate", "ANOVA with categorical covariate (group)",
  "bojkova2020", "protein", mode = "anova",
  control_cols = list(Column = "group", Type = "categorical"))

runs[["oat_single_comparison"]] <- mk_run(
  "oat_single_comparison", "Discovery, exactly one custom comparison",
  "bojkova2020", "protein", comparison_type = "custom",
  custom_comparisons = list(left = "2h", right = "6h"))

runs[["oat_five_comparisons"]] <- mk_run(
  "oat_five_comparisons", "Discovery, 5 custom comparisons (demichev2021, 5 severities)",
  "demichev2021", "protein", comparison_type = "custom",
  custom_comparisons = list(
    left  = c("severity-3", "severity-3", "severity-4", "severity-5", "severity-6"),
    right = c("severity-5", "severity-7", "severity-6", "severity-7", "severity-7")))

runs[["oat_pancancer_smoke"]] <- mk_run(
  "oat_pancancer_smoke", "panCancer smoke run, protein, defaults, 2 conditions",
  "panCancer", "protein")

runs[["oat_monkeypox_all"]] <- mk_run(
  "oat_monkeypox_all", "monkeyPox, comparisonType=all (peptide coverage completeness)",
  "monkeyPox", "peptide", comparison_type = "all")

runs[["oat_blocking_subject_sep"]] <- mk_run(
  "oat_blocking_subject_sep", "Blocking fixture (subject covariate), fitSeparateModels=TRUE",
  "bojkova2020", "protein", transform = "blocking_split",
  control_cols = list(Column = "subject", Type = "categorical"),
  fit_separate_models = TRUE,
  notes = "Mirrors data-raw/test-data-blocking.R's sample_name -> group/condition/subject split.")

runs[["oat_blocking_subject_onemodel"]] <- mk_run(
  "oat_blocking_subject_onemodel", "Blocking fixture (subject covariate), fitSeparateModels=FALSE",
  "bojkova2020", "protein", transform = "blocking_split",
  control_cols = list(Column = "subject", Type = "categorical"),
  fit_separate_models = FALSE,
  notes = "Mirrors data-raw/test-data-blocking.R's one-model (fitSeparateModels=FALSE) fixture.")

runs[["oat_decide_test_column_off"]] <- mk_run(
  "oat_decide_test_column_off", "returnDecideTestColumn=FALSE (contrast against the corpus-wide default TRUE)",
  "bojkova2020", "protein", return_decide_test_column = FALSE)

runs[["oat_demichev_low_rep_condition"]] <- mk_run(
  "oat_demichev_low_rep_condition", "Custom comparison stressing severity-6 (n=6, the smallest demichev group)",
  "demichev2021", "protein", comparison_type = "custom",
  custom_comparisons = list(left = "severity-6", right = "severity-7"),
  boundary_sensitive = TRUE)

# ---- 3. Edge cases (9 runs) ---------------------------------------------

runs[["edge_two_reps_per_condition"]] <- mk_run(
  "edge_two_reps_per_condition", "Only 2 replicates per condition (2h, 6h subset)",
  "bojkova2020", "protein", transform = "subset_2reps", comparison_type = "custom",
  custom_comparisons = list(left = "2h", right = "6h"), boundary_sensitive = TRUE)

runs[["edge_single_rep_condition_ebayes_fail"]] <- mk_run(
  "edge_single_rep_condition_ebayes_fail", "One condition reduced to 1 replicate under fitSeparateModels=TRUE",
  "bojkova2020", "protein", transform = "single_rep_condition", comparison_type = "custom",
  custom_comparisons = list(left = "2h", right = "6h"), fit_separate_models = TRUE,
  notes = "R fits 1 vs 2 replicates (df 1) and returns results; the genuine eBayes failure is edge_single_rep_pair_ebayes_fail.")

runs[["edge_count_threshold_exceeds_replicates"]] <- mk_run(
  "edge_count_threshold_exceeds_replicates", "Count filter threshold (100) exceeds any condition's replicate count",
  "bojkova2020", "protein", filter_criteria = "count", filter_threshold = 100,
  expected_error = "Not enough features")

runs[["edge_collinear_covariate"]] <- mk_run(
  "edge_collinear_covariate", "Covariate perfectly collinear with condition -> rank-deficient design",
  "bojkova2020", "protein", transform = "collinear_covariate",
  control_cols = list(Column = "condition_dup", Type = "categorical"),
  expected_error = "perfectly collinear")

runs[["edge_high_missingness_boundary"]] <- mk_run(
  "edge_high_missingness_boundary", "~85% Imputed flags forced in one condition, filter threshold=0.5",
  "bojkova2020", "protein", transform = "high_missingness",
  filter_criteria = "percentage", filter_threshold = 0.5, boundary_sensitive = TRUE)

runs[["edge_unicode_condition_names"]] <- mk_run(
  "edge_unicode_condition_names", "Condition labels with unicode / special characters",
  "bojkova2020", "protein", transform = "unicode_conditions")

runs[["edge_skip_branch_partial"]] <- mk_run(
  "edge_skip_branch_partial", "Separate-models run where one contrast has <=3 quantifiable features, another has enough",
  "bojkova2020", "protein", transform = "skip_branch_subset", comparison_type = "custom",
  custom_comparisons = list(left = c("2h", "6h"), right = c("6h", "24h")),
  fit_separate_models = TRUE, filter_criteria = "count", filter_threshold = 3,
  boundary_sensitive = TRUE,
  notes = "Exercises limmaFitSeparateModels' per-contrast >3-feature skip (R/limmaStatsFun.R).")

runs[["edge_negative_intensities"]] <- mk_run(
  "edge_negative_intensities", "One forced-negative intensity value",
  "bojkova2020", "protein", transform = "negative_intensity",
  expected_error = "negative intensities")

runs[["edge_single_condition_level"]] <- mk_run(
  "edge_single_condition_level", "Only one condition level present, comparisonType=all",
  "bojkova2020", "protein", transform = "single_condition_level", comparison_type = "all",
  expected_error = "at least two distinct levels")

# ---- 4. Missing values at production shape (fix-plan Step 5) -------------
# Intensity=NA with Imputed=1, which takes limma's unequal-df path
# (fitFDistUnequalDF1). Shaped like production discovery (process.R:38-58):
# custom comparisons, no decide columns, separate models, one_condition @ 0.5.
na_sets <- list(
  panCancer = list(transform = "restore_missing",
                   cmp = list(left = c("Early", "Late"), right = c("Healthy", "Healthy"))),
  bojkova2020 = list(transform = "mnar_low_abundance",
                     cmp = list(left = c("2h", "6h"), right = c("6h", "24h"))),
  demichev2021 = list(transform = "restore_missing",
                      cmp = list(left = c("severity-3", "severity-5"), right = c("severity-5", "severity-7")))
)
na_grid <- expand.grid(dataset = names(na_sets), limma_trend = c(TRUE, FALSE),
                       limma_robust = c(TRUE, FALSE), stringsAsFactors = FALSE)
for (i in seq_len(nrow(na_grid))) {
  r <- na_grid[i, ]
  if (r$dataset == "demichev2021" && !(r$limma_trend && r$limma_robust)) next
  s <- na_sets[[r$dataset]]
  id <- sprintf("na_%s_trend%s_rob%s", r$dataset, r$limma_trend, r$limma_robust)
  runs[[id]] <- mk_run(
    id, sprintf("NA intensities, production-shaped custom call: %s trend=%s robust=%s",
                r$dataset, r$limma_trend, r$limma_robust),
    r$dataset, "protein", comparison_type = "custom", custom_comparisons = s$cmp,
    limma_trend = r$limma_trend, limma_robust = r$limma_robust,
    return_decide_test_column = FALSE, transform = s$transform,
    notes = "Unequal-df eBayes path at full size (review 2026-09-28 C1 / B1).")
}

runs[["na_panCancer_anova"]] <- mk_run(
  "na_panCancer_anova", "ANOVA with NA intensities (panCancer, restore_missing)",
  "panCancer", "protein", mode = "anova", transform = "restore_missing",
  return_decide_test_column = FALSE,
  notes = paste("Moderated F on the unequal-df path. panCancer has no outlying variances, so",
                "robust=TRUE leaves df.prior constant (4.133, the omnibus fit of",
                "na_panCancer_trendTRUE_robTRUE); na_bojkova2020_anova is the robust ANOVA arm."))

# All-NA gene under the gene shim (review B3). The shim forces trend=TRUE, so
# only robust varies. R (limma 3.66 + MD wrapper) fails the whole run.
all_na_err <- c("TRUE" = "covariate contains NA or infinite values",
                "FALSE" = "NA covariate values not allowed")
for (rob in c(TRUE, FALSE)) {
  id <- sprintf("edge_gene_all_na_rob%s", rob)
  runs[[id]] <- mk_run(
    id, sprintf("Gene shim, one gene all Intensity=NA/Imputed=1, robust=%s", rob),
    "gene_synth", "gene", gene_shim = TRUE, limma_robust = rob,
    transform = "all_na_gene", expected_error = all_na_err[[as.character(rob)]],
    notes = "limma 3.66 stops on an NA trend covariate; the port must fail the same way.")
}

runs[["edge_near_collinear_numeric"]] <- mk_run(
  "edge_near_collinear_numeric", "Numeric covariate collinear with condition to within 1e-9",
  "bojkova2020", "protein", transform = "near_collinear_numeric",
  control_cols = list(Column = "near_dup", Type = "numerical"),
  expected_error = "perfectly collinear",
  notes = "qr() rank tolerance 1e-7 (review B4, fix-plan Step 3).")

# ---- Fix-plan Step 4: untested contracts ----------------------------------
# Expected errors below were set from what R returned when first generated.
for (n in c(3, 4)) {
  for (md in c("discovery", "anova")) {
    id <- sprintf("edge_n%d_features_%s", n, md)
    runs[[id]] <- mk_run(
      id, sprintf("Exactly %d quantifiable features, %s", n, md),
      "bojkova2020", "protein", mode = md, transform = sprintf("first%d_features", n),
      expected_error = if (n == 3) "Not enough features available after filtration" else NULL,
      notes = "Pins the > 3 features gate (limmaStatsFun.R).")
  }
}

runs[["edge_single_rep_pair_ebayes_fail"]] <- mk_run(
  "edge_single_rep_pair_ebayes_fail", "Separate 2h-vs-6h model has no residual df; 6h-vs-24h and the omnibus fit do",
  "bojkova2020", "protein", transform = "single_rep_pair", comparison_type = "custom",
  custom_comparisons = list(left = c("2h", "6h"), right = c("6h", "24h")), fit_separate_models = TRUE,
  expected_error = "Empirical Bayes estimation failed")

runs[["edge_df0_omnibus"]] <- mk_run(
  "edge_df0_omnibus", "Only the df-0 pair is compared, so the unguarded omnibus eBayes fails",
  "bojkova2020", "protein", transform = "single_rep_pair", comparison_type = "custom",
  custom_comparisons = list(left = "2h", right = "6h"), fit_separate_models = TRUE,
  expected_error = "No residual degrees of freedom in linear model fits")

runs[["edge_condition_col_time_point"]] <- mk_run(
  "edge_condition_col_time_point", "Condition column named 'Time point' (non-syntactic)",
  "bojkova2020", "protein", transform = "condition_time_point", condition_col = "Time point",
  comparison_type = "custom", custom_comparisons = list(left = c("2h", "6h"), right = c("6h", "24h")))

runs[["edge_notallok_equal_df"]] <- mk_run(
  "edge_notallok_equal_df", "Complete matrix plus one all-NA feature, robust, no trend",
  "bojkova2020", "protein", transform = "all_na_feature", filter_threshold = 0,
  limma_trend = FALSE, limma_robust = TRUE, fit_separate_models = FALSE,
  notes = "Equal positive df plus a df-0 row: fitFDistRobustly notallok recursion (r-rust.md Moderate 2).")

# ---- Round-3 review runs (2026-09-29) ---------------------------------------
# Expected errors below were set from what R returned when first generated.
prod_cmp <- list(left = c("2h", "6h"), right = c("6h", "24h"))

for (tr in c(TRUE, FALSE)) {
  id <- sprintf("edge_two_informative_trend%s_rob%s", tr, tr)
  runs[[id]] <- mk_run(
    id, sprintf("2h-vs-6h model with exactly two informative variances, unequal df, trend=robust=%s", tr),
    "bojkova2020", "protein", transform = "two_informative_pair", comparison_type = "custom",
    custom_comparisons = prod_cmp, filter_threshold = 0, limma_trend = tr, limma_robust = tr,
    return_decide_test_column = FALSE,
    notes = "n.informative == 2 branch of fitFDistUnequalDF1 (round-3 must-fix 1).")
}

runs[["na_cov_bojkova_subject"]] <- mk_run(
  "na_cov_bojkova_subject", "NA intensities plus a categorical subject covariate, separate models",
  "bojkova2020", "protein", transform = "mnar_blocking", comparison_type = "custom",
  custom_comparisons = prod_cmp, control_cols = list(Column = "subject", Type = "categorical"),
  return_decide_test_column = FALSE,
  notes = "contrasts.fit non-orthogonal approximation with NA (round-3 must-fix 2).")

runs[["na_cov_panCancer_numeric"]] <- mk_run(
  "na_cov_panCancer_numeric", "NA intensities plus a numeric covariate, separate models",
  "panCancer", "protein", transform = "restore_missing_numeric_cov", comparison_type = "custom",
  custom_comparisons = na_sets$panCancer$cmp, control_cols = list(Column = "numeric_batch", Type = "numerical"),
  return_decide_test_column = FALSE,
  notes = "contrasts.fit non-orthogonal approximation with NA (round-3 must-fix 2).")

runs[["na_bojkova2020_anova"]] <- mk_run(
  "na_bojkova2020_anova", "ANOVA with MNAR NA intensities (bojkova2020)",
  "bojkova2020", "protein", mode = "anova", transform = "mnar_low_abundance",
  return_decide_test_column = FALSE,
  notes = "Moderated F on the unequal-df path where robust changes df.prior (round-3 must-fix 4).")

for (rob in c(TRUE, FALSE)) {
  id <- sprintf("na_bojkova2020_softmnar_rob%s", rob)
  runs[[id]] <- mk_run(
    id, sprintf("Seeded MNAR+MAR missingness at a higher rate, robust=%s", rob),
    "bojkova2020", "protein", transform = "soft_mnar", comparison_type = "custom",
    custom_comparisons = prod_cmp, limma_robust = rob, return_decide_test_column = FALSE,
    notes = "Robust arm with outlying variances (round-3 must-fix 4).")
}

for (n in c(3, 4)) {
  id <- sprintf("edge_pair_gate_keep%d", n)
  runs[[id]] <- mk_run(
    id, sprintf("2h-vs-6h pair keeps exactly %d quantifiable features", n),
    "bojkova2020", "protein", transform = sprintf("skip_branch_keep%d", n), comparison_type = "custom",
    custom_comparisons = prod_cmp, filter_criteria = "count", filter_threshold = 3,
    notes = "Pins the per-pair > 3 gate (limmaStatsFun.R:289).")
}

for (tr in c(TRUE, FALSE)) {
  id <- sprintf("na_all_na_feature_trend%s", tr)
  runs[[id]] <- mk_run(
    id, sprintf("MNAR NA plus one all-NA feature, filter 0, trend=%s", tr),
    "bojkova2020", "protein", transform = "mnar_all_na", comparison_type = "custom",
    custom_comparisons = prod_cmp, filter_threshold = 0, limma_trend = tr,
    return_decide_test_column = FALSE,
    expected_error = if (tr) "prior.weights contain NA values" else NULL,
    notes = "Unequal-df path with a zero-observation feature.")
}

runs[["edge_reserved_word_conditions"]] <- mk_run(
  "edge_reserved_word_conditions", "Condition labels TRUE / non-ASCII / function",
  "bojkova2020", "protein", transform = "reserved_word_conditions")

runs[["gene_no_shim_discovery"]] <- mk_run(
  "gene_no_shim_discovery", "Gene entity without the process_r.py override (MDEnrichment path)",
  "gene_synth", "gene")
runs[["gene_no_shim_anova"]] <- mk_run(
  "gene_no_shim_anova", "Gene entity ANOVA without the override", "gene_synth", "gene", mode = "anova")
runs[["gene_libsize_warning"]] <- mk_run(
  "gene_libsize_warning", "Gene entity with one library 5x larger (limma-trend warning)",
  "gene_synth", "gene", transform = "gene_libsize")

runs[["entity_ptm"]] <- mk_run(
  "entity_ptm", "bojkova2020 served as PTM entity", "bojkova2020", "ptm", transform = "as_ptm",
  comparison_type = "custom", custom_comparisons = prod_cmp, return_decide_test_column = FALSE)
runs[["entity_metabolite"]] <- mk_run(
  "entity_metabolite", "bojkova2020 served as metabolite entity", "bojkova2020", "metabolite",
  transform = "as_metabolite", comparison_type = "custom", custom_comparisons = prod_cmp,
  return_decide_test_column = FALSE)

runs[["edge_one_level_covariate_in_pair"]] <- mk_run(
  "edge_one_level_covariate_in_pair", "Covariate constant inside the 2h-vs-6h pair",
  "bojkova2020", "protein", transform = "one_level_in_pair", comparison_type = "custom",
  custom_comparisons = prod_cmp, control_cols = list(Column = "batch", Type = "categorical"),
  expected_error = "column has only 1 level after filtering")
runs[["edge_empty_string_covariate"]] <- mk_run(
  "edge_empty_string_covariate", "Covariate with empty-string values",
  "bojkova2020", "protein", transform = "empty_string_covariate", comparison_type = "custom",
  custom_comparisons = prod_cmp, control_cols = list(Column = "subject", Type = "categorical"))
runs[["edge_raw_intensity_only"]] <- mk_run(
  "edge_raw_intensity_only", "Intensity column only, no NormalisedIntensity",
  "bojkova2020", "protein", transform = "raw_intensity_only")
runs[["edge_under_3_samples"]] <- mk_run(
  "edge_under_3_samples", "One sample per condition across the only contrast",
  "bojkova2020", "protein", transform = "two_samples", comparison_type = "custom",
  custom_comparisons = list(left = "2h", right = "6h"),
  expected_error = "Not enough replicates across all contrasts")
runs[["edge_all_pairs_skipped"]] <- mk_run(
  "edge_all_pairs_skipped", "Each pair keeps 2 quantifiable features, the omnibus keeps 4",
  "bojkova2020", "protein", transform = "all_pairs_starved", comparison_type = "custom",
  custom_comparisons = prod_cmp, filter_criteria = "count", filter_threshold = 3,
  expected_error = "All pairwise comparisons were skipped")

# ---- Public PTM and metabolite datasets --------------------------------
# Real (non-relabelled) data for the ptm and metabolite entity branches; see
# build_public_datasets.R. Both are 2 x 2 designs with 6 replicates per cell:
# phosr_insulin (5000 phosphosites, 76% NA) and st000001_fatbie (102
# metabolites, fully observed).
public_sets <- list(
  list(key = "ptm_phosr", dataset = "phosr_insulin", entity = "ptm", factor2 = "cellline"),
  list(key = "met_st000001", dataset = "st000001_fatbie", entity = "metabolite", factor2 = "genotype"))
for (ps in public_sets) {
  id <- paste0(ps$key, "_all")
  runs[[id]] <- mk_run(id, sprintf("%s, all pairwise over the 4 groups", ps$dataset), ps$dataset, ps$entity)
  id <- paste0(ps$key, "_anova")
  runs[[id]] <- mk_run(id, sprintf("%s, ANOVA over the 4 groups", ps$dataset), ps$dataset, ps$entity,
                       mode = "anova", return_decide_test_column = FALSE)
  id <- paste0(ps$key, "_treatment_cov")
  runs[[id]] <- mk_run(id, sprintf("%s, treatment adjusted for %s", ps$dataset, ps$factor2),
                       ps$dataset, ps$entity, condition_col = "treatment",
                       control_cols = list(Column = ps$factor2, Type = "categorical"))
  id <- paste0(ps$key, "_joint_trendFALSE_robFALSE")
  runs[[id]] <- mk_run(id, sprintf("%s, one joint fit, trend and robust off", ps$dataset),
                       ps$dataset, ps$entity, fit_separate_models = FALSE, limma_trend = FALSE,
                       limma_robust = FALSE)
}

# ---- Round-4 review runs (2026-09-29) ------------------------------------
for (id in grep("^ptm_phosr_", names(runs), value = TRUE))
  runs[[id]]$notes <- "Source keeps 35 REV__ decoys and 2 CON__ contaminants; parity data, not clean biology."
# C3: small metabolite panels, where the robust trended prior swings (df.prior 4e-5 to Inf).
for (n in c(10, 30)) {
  id <- sprintf("met_st000001_first%d", n)
  runs[[id]] <- mk_run(id, sprintf("st000001_fatbie, first %d metabolites, separate models", n),
                       "st000001_fatbie", "metabolite", transform = sprintf("first%d_features", n))
}
# Per-contrast t from one joint fit with trend and robust on, on real data with NA.
for (ps in public_sets) {
  id <- paste0(ps$key, "_joint_trendTRUE_robTRUE")
  runs[[id]] <- mk_run(id, sprintf("%s, one joint fit, trend and robust on", ps$dataset),
                       ps$dataset, ps$entity, fit_separate_models = FALSE)
}
phosr_cmp <- list(left = c("FL83B_Ins", "Hepa1.6_Ins"), right = c("FL83B_Control", "Hepa1.6_Control"))
runs[["ptm_phosr_production"]] <- mk_run(
  "ptm_phosr_production", "phosr_insulin, production call: custom pairs, decide off",
  "phosr_insulin", "ptm", comparison_type = "custom", custom_comparisons = phosr_cmp,
  return_decide_test_column = FALSE)
runs[["ptm_phosr_sparse_rows"]] <- mk_run(
  "ptm_phosr_sparse_rows", "phosr_insulin with missing values as absent rows (sparse long table)",
  "phosr_insulin", "ptm", transform = "drop_missing_rows", comparison_type = "custom",
  custom_comparisons = phosr_cmp, return_decide_test_column = FALSE)
runs[["ptm_phosr_na_imputed0"]] <- mk_run(
  "ptm_phosr_na_imputed0", "phosr_insulin with NA intensities flagged Imputed = 0",
  "phosr_insulin", "ptm", transform = "na_not_imputed", comparison_type = "custom",
  custom_comparisons = phosr_cmp, return_decide_test_column = FALSE,
  expected_error = "prior.weights contain NA values",
  notes = "NA with Imputed=0 counts as valid in the filter, so an all-NA feature reaches eBayes with an NA Amean; the trend fit carries it into the prior weights and limma stops.")
# Round-5 review: the same data with trend off is a success path (NA counted valid, all-NA rows
# leave fitFDist, BH n = the non-NA p-values).
runs[["ptm_phosr_na_imputed0_trendFALSE"]] <- mk_run(
  "ptm_phosr_na_imputed0_trendFALSE", "phosr_insulin with NA intensities flagged Imputed = 0, trend off",
  "phosr_insulin", "ptm", transform = "na_not_imputed", comparison_type = "custom",
  custom_comparisons = phosr_cmp, return_decide_test_column = FALSE, limma_trend = FALSE)
runs[["synth_scale_50k_100"]] <- mk_run(
  "synth_scale_50k_100", "Synthetic 50,000 proteins x 100 samples, 13% NA, all pairs",
  "synth_scale", "protein")
for (sep in c(TRUE, FALSE)) {
  id <- sprintf("synth_blocking_18_levels_age_sep%s", sep)
  runs[[id]] <- mk_run(id, "Unbalanced 24/18/12, 18-level batch plus numeric age, 13% NA",
                       "synth_blocking", "protein", fit_separate_models = sep,
                       control_cols = list(Column = c("batch", "age"), Type = c("categorical", "numerical")))
}

# Sanity: ids must be unique (names(runs) used directly as directory names).
stopifnot(!any(duplicated(names(runs))))

# ---- 4. edgeR / DESeq2 gene-entity corpus (count_runs) --------------------
# Phase 2 of the limma rewrite, PR-E0/E1. Every run is gene entity with the
# process_r.py gene shim on (production always applies it; the limma-only
# filter keys it sets are inert for the count engines, whose low-count filter
# is edgeR::filterByExpr inside the engine). Fixtures (generate_golden_corpus.R):
#   airway      -- Bioconductor airway, all 8 samples, first 8000 Ensembl ids in
#                  sorted order (about half are all-zero, so filterByExpr bites);
#                  condition = dex (trt / untrt), cell (4 levels), avgLength (numeric).
#   count_synth -- negative-binomial counts, 2000 genes, 3 conditions x 4 reps
#                  (ctrl / doseA / doseB), batch (2 levels, crossed), rin (numeric);
#                  gene 1 all zero, 15% DE, dispersion trend 0.04 + 2/mu.
#   count_synth_cooks -- same generator, 3 x 7 reps (minReplicatesForReplace = 7),
#                  20 genes with one sample inflated 50x so Cook's replacement fires.
# Not covered, on purpose: isLogScale = TRUE. runDiscovery/runANOVA never pass it
# (edgeRStatsFun.R:330 default FALSE, deseq2 path has no argument), so no
# production request reaches that branch.
count_runs <- list()
# known_production_behaviour: production does something we would not choose, and the port
# reproduces it as is (plan Open question 5). Kept off mk_run so the limma runs are unchanged.
mk_count <- function(id, description, dataset, de_method, ..., known_production_behaviour = NULL) {
  r <- mk_run(id, description, dataset, "gene", de_method = de_method, gene_shim = TRUE, ...)
  r$known_production_behaviour <- known_production_behaviour
  r
}
count_ctrl <- list(
  airway = list(none = NULL,
                factor = list(Column = "cell", Type = "categorical"),
                factor_numeric = list(Column = c("cell", "avgLength"), Type = c("categorical", "numerical"))),
  count_synth = list(none = NULL,
                     factor = list(Column = "batch", Type = "categorical"),
                     factor_numeric = list(Column = c("batch", "rin"), Type = c("categorical", "numerical"))))
# Custom pairs put a non-reference level on the right, so DESeq2's relevel-and-refit
# (deseq2StatsFun.R:185) runs.
count_custom <- list(
  airway = list(left = "trt", right = "untrt"),
  count_synth = list(left = c("doseA", "doseB", "doseB"), right = c("ctrl", "ctrl", "doseA")))

# 4a. Core cross: engine x dataset x {all, custom, anova} x control (36 runs).
for (eng in c("edgeR", "DESeq2")) for (dsn in c("airway", "count_synth"))
  for (shape in c("all", "custom", "anova")) for (ctl in names(count_ctrl[[dsn]])) {
    id <- sprintf("count_%s_%s_%s_ctl%s", tolower(eng), dsn, shape, ctl)
    count_runs[[id]] <- mk_count(
      id, sprintf("%s %s, %s, control=%s", eng, dsn, shape, ctl), dsn, eng,
      mode = if (shape == "anova") "anova" else "discovery",
      comparison_type = if (shape == "custom") "custom" else "all",
      custom_comparisons = if (shape == "custom") count_custom[[dsn]] else NULL,
      control_cols = count_ctrl[[dsn]][[ctl]],
      known_production_behaviour = if (eng == "edgeR" && dsn == "airway" && ctl == "factor")
        paste("ENSG00000119698 gets a round-off negative QL F (-4.2e-10); production's sqrt(F) makes",
              "stat, SE and CI NA ('NaNs produced'). Compared under edger_f_floor."))
  }

# 4b. One-at-a-time from the count_synth / all / no-control default.
for (a in c(0.01, 0.1)) {
  id <- sprintf("count_deseq2_alpha%s", a)
  count_runs[[id]] <- mk_count(id, sprintf("DESeq2 independent filtering at alpha=%s", a),
                               "count_synth", "DESeq2", deseq2_alpha = a)
}
id <- "count_deseq2_anova_alpha0.1"
count_runs[[id]] <- mk_count(id, "DESeq2 LRT omnibus with alpha=0.1", "count_synth", "DESeq2",
                             mode = "anova", deseq2_alpha = 0.1)
for (dsn in c("airway", "count_synth")) for (sh in c("normal", "apeglm", "ashr")) {
  id <- sprintf("count_deseq2_%s_shrink_%s", dsn, sh)
  count_runs[[id]] <- mk_count(id, sprintf("DESeq2 %s, lfcShrink type=%s", dsn, sh), dsn, "DESeq2",
                               deseq2_lfc_shrinkage = sh,
                               notes = if (sh %in% c("apeglm", "ashr")) "G1: golden only; stays on the R path in Rust v1." else "")
}
for (sh in c("normal", "apeglm")) {
  id <- sprintf("count_deseq2_count_synth_shrink_%s_ctlfactor", sh)
  count_runs[[id]] <- mk_count(id, sprintf("DESeq2 shrink=%s with a batch covariate", sh), "count_synth", "DESeq2",
                               deseq2_lfc_shrinkage = sh, control_cols = count_ctrl$count_synth$factor)
}
# deseq2-rust shrink-core additions: every shrink type meets a factor + numeric control and a
# design with m - p <= 3 (airway + cell: 8 samples, 5 coefficients).
id <- "count_deseq2_count_synth_shrink_ashr_ctlfactor"
count_runs[[id]] <- mk_count(id, "DESeq2 shrink=ashr with a batch covariate", "count_synth", "DESeq2",
                             deseq2_lfc_shrinkage = "ashr", control_cols = count_ctrl$count_synth$factor)
for (sh in c("normal", "apeglm", "ashr")) {
  id <- sprintf("count_deseq2_count_synth_shrink_%s_ctlfactor_numeric", sh)
  count_runs[[id]] <- mk_count(id, sprintf("DESeq2 shrink=%s with batch + rin covariates", sh), "count_synth",
                               "DESeq2", deseq2_lfc_shrinkage = sh,
                               control_cols = count_ctrl$count_synth$factor_numeric)
  id <- sprintf("count_deseq2_airway_shrink_%s_ctlfactor", sh)
  count_runs[[id]] <- mk_count(id, sprintf("DESeq2 airway shrink=%s with cell (m - p = 3)", sh), "airway",
                               "DESeq2", deseq2_lfc_shrinkage = sh, control_cols = count_ctrl$airway$factor)
}
# No apeglm_seed run: seed 2 was byte-identical to seed 1 on count_synth, so it tested nothing.
for (nm in c("RLE", "upperquartile", "none")) {
  id <- sprintf("count_edger_norm_%s", nm)
  count_runs[[id]] <- mk_count(id, sprintf("edgeR calcNormFactors(method=%s)", nm), "count_synth", "edgeR",
                               edger_norm_method = nm)
}
count_runs[["count_edger_airway_norm_RLE"]] <- mk_count(
  "count_edger_airway_norm_RLE", "edgeR airway, calcNormFactors(method=RLE)", "airway", "edgeR",
  edger_norm_method = "RLE")

# 4c. Edge cases, both engines unless the case is engine-specific.
for (eng in c("edgeR", "DESeq2")) {
  e <- tolower(eng)
  count_runs[[sprintf("edge_%s_filter_drops_all", e)]] <- mk_count(
    sprintf("edge_%s_filter_drops_all", e), "Every count <= 1, filterByExpr keeps nothing", "count_synth", eng,
    transform = "tiny_counts", expected_error = "filterByExpr removed every gene")
  count_runs[[sprintf("edge_%s_one_rep_per_condition", e)]] <- mk_count(
    sprintf("edge_%s_one_rep_per_condition", e), "One replicate per condition (no residual df)", "count_synth", eng,
    transform = "one_rep_per_condition",
    expected_error = c(edgeR = "missing value where TRUE/FALSE needed",
                       DESeq2 = "same number of samples and coefficients")[[eng]],
    notes = paste("Production fails here. edgeR warns 'No residual df: setting dispersion to NA' and then",
                  "crashes on an unguarded NA test; DESeq2 refuses to estimate dispersion without replicates."),
    known_production_behaviour = if (eng == "edgeR")
      "Unhandled R crash, not a contract: the port gates on 'errors', not on this message text.")
  count_runs[[sprintf("edge_%s_cooks", e)]] <- mk_count(
    sprintf("edge_%s_cooks", e), "7 reps per condition with injected outliers", "count_synth_cooks", eng,
    notes = "DESeq2 replaces outlier counts (minReplicatesForReplace = 7) and refits; edgeR is the same data for contrast.",
    known_production_behaviour = if (eng == "DESeq2")
      paste("comparison_type = all gives ctrl - doseA, ctrl - doseB, doseA - doseB: the reference level",
            "ctrl is never on the right, so every pair takes the relevel-and-nbinomWaldTest refit",
            "(deseq2StatsFun.R:183-190), which refits on the original counts and undoes the Cook's",
            "replacement. The port reproduces this; do not fix it toward DESeq2's documented behaviour.",
            "The no-refit branch on replaced counts is edge_deseq2_cooks_ref_right."))
  count_runs[[sprintf("edge_%s_rank_deficient", e)]] <- mk_count(
    sprintf("edge_%s_rank_deficient", e), "Covariate collinear with condition", "count_synth", eng,
    transform = "collinear_covariate", control_cols = list(Column = "condition_dup", Type = "categorical"),
    expected_error = "perfectly collinear")
  count_runs[[sprintf("edge_%s_missing_rows", e)]] <- mk_count(
    sprintf("edge_%s_missing_rows", e), "Every 19th long-table row absent: one NA cell (coerced to 0) in 1263 of 2000 genes", "count_synth", eng,
    transform = "drop_count_rows")
  count_runs[[sprintf("edge_%s_unicode_conditions", e)]] <- mk_count(
    sprintf("edge_%s_unicode_conditions", e), "Non-syntactic / unicode condition labels", "count_synth", eng,
    transform = "unicode_count_conditions")
}
# The no-refit branch: the reference level ctrl is on the right of both pairs, so results() runs
# on the Cook's-replaced fit and the relevel is a no-op.
count_runs[["edge_deseq2_cooks_ref_right"]] <- mk_count(
  "edge_deseq2_cooks_ref_right", "Cook's replacement, right side = reference level (no refit)",
  "count_synth_cooks", "DESeq2", comparison_type = "custom",
  custom_comparisons = list(left = c("doseA", "doseB"), right = c("ctrl", "ctrl")))
count_runs[["edge_deseq2_cooks_anova"]] <- mk_count(
  "edge_deseq2_cooks_anova", "Cook's replacement under the LRT", "count_synth_cooks", "DESeq2", mode = "anova")
# The two-group Cook's rescue (results() keeps a flagged gene's p-value when >= 3 samples exceed
# the flagged count): one two-level factor, no covariate, 4 v 4 so nothing is replaced.
count_runs[["edge_deseq2_cooks_rescue"]] <- mk_count(
  "edge_deseq2_cooks_rescue", "Cook's two-group rescue: 6 rescued and 3 flagged airway genes", "airway", "DESeq2",
  transform = "cooks_rescue",
  notes = "Gates the rescue's NA pattern exactly; edge_deseq2_cooks* never reach it (3 levels).")
count_runs[["edge_deseq2_non_integer"]] <- mk_count(
  "edge_deseq2_non_integer", "Non-integer counts", "count_synth", "DESeq2",
  transform = "non_integer_counts", expected_error = "Non-integer values detected")
count_runs[["edge_edger_negative"]] <- mk_count(
  "edge_edger_negative", "One negative count", "count_synth", "edgeR",
  transform = "negative_intensity", expected_error = "negative")
count_runs[["edge_edger_protein_entity"]] <- mk_run(
  "edge_edger_protein_entity", "edgeR requested on protein entity", "bojkova2020", "protein",
  de_method = "edgeR", expected_error = "only supported for gene entity type")

stopifnot(!any(duplicated(names(count_runs))), !any(names(count_runs) %in% names(runs)))
