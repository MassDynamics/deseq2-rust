# Record the mixsqp SQP / active-set path for every ashr golden in reference-shrink/, using
# the instrumented mixsqp_trace.cpp, after checking it reproduces the dumped solution bit for
# bit. Writes shrink_cmp_XX_ashr_trace_{sqp,qp}.csv next to the existing ashr goldens.
#   Rscript trace_mixsqp.R   (inside the pinned image; see gen_mixsqp_trace.sh)
here <- dirname(normalizePath(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))))
corpus <- Sys.getenv("MD_COUNT_CORPUS_DIR", "/corpus")
Rcpp::sourceCpp(file.path(here, "mixsqp_trace.cpp"), cacheDir = tempdir())
fw <- function(df, path) {
  df <- as.data.frame(df)
  for (cc in names(df)) if (is.double(df[[cc]]))
    df[[cc]] <- ifelse(is.na(df[[cc]]), NA_character_, sprintf("%.17g", df[[cc]]))
  data.table::fwrite(df, path, na = "NA", quote = FALSE)
}
root <- file.path(corpus, "reference-shrink")
ok <- TRUE
for (run in list.files(root, pattern = "_ashr")) {
  for (Lf in list.files(file.path(root, run), pattern = "_ashr_L\\.csv$", full.names = TRUE)) {
    pre <- sub("_L\\.csv$", "", Lf)
    L <- as.matrix(data.table::fread(Lf, colClasses = list(character = "id"))[, -1])
    dimnames(L) <- NULL
    n <- nrow(L); m <- ncol(L)
    stopifnot(all(apply(L, 2, max) > 0), all(apply(L, 1, max) == 1))
    w <- rep(1, n); w <- w / sum(w)
    x0 <- rep(1, m); x0 <- x0 / sum(x0)
    eps <- rep(1e-6 - min(0, min(L)), n)
    tr <- mixsqp_trace(L, w, rep(0, n), x0, eps, 20L, 1000L, as.integer(min(20, m + 1)))
    gold <- data.table::fread(paste0(pre, "_mixsqp_x.csv"))
    xs <- drop(tr$x); xs <- xs / sum(xs)
    same <- identical(xs, gold$x) && identical(drop(tr$xem_final), gold$em_x)
    cat(sprintf("%s %s n=%d m=%d sqp_iter=%d identical=%s\n", run, basename(pre), n, m, tr$niter, same))
    ok <- ok && same
    k <- tr$niter
    sqp <- data.frame(iter = seq_len(k) - 1, obj = tr$obj, gmin = tr$gmin, step = tr$step,
                      nqp = tr$nqp, nls = tr$nls)
    sqp$step[k] <- NA; sqp$nqp[k] <- NA; sqp$nls[k] <- NA
    add <- function(M, tag) { M <- t(M); colnames(M) <- paste0(tag, seq_len(m)); M }
    Y <- add(tr$X_y, "y"); Xn <- add(tr$X_new, "xnew"); Y[k, ] <- NA; Xn[k, ] <- NA
    sqp <- cbind(sqp, add(tr$X_em, "xem"), Y, Xn)
    fw(sqp, paste0(pre, "_trace_sqp.csv"))
    qp <- as.data.frame(tr$qp)
    names(qp) <- c("sqp_iter", "qp_iter", "n_ws", "a_corr", "chol_tries", "guess_sympd",
                   "rcond", "noapprox_ok", "pnorm_inf", "kind", "k", "step")
    fw(qp, paste0(pre, "_trace_qp.csv"))
    emx <- t(tr$em_x); colnames(emx) <- paste0("x", seq_len(m))
    fw(as.data.frame(emx), paste0(pre, "_trace_em.csv"))
  }
}
if (!ok) stop("trace does not reproduce the golden mixsqp solution")
