# Regenerates tests/data/ash_small_wide2 (review r1): docker run --rm -v "$PWD":/w md-flexi-r45-local:latest Rscript /w/ash_case.R ash_small_wide2
args <- commandArgs(TRUE); case <- args[1]
out <- file.path("/w/cases", case); dir.create(out, showWarnings=FALSE, recursive=TRUE)
wn <- function(x, f) { x <- as.matrix(x); s <- matrix(sprintf("%.17g", x), nrow=nrow(x)); s[is.na(x)] <- "NA"; s[is.nan(x)] <- "NA"; write.table(s, file.path(out,f), quote=FALSE, row.names=FALSE, col.names=FALSE) }
set.seed(7)
gen <- switch(case,
 ash_big_m = { n <- 2000; s <- exp(runif(n, log(0.005), log(0.5))); b <- rnorm(n, 0, sqrt(s^2 + 0.3^2)); b[1:30] <- sample(c(-1,1),30,TRUE)*runif(30,20,28); s[1:30] <- runif(30,3,4.5); list(b=b,s=s)},
 ash_big_m2 = { n <- 6000; s <- exp(runif(n, log(0.01), log(1))); b <- rnorm(n, 0, sqrt(s^2 + 0.5^2)); b[1:50] <- sample(c(-1,1),50,TRUE)*runif(50,15,25); s[1:50] <- runif(50,2,4); list(b=b,s=s)},
 ash_equal_se = { n <- 500; s <- rep(0.3, n); b <- rnorm(n, 0, 0.6); list(b=b,s=s)},
 ash_special = { n <- 300; s <- exp(runif(n, log(0.05), log(1))); b <- rnorm(n,0,sqrt(s^2+0.4^2)); s[1] <- 0; s[2] <- Inf; b[3] <- NA; s[4] <- NA; b[5] <- NA; s[5] <- NA; list(b=b,s=s)},
 ash_n1 = list(b=1.3, s=0.4),
 ash_n2 = list(b=c(1.3,-0.2), s=c(0.4,0.3)),
 ash_n3 = list(b=c(2.5,-0.2,0.1), s=c(0.4,0.3,0.5)),
 ash_allnull = { n <- 200; s <- runif(n, 0.2, 0.6); b <- runif(n,-1,1)*s; list(b=b,s=s)},
 ash_small_wide = { n <- 12; s <- exp(runif(n, log(0.01), log(2))); b <- rnorm(n,0,3); list(b=b,s=s)},
 ash_small_wide2 = { n <- 40; s <- exp(runif(n, log(0.005), log(2))); b <- rnorm(n,0,4); b[1] <- 25; s[1] <- 3; list(b=b,s=s)},
ash_rank1 = list(b=rep(0.4, 50), s=rep(0.3, 50)),
 ash_rank3 = list(b=rep(c(0.4,-1.2,2.5), 40), s=rep(c(0.3,0.5,0.2), 40)),
 ash_rank3n = { b <- rep(c(0.4,-1.2,2.5), 40); s <- rep(c(0.3,0.5,0.2), 40); b[1] <- 0.4 + 1e-9; list(b=b,s=s)},
 ash_dup = { n <- 400; s <- rep(c(0.2,0.5), n/2); b <- rep(c(0.1,-0.3,1.5,0), n/4); list(b=b,s=s)},
)
wn(cbind(gen$b, gen$s), "ashr_in.txt")
b <- gen$b; s <- gen$s
# replicate ashr's L to inspect tsvd + mixsqp
data <- ashr::set_data(b, s, ashr:::lik_normal(), 0)
mixsd <- ashr:::autoselect.mixsd(data, sqrt(2), 0, c(-Inf,Inf), "normal")
ex <- ashr:::get_exclusions(data)
ll <- sapply(mixsd, function(sd) { v <- sqrt(s[!ex]^2+sd^2); dnorm(b[!ex]/v, log=TRUE) - log(v) })
ll <- matrix(ll, ncol=length(mixsd)); L <- exp(ll - apply(ll,1,max))
L <- L[, apply(L,2,max) > 0, drop=FALSE]
tv <- if (ncol(L) > 4) tryCatch(mixsqp:::tsvd(L, 1e-6), error=function(e) "ERR") else "skipped(m<=4)"
cat(case, "n=", sum(!ex), "m=", length(mixsd), "tsvd:", if (is.null(tv)) "NULL(full L)" else if (is.character(tv)) tv else paste("LOW-RANK rank", ncol(tv$U)), "\n")
sv <- svd(L)$d; cat("  singular values (first 10):", sprintf("%.2e", head(sv,10)), "\n")
ash <- tryCatch(withCallingHandlers(ashr::ash(b, s, mixcompdist="normal", method="shrink"),
    warning=function(w) {cat("  R WARNING:", conditionMessage(w), "\n"); invokeRestart("muffleWarning")}), error=function(e) e)
if (inherits(ash, "error")) cat("  R ERROR:", conditionMessage(ash), "\n") else {
  r <- ash$result; wn(cbind(r$PosteriorMean, r$PosteriorSD, r$NegativeProb, r$lfsr, r$svalue), "ashr_R.txt")
  wn(ash$fitted_g$pi, "ashr_R_pi.txt")
}
