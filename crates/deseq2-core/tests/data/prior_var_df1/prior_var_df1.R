# Regenerates tests/data/prior_var_df1 (review r1, stats item 8a). Run in the production image:
# docker run --rm --platform linux/amd64 -v "$PWD":/w -w /w md-flexi-r45-local:latest Rscript prior_var_df1.R
# The m - p = 1 simulation under set.seed(2): the simulated histograms depend only on df and
# the seed, so this pins the rchisq(df = 1) draws (gamma GS branch, exp_rand) and the rnorm
# interleaving bit for bit. The residuals are synthetic with variance 1.5 so the argmin is
# interior rather than floored at 0.25. Body copied from corpus/count-reference/ref_deseq2.R.
plain_prior_var_sim <- function(obs, df) {
  set.seed(2)
  brks <- -20:20 / 2
  obs <- obs[obs > min(brks) & obs < max(brks)]
  grid <- seq(from = 0, to = 8, length = 200)
  obs_hist <- hist(obs, breaks = brks, plot = FALSE)
  sim_counts <- matrix(0L, length(grid), length(brks) - 1)
  kl <- numeric(length(grid))
  for (i in seq_along(grid)) {
    rd <- log(rchisq(10000, df = df)) + rnorm(10000, 0, sqrt(grid[i])) - log(df)
    rd <- rd[rd > min(brks) & rd < max(brks)]
    rh <- hist(rd, breaks = brks, plot = FALSE)
    sim_counts[i, ] <- rh$counts
    z <- c(obs_hist$density, rh$density)
    small <- min(z[z > 0])
    kl[i] <- sum(obs_hist$density * (log(obs_hist$density + small) - log(rh$density + small)))
  }
  lofit <- loess(kl ~ grid, span = 0.2)
  fine_grid <- seq(from = 0, to = 8, length = 1000)
  fine_fitted <- predict(lofit, fine_grid)
  j <- which.min(fine_fitted)
  list(sim_counts = sim_counts, kl = kl, loess_fitted = unname(fitted(lofit)),
       fine_fitted = fine_fitted, j = j, prior_var = pmax(fine_grid[j], 0.25))
}
set.seed(11)
obs <- log(rchisq(3000, df = 1)) + rnorm(3000, 0, sqrt(1.5))
sim <- plain_prior_var_sim(obs, 1)
w <- function(x, f) writeLines(sprintf("%.17g", x), f)
w(obs, "obs.txt")
write.table(sim$sim_counts, "sim_counts.txt", row.names = FALSE, col.names = FALSE)
w(sim$kl, "kl.txt")
w(sim$loess_fitted, "loess_fitted.txt")
w(sim$fine_fitted, "fine_predicted.txt")
writeLines(c(sprintf("%d", sim$j), sprintf("%.17g", sim$prior_var)), "argmin.txt")
cat("argmin index", sim$j, "prior var", sim$prior_var, "\n")
