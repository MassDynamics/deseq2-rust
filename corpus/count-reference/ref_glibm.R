# Reference exp / log / log1p values from the reference R's glibc (2.34, x86_64 FMA build),
# for shrink-core's glibm port. Run in md-flexi-r45-local:
#   docker run --rm --platform linux/amd64 -v <corpus>/reference-glibm:/out md-flexi-r45-local \
#     Rscript /src/ref_glibm.R
# Output: glibm_ref.bin, little-endian doubles: xe, exp(xe), xl, log(xl), xp, log1p(xp).
set.seed(20261002)
n <- 1e5
xe <- c(runif(n, -40, 20), rnorm(n, 4, 2), -745.2, -708.5, 709.7, 0, 1e-300)
xl <- c(exp(runif(n, -700, 700)), runif(n, 0.9, 1.1), 1, 4.9e-324, 2.2e-308)
xp <- c(exp(runif(n, -60, 30)), runif(n, -0.5, 0.5), -0.9999, 1e-20, 0)
con <- file("/out/glibm_ref.bin", "wb")
writeBin(c(length(xe), length(xl), length(xp)) + 0, con)
writeBin(c(xe, exp(xe), xl, log(xl), xp, log1p(xp)), con)
close(con)
cat(R.version.string, "\n")
