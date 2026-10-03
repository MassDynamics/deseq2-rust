# Regenerate tests/data/ext_cases.txt: docker run --rm --platform linux/amd64 -v "$PWD":/w md-flexi-r45-local:latest Rscript /w/ext_cases.R
set.seed(7)
out <- c()
for (i in 1:3000) {
  n <- sample(2:25, 1)
  x <- switch(i %% 3 + 1, rnorm(n) * 10^runif(n, -3, 8), round(rexp(n) * 1000) / sample(c(1, 3, 7), 1), log(rpois(n, 50) + 1) - 3.9)
  out <- c(out, paste(c(sprintf("%.17g", x), "|", sprintf("%.17g", c(sum(x), mean(x), median(x), mean(x, trim = 0.2), mean(x, trim = 1/8), rowMeans(matrix(x, 1))))), collapse = " "))
}
writeLines(out, "/w/ext_cases.txt")
