fr <- function(x) { s <- 0; for (i in 1:(length(x)-1)) { a <- x[i+1] - x[i]*x[i]; b <- 1 - x[i]; s <- s + 100*a*a + b*b }; s }
cases <- list(
  list(par=c(-1.2, 1), lo=c(-30,-30), up=c(30,30)),
  list(par=c(-1.2, 1), lo=c(-2,-2), up=c(0.5, 2)),
  list(par=c(3, -3, 2, 0.5), lo=rep(-4,4), up=rep(4,4)),
  list(par=c(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7), lo=rep(-1,7), up=c(rep(2,6), 0.8)),
  list(par=c(40, -35), lo=c(-30,-30), up=c(30,30))
)
for (k in seq_along(cases)) { cs <- cases[[k]]
  o <- optim(cs$par, fr, method="L-BFGS-B", lower=cs$lo, upper=cs$up)
  cat(sprintf("case %d conv %d counts %d value %.17g msg %s\n", k, o$convergence, o$counts[1], o$value, o$message))
  cat("par", sprintf("%.17g", o$par), "\n")
}
