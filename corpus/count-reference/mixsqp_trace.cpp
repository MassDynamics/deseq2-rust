// Instrumented copy of mixsqp 0.3-54 src/mixsqp.cpp + mixem.cpp (full-L path only),
// compiled in the same image, used by trace_mixsqp.R to record the SQP / active-set
// path that shrink-core's mixsqp port is compared against.
//
// The arithmetic is the package code unchanged. Additions only record values: per SQP
// iteration the EM-updated x, objective, gmin, QP solution y, step size and xnew; per
// active-set iteration the working-set size, the identity correction a, guess_sympd(B,16),
// rcond(B) (LU 1-norm estimate), whether solve(no_approx) succeeds (false means the
// default solve took the approximate SVD route), and the step.

// [[Rcpp::depends(RcppArmadillo)]]
#define ARMA_DONT_PRINT_ERRORS
#include <RcppArmadillo.h>
#include <vector>
using namespace Rcpp;
using namespace arma;

static std::vector<double> g_qp;   // rows of 12
static int g_mode = 0;              // 0 = package solve; 1 = pinv in the approx case; 2 = y perturbed 1e-15 in the approx case
static int g_sqp_iter = 0;

static void scalecols (mat& A, const vec& b) {
  unsigned int n = A.n_cols;
  for (unsigned int i = 0; i < n; i++) A.col(i) *= b[i];
}
static void normalizerows (mat& A) { vec b = sum(A,1); A.each_col() /= b; }
static void normalizerowsbymax (mat& A) { vec b = max(A,1); A.each_col() /= b; }
static void mixem_update (const mat& L, const vec& w, vec& x, mat& P) {
  double e = 1e-15;
  P = L; scalecols(P,x + e); normalizerowsbymax(P); P += e; normalizerows(P);
  x = trans(P) * w;
}
static double objh (const vec& u, const vec& w, const vec& z) {
  if (u.min() <= 0) stop("Objective is -Inf");
  return -sum(w % (z + log(u)));
}
static double obj (const mat& L, const vec& w, const vec& x, const vec& z, const vec& e) {
  vec u = L*x; u += e; return objh(u,w,z);
}
static void compute_grad (const mat& L, const vec& w, const vec& x, const vec& e,
                          vec& g, mat& H, mat& Z) {
  vec u = L*x + e;
  g = -trans(L) * (w/u);
  Z = L;
  Z.each_col() %= (sqrt(w)/u);
  H = trans(Z) * Z;
}
static inline void feasible_stepsize (const vec& x, const vec& p, int& j, double& a) {
  uvec i = find(p < 0); a = 1; j = -1;
  if (!i.is_empty()) { vec t = -x(i)/p(i); j = t.index_min(); if (t(j) < 1) a = t(j); j = i(j); }
}
static double g_a_corr; static int g_tries; static int g_guess; static double g_rcond; static int g_noapprox_ok;
static void searchdir (const mat& H, const vec& y, vec& p, mat& B, double ainc) {
  double a0 = 1e-15, amax = 1e15; int n = y.n_elem;
  mat I(n,n,fill::eye); mat R(n,n);
  double d = H.diag().min(); double a = (d > a0) ? 0 : a0 - d;
  int tries = 0;
  while (true) {
    B = H + a*I; tries++;
    if (a*ainc > amax) break; else if (chol(R,B)) break; else if (a <= 0) a = a0; else a *= ainc;
  }
  g_a_corr = a; g_tries = tries;
  g_guess = sym_helper::guess_sympd(B, uword(16)) ? 1 : 0;
  g_rcond = arma::rcond(B);
  vec p2; g_noapprox_ok = solve(p2, B, -y, solve_opts::no_approx) ? 1 : 0;
  if (g_mode == 0 || g_noapprox_ok) p = solve(B,-y);
  else if (g_mode == 1) p = pinv(B) * (-y);
  else { vec y2 = y % (1 + 1e-15 * linspace<vec>(-1, 1, n)); p = solve(B,-y2); }
}
static int activesetqp (const mat& H, const vec& g, vec& y, int maxiter,
                        double zerosearchdir, double tol, double ainc) {
  int m = g.n_elem; int k = -1; int iter; double a;
  vec b(m), p(m), bs(m), ps(m); mat Hs(m,m), Bs(m,m); uvec i(m), j(m); bool add;
  uvec t = (y > 0);
  for (iter = 0; iter < maxiter; iter++) {
    i = find(t != 0); j = find(t == 0);
    y(j).fill(0);
    Hs = H(i,i); b = g; b(i) += Hs*y(i); bs = b(i);
    p.fill(0);
    searchdir(Hs,bs,ps,Bs,ainc);
    p(i) = ps;
    a = 1; add = false;
    double pn = norm(p,"inf"); int kind;
    if (pn <= zerosearchdir) {
      kind = 0;
      b = g + H*y;
      if (j.is_empty()) { kind = 2; }
      else if (b(j).min() >= -tol) { kind = 3; }
      else { k = j(b(j).index_min()); t(k) = 1; }
    } else {
      kind = 1;
      feasible_stepsize(y,p,k,a);
      if (k >= 0 && a < 1) { if (i.n_elem > 1) add = true; }
      y += a*p; j = find(y < 0); y(j).fill(0); y(j).fill(0);
      if (add) { t(k) = 0; y(k) = 0; }
    }
    double row[12] = {double(g_sqp_iter), double(iter), double(i.n_elem), g_a_corr, double(g_tries),
                      double(g_guess), g_rcond, double(g_noapprox_ok), pn, double(kind), double(k), a};
    g_qp.insert(g_qp.end(), row, row + 12);
    if (kind == 2 || kind == 3) { iter++; break; }
  }
  return iter;
}
static int linesearch (double f, const mat& L, const vec& w, const vec& z, const vec& g,
                       const vec& x, const vec& y, const vec& e, double suffdecr,
                       double beta, double amin, double& a, vec& xnew) {
  int k; double afeas, fnew; int nls = 0;
  vec p = y - x; feasible_stepsize(x,p,k,afeas);
  if (afeas <= amin) { a = afeas; xnew = afeas*y + (1 - afeas)*x; }
  else {
    a = (1 < afeas) ? 1 : afeas;
    while (true) {
      xnew = a*y + (1 - a)*x; fnew = obj(L,w,xnew,z,e); nls++;
      if ((xnew.min() >= 0) && (fnew + sum(xnew) <= f + sum(x) + suffdecr*a*dot(y - x,g + 1))) break;
      else if (a*beta < amin) { a = amin; xnew = a*y + (1 - a)*x; if (xnew.min() < 0) { a = 0; xnew = x; } break; }
      a *= beta;
    }
  }
  return nls;
}

// [[Rcpp::export]]
List mixsqp_trace (const arma::mat& L, const arma::vec& w, const arma::vec& z,
                   const arma::vec& x0, const arma::vec& eps, int numiter_em,
                   int maxitersqp, int maxiteractiveset, int mode = 0) {
  g_qp.clear(); g_mode = mode;
  int m = L.n_cols;
  mat P = L; vec x = x0;
  mat em_x(m, numiter_em);
  for (int i = 0; i < numiter_em; i++) { mixem_update(L,w,x,P); em_x.col(i) = x; }
  vec xem_final = x;
  double convtolsqp = 1e-8, convtolas = 1e-10, zts = 1e-8, ztsd = 1e-14, suffdecr = 0.01,
         beta = 0.75, amin = 1e-8, ainc = 10;
  mat X_em(m, maxitersqp, fill::zeros), X_y(m, maxitersqp, fill::zeros), X_new(m, maxitersqp, fill::zeros);
  vec objv(maxitersqp), gminv(maxitersqp), stepv(maxitersqp), nqpv(maxitersqp), nlsv(maxitersqp);
  vec g(m), ghat(m), y(m), xnew(m); mat H(m,m), Z; uvec j0, j1;
  double status = 1; int i;
  for (i = 0; i < maxitersqp; i++) {
    g_sqp_iter = i;
    mixem_update(L,w,x,P);
    X_em.col(i) = x;
    j0 = find(x <= zts); j1 = find(x > zts); x(j0).fill(0);
    objv(i) = obj(L,w,x,z,eps);
    compute_grad(L,w,x,eps,g,H,Z);
    gminv(i) = 1 + g(j1).min();
    if (gminv(i) >= -convtolsqp) { status = 0; i++; break; }
    ghat = g - H*x + 1; y = x;
    nqpv(i) = activesetqp(H,ghat,y,maxiteractiveset,ztsd,convtolas,ainc);
    X_y.col(i) = y;
    double a; nlsv(i) = linesearch(objv(i),L,w,z,g,x,y,eps,suffdecr,beta,amin,a,xnew);
    stepv(i) = a; X_new.col(i) = xnew;
    x = xnew;
  }
  NumericMatrix qp(g_qp.size()/12, 12);
  for (size_t r = 0; r < g_qp.size()/12; r++) for (int c = 0; c < 12; c++) qp(r,c) = g_qp[r*12+c];
  return List::create(Named("x") = x, Named("status") = status, Named("em_x") = em_x,
    Named("xem_final") = xem_final, Named("X_em") = X_em.cols(0,i-1),
    Named("X_y") = X_y.cols(0,i-1), Named("X_new") = X_new.cols(0,i-1),
    Named("obj") = objv.head(i), Named("gmin") = gminv.head(i), Named("step") = stepv.head(i),
    Named("nqp") = nqpv.head(i), Named("nls") = nlsv.head(i), Named("qp") = qp,
    Named("niter") = i);
}
