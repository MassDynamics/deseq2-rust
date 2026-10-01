// Instrumented copy of apeglm 1.32.0 src/nbinomGLM.cpp, compiled in the same image
// with the same flags, used by ref_shrink.R to record the LBFGSpp optimizer path.
//
// The objective, gradient and optimizer call are the package code unchanged. The only
// addition is that every f_grad evaluation (beta, f, grad) is appended to a trace for
// the genes flagged in `trace_cols`, and every gene's evaluation count is returned.
// ref_shrink.R checks that betas / value / convergence equal apeglm:::nbinomGLM bit
// for bit before trusting the trace.

// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::depends(RcppNumerical)]]

#include <RcppNumerical.h>
#include <vector>
using namespace Numer;

typedef Eigen::Map<Eigen::MatrixXd> MapMat;
typedef Eigen::Map<Eigen::VectorXd> MapVec;

static std::vector<double> g_trace;
static int g_nevals = 0;
static bool g_record = false;

class optimFunTrace: public MFuncGrad
{
private:
  const MapMat x;
  const MapMat Y;
  const MapVec size;
  const MapMat weights;
  const MapMat offset;
  double sigma2;
  double S2;
  const MapVec no_shrink;
  const MapVec shrink;
  const MapVec cnst;
  int i;
public:
  optimFunTrace(const MapMat x_, const MapMat Y_, const MapVec size_, const MapMat weights_, const MapMat offset_, double sigma2_, double S2_, const MapVec no_shrink_, const MapVec shrink_, const MapVec cnst_, int i_) : x(x_), Y(Y_), size(size_), weights(weights_), offset(offset_), sigma2(sigma2_), S2(S2_), no_shrink(no_shrink_), shrink(shrink_), cnst(cnst_), i(i_) {}

  double f_grad(Constvec& beta, Refvec grad)
  {

    int B = x.cols(); // e.g. number of betas
    
    Eigen::ArrayXd xbeta = x * beta;
    Eigen::ArrayXd xbeta_off = xbeta + offset.col(i).array();
    Eigen::ArrayXd exp_xbeta_off = xbeta_off.exp();

    Eigen::ArrayXd a = Y.col(i).array() + size[i];
    Eigen::ArrayXd b = exp_xbeta_off + size[i];

    Eigen::VectorXd c = Y.col(i).array() - a * exp_xbeta_off * b.inverse();
    Eigen::VectorXd cw = c.array() * weights.col(i).array();
    
    Eigen::ArrayXd d = Y.col(i).array() * xbeta - a * b.log();
    Eigen::ArrayXd dw = d * weights.col(i).array();

    double neg_prior = 0.0;
    Eigen::ArrayXd d_neg_prior(B);

    for (int j = 0; j < no_shrink.size(); j++) {
      int k = no_shrink[j] - 1;
      neg_prior += pow(beta[k], 2.0)/(2.0 * sigma2);
      d_neg_prior[k] = beta[k]/sigma2;
    }

    for (int j = 0; j < shrink.size(); j++) {
      int k = shrink[j] - 1;
      neg_prior += log1p(pow(beta[k], 2.0)/S2);
      d_neg_prior[k] = 2.0 * beta[k] / (S2 + pow(beta[k], 2.0));
    }

    const double f = -1.0 * dw.sum() / cnst[i] + neg_prior / cnst[i] + 10.0;

    Eigen::ArrayXd d_neg_lik = -1.0 * x.transpose() * cw;
    grad = d_neg_lik / cnst[i] + d_neg_prior / cnst[i];

    g_nevals++;
    if (g_record) {
      g_trace.push_back((double) i);
      for (int k = 0; k < B; k++) g_trace.push_back(beta[k]);
      g_trace.push_back(f);
      for (int k = 0; k < B; k++) g_trace.push_back(grad[k]);
    }
    return f;
  }
};

// [[Rcpp::export]]
Rcpp::List nbinomGLMTrace(Rcpp::NumericMatrix x, Rcpp::NumericMatrix Y,
		     Rcpp::NumericVector size, Rcpp::NumericMatrix weights,
		     Rcpp::NumericMatrix offset, double sigma2, double S2,
		     Rcpp::NumericVector no_shrink, Rcpp::NumericVector shrink,
		     Rcpp::NumericVector init, Rcpp::NumericVector cnst,
		     Rcpp::LogicalVector trace_cols)
{
  const MapMat mx = Rcpp::as<MapMat>(x);
  const MapMat mY = Rcpp::as<MapMat>(Y);
  const MapVec msize = Rcpp::as<MapVec>(size);
  const MapMat mweights = Rcpp::as<MapMat>(weights);
  const MapMat moffset = Rcpp::as<MapMat>(offset);
  const MapVec mno_shrink = Rcpp::as<MapVec>(no_shrink);
  const MapVec mshrink = Rcpp::as<MapVec>(shrink);
  const MapVec minit = Rcpp::as<MapVec>(init);
  const MapVec mcnst = Rcpp::as<MapVec>(cnst);

  int G = Y.ncol();
  int B = x.ncol();

  Eigen::MatrixXd betas(B, G);
  Eigen::VectorXd beta(B);

  Rcpp::NumericVector value(G);
  Rcpp::IntegerVector convergence(G);
  Rcpp::IntegerVector nevals(G);
  g_trace.clear();

  double fopt;
  for (int i = 0; i < G; i++) {
    optimFunTrace nll(mx, mY, msize, mweights, moffset, sigma2, S2, mno_shrink, mshrink, mcnst, i);
    beta = minit;
    g_nevals = 0;
    g_record = trace_cols[i];
    int status = optim_lbfgs(nll, beta, fopt, 300, 1e-8, 1e-8);
    betas.col(i) = beta;
    value[i] = fopt;
    convergence[i] = status;
    nevals[i] = g_nevals;
  }
  Rcpp::NumericVector tr(g_trace.begin(), g_trace.end());
  return Rcpp::List::create(Rcpp::Named("betas") = betas,
			    Rcpp::Named("value") = value,
			    Rcpp::Named("convergence") = convergence,
			    Rcpp::Named("nevals") = nevals,
			    Rcpp::Named("trace") = tr);
}
