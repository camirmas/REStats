data {
  int<lower=0> K; // lags
  // int<lower=0> steps; // num steps
  int<lower=0> N; // num observations
  array[N] real y; // power time-series
}
parameters {
  real alpha;
  array[K] real beta;
  real<lower=0> sigma;
}
generated quantities {
  array[N] real y_pred;

  for (n in 1:K)
    y_pred[n] = y[n];
  
  for (n in (K+1):N) {
    real mu = alpha;

    for (k in 1:K) {
      mu += beta[k] * y[n-k];
    }
    
    y_pred[n] = normal_rng(mu, sigma);
  }

  // predict forward
  // for (n in N+1:N+steps) {
  //   real mu = alpha;

  //   for (k in 1:K) {
  //     mu += beta[k] * y_pred[n-k];
  //   }
    
  //   y_pred[n] = normal_rng(mu, sigma);
  // }
}
