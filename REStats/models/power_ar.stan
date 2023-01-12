data {
  int<lower=0> K; // lags
  int<lower=0> N; // num observations
  array[N] real y; // power time-series
}
parameters {
  real alpha;
  array[K] real beta;
  real<lower=0> sigma;
}
model {
  alpha ~ normal(0, 10);
  beta[K] ~ normal(0, 1);
  sigma ~ lognormal(0, 0.5);

  for (n in (K+1):N) {
    real mu = alpha;

    for (k in 1:K) {
      mu += beta[k] * y[n-k];
    }
    
    y[n] ~ normal(mu, sigma);
  }
}
