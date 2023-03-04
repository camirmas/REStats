data {
  int<lower=0> N;
  vector[N] y;
}
parameters {
  real<lower=0> alpha;
  real<lower=0> sigma;
}
model {
  alpha ~ normal(2, 2);
  sigma ~ normal(0, 10);
  y ~ weibull(alpha, sigma);
}
generated quantities {
  vector[N] log_lik;
  vector[N] y_rep;

  for (i in 1:1000) {
    y_rep[i] = weibull_rng(alpha, sigma);  
  }

  for (n in 1:N) {
    log_lik[n] = weibull_lpdf(y[n] | alpha, sigma);
  }
}