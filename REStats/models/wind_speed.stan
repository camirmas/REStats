data {
  int<lower=1> T;            // num observations
  array[T] real y;           // observed outputs
}
parameters {
  vector[2] phi;                  // autoregression coeff
  real theta;                // moving avg coeff
  real<lower=0> sigma;       // noise scale
}
transformed parameters {
  vector[T] nu;              // prediction for time t
  vector[T] err;             // error for time t

  nu[1] = y[1];              
  nu[2] = y[2];
  err[1] = 0;
  err[2] = 0;
  for (t in 3:T) {
    nu[t] = phi[1] * y[t - 1] + phi[2] * y[t - 2] + theta * err[t - 1];
    err[t] = y[t] - nu[t];
  }
}
model {
  phi ~ normal(0, 2);
  theta ~ normal(0, 2);
  sigma ~ cauchy(0, .5);
  err ~ normal(0, sigma);    // likelihood
}
generated quantities {
  array[T] real y_rep = normal_rng(nu, sigma);
  vector[T] log_lik;

  for (i in 1:T) {
    log_lik[i] = normal_lpdf(y[i] | nu[i], sigma);
  }
}