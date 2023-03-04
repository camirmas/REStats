data {
  int<lower=1> T;            // num observations
  array[T] real y;           // observed outputs
  // int<lower=1> p = 2;
}
parameters {
  real mu;                   // mean coeff
  real phi;                  // autoregression coeff
  real theta;                // moving avg coeff
  real<lower=0> sigma;       // noise scale
}
transformed parameters {
  vector[T] nu;              // prediction for time t
  vector[T] err;             // error for time t
  nu[1] = mu + phi * mu;    // assume err[0] == 0
  err[1] = y[1] - nu[1];
  for (t in 2:T) {
    nu[t] = phi * y[t - 1] + theta * err[t - 1];
    err[t] = y[t] - nu[t];
  }
}
model {
  mu ~ normal(0, 10);         // priors
  phi ~ normal(0, 2);
  theta ~ normal(0, 2);
  sigma ~ cauchy(0, 5);
  err ~ normal(0, sigma);    // likelihood
}
generated quantities {
  array[T] real y_rep = normal_rng(nu, sigma);
  vector[T] log_lik;

  for (i in 1:T) {
    log_lik[i] = normal_lpdf(y[i] | nu[i], sigma);
  }
}