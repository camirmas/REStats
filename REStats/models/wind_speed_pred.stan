data {
  int<lower=2> T;            // num observations
  array[T] real y;           // observed outputs
  int<lower=0> n_pred;
}
transformed data {
  int<lower=0> X = T + n_pred;
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
generated quantities {
  vector[X] nu_predict;
  vector[X] predict;
  vector[X] err_predict;

  for (t in 1:X) {
    if (t <= T) {
      predict[t] = normal_rng(nu[t], sigma);
      nu_predict[t] = nu[t];
      err_predict[t] = err[t];
    } else {
      nu_predict[t] = phi[1] * predict[t - 1] + phi[2] * predict[t - 2] + theta * err_predict[t - 1];
      predict[t] = normal_rng(nu_predict[t], sigma);
      err_predict[t] = predict[t] - nu_predict[t];
    }
  }
}