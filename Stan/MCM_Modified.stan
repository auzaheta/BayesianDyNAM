// MCM_modified.stan
// Multinomial choice model with P covariates
// Update to avoid warnings from deprecated syntax
// trying changing log_sum_exp in transformed quantities for speed up
//   from mc-stan discourse: speeding up a hierarchical negative binomial
//     regression with a huge number of groups
//
data {
  int N; // number of events * choice set (actors - 1)
  int T; // number of events
  int P; // number of covariates

  array[T] int<lower = 1, upper = N> chose; // position receiver chose
  matrix[N, P] X;

  array[T] int<lower = 1, upper = N> start; // the starting observation for each event
  array[T] int<lower = 1, upper = N> end; // the ending observation for each event
}
parameters {
  vector[P] beta;
}
model {
  // priors
  target += normal_lpdf(beta | 0, 1.5);

  // helper for likelihood
  {
    vector[N] xb;
    xb = X * beta;

    for(t in 1:T) target += xb[chose[t]] - log_sum_exp(xb[start[t]:end[t]]);
  }
}
generated quantities {
  vector[T] log_lik;

  {
    vector[N] xb;
    xb = X * beta;

    for(t in 1:T) log_lik[t] = xb[chose[t]] - log_sum_exp(xb[start[t]:end[t]]);
  }
}
