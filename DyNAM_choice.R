# # Documentation ----
# # DyNAM_choice.R
# # R Versions: 4.2.0 x86
# #
# # Author(s): AU
# #
# #
# # Description: Code to perform Bayesian Inference on the
# #   DyNAM-choice sub-model

# installing packages -----------------------------------------------------
require(remotes)

remotes::install_github(
  repo = "snlab-ch/goldfish",
  ref = "feature_simulation",
  build_vignettes = FALSE
)

remotes::install_github(
  repo = "snlab-ch/goldfish.latent",
  build_vignettes = FALSE
)

# I'm using cmdstanr, but the code should work with stanr
# provide the stan version is not that old
install.packages(
  pkgs = "cmdstanr",
  repos = c("https://mc-stan.org/r-packages/", getOption("repos"))
)


# packages ----------------------------------------------------------------
library(goldfish)         # # 1.6.6 feature_simulation
library(goldfish.latent)  # # 0.0.1.9000
library(cmdstanr)         # # 0.5.2

# PSIS and plotting
library(loo)
library(bayesplot)
library(posterior)

# Stan needs to be compiled for cmdstanr
# I set a location for that, you choose
# (here is how mine looks like  to avoid admin issues)
pathStan <- "C:/Users/xxx/AppData/Local/Programs/.cmdstan"

# compiler optimisations flags, as far I'm aware only CXX14FLAGS is required
cpp_options <- list(
  "CXX14FLAGS=-O3 -mmmx -msse -msse2 -msse3 -mssse3 -msse4.1 -msse4.2"
)

# if you want to compile stan to use with cmdstanr, run the following lines
install_cmdstan(dir = pathStan, cores = 4L,
                cpp_options = cpp_options, overwrite = TRUE)

  # set path to cmdstan
set_cmdstan_path(file.path(pathStan, "cmdstan-2.29.2"))

# set where cmdstanr compile the code
if (!dir.exists("Stan")) dir.create("Stan")
options(
  cmdstanr_write_stan_file_dir = "Stan"
)

# mock data ---------------------------------------------------------------
data("Social_Evolution")
callNetwork <- defineNetwork(nodes = actors, directed = TRUE) |>
  linkEvents(changeEvent = calls, nodes = actors)

callsDependent <- defineDependentEvents(events = calls, nodes = actors,
                                        defaultNetwork = callNetwork)

friendshipNetwork <- defineNetwork(nodes = actors, directed = TRUE) |>
  linkEvents(changeEvents = friendship, nodes = actors)

# check missing data. It doesn't play well with Stan
actors |> lapply(\(x) table(is.na(x)))

# goldfish to stan -------------------------------------------------------

dataMod <- CreateData(list(inertia ~ 1),
  callsDependent ~ inertia + recip + trans + tie(friendshipNetwork) +
    recip(friendshipNetwork) + trans(friendshipNetwork)
)

str(dataMod$dataStan)



# model -------------------------------------------------------------------

modelWORE <- cmdstan_model("Stan/MCM_Modified.stan")

if (!dir.exists("output")) dir.create("output")

modWORESamples <- modelWORE$sample(
  data = dataMod$dataStan,
  parallel_chains = 4, chains = 4,  iter_warmup = 500, iter_sampling = 500,
  output_dir = "output"
)


modWORESamples$summary("beta")
modWORESamples$time()
system.time({summary(estimate(
  callsDependent ~ inertia + recip + trans + tie(friendshipNetwork) +
    recip(friendshipNetwork) + trans(friendshipNetwork)
))})


# convergence -------------------------------------------------------------
modWORESamples$cmdstan_diagnose()

betas <- modWORESamples$draws("beta")
mcmc_dens_chains(betas)
mcmc_trace(betas)
mcmc_rank_overlay(betas)

mcmc_intervals(betas)

# PSIS --------------------------------------------------------------------
loo(modWORESamples$draws("log_lik"))

looMod <- modWORESamples$loo()
loo::pareto_k_influence_values(looMod)
pareto_k_ids(looMod)
