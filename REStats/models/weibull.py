import pyro
import arviz as az
import numpy as np
import torch
import pyro.infer
import pyro.optim
import matplotlib.pyplot as plt
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS, Predictive
from scipy.stats import gamma, weibull_min


def weibull_model(data):
    """
    Define the Pyro model for fitting a Weibull distribution.

    Args:
        data (torch.Tensor): A tensor of data samples from a Weibull distribution.

    Returns:
        None
    """
    shape = pyro.sample("shape", dist.Gamma(7.5, 0.4))
    scale = pyro.sample("scale", dist.Gamma(10, 1))

    with pyro.plate("data", len(data)):
        pyro.sample("obs", dist.Weibull(scale, shape), obs=data)


def fit(ws):
    """
    Fit a Weibull distribution to wind speed data using Pyro and MCMC.

    Args:
        ws (List[float]): A list of wind speed data points.

    Returns:
        arviz.InferenceData: The fitted Weibull distribution in an `arviz.InferenceData`
            object.
    """
    ws_t = torch.tensor(ws)

    # Set up the MCMC sampler and run it
    nuts_kernel = NUTS(weibull_model)
    mcmc = MCMC(nuts_kernel, num_chains=2, num_samples=1000, warmup_steps=500)
    mcmc.run(ws_t)

    # Generate prior and posterior predictive distributions using `Predictive`
    prior_predictive = Predictive(weibull_model, num_samples=1000).forward(ws_t)
    posterior_predictive = Predictive(
        weibull_model, posterior_samples=mcmc.get_samples()
    ).forward(ws_t)

    # Convert the results to an `arviz.InferenceData` object
    idata = az.from_pyro(
        mcmc,
        prior=prior_predictive,
        posterior_predictive=posterior_predictive,
    )

    # Display results
    return idata


def get_params(idata_wb):
    """
    Extract the shape and scale parameters from the fitted Weibull distribution.

    Args:
        idata_wb (arviz.InferenceData): An `arviz.InferenceData` object containing the
            fitted Weibull distribution.

    Returns:
        Tuple[float, float]: A tuple containing the shape and scale parameters.
    """
    shape = idata_wb.posterior.shape.mean(["chain", "draw"]).item(0)
    scale = idata_wb.posterior.scale.mean(["chain", "draw"]).item(0)

    return shape, scale


def calc_m(shape):
    """
    Calculate the Weibull modulus (m) from the shape parameter.

    Args:
        shape (float): The shape parameter of a Weibull distribution.

    Returns:
        float: The Weibull modulus (m).
    """
    return shape / 3.6


def plot_prior_samples(shape_prior_params, scale_prior_params, num_samples):
    # draw samples from the gamma priors
    shape_samples = gamma.rvs(
        a=shape_prior_params[0], scale=shape_prior_params[1], size=num_samples
    )
    scale_samples = gamma.rvs(
        a=scale_prior_params[0], scale=scale_prior_params[1], size=num_samples
    )

    # create a figure
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xmargin(0)

    # for each pair of shape and scale samples, draw a sample from the
    # corresponding Weibull distribution
    for shape, scale in zip(shape_samples, scale_samples):
        x = np.linspace(0, 18, 1000)
        y = weibull_min.pdf(x, c=shape, scale=scale)
        ax.plot(x, y)  # use low alpha to see overlapping lines

    # set labels
    ax.set_xlabel("Wind speed (m/s)")
    ax.set_ylabel("Probability Density")
    ax.set_title("Weibull Prior Predictive Samples")

    return fig
