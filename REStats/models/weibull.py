import arviz as az
import pyro
import pyro.distributions as dist
import pyro.infer
import pyro.optim
import torch
from pyro.infer import MCMC, NUTS, Predictive
import matplotlib.pyplot as plt
import numpy as np


def weibull_model(data):
    """
    Define the Pyro model for fitting a Weibull distribution.

    Args:
        data (torch.Tensor): A tensor of data samples from a Weibull distribution.

    Returns:
        None
    """
    shape = pyro.sample("shape", dist.Gamma(2, 0.5))
    scale = pyro.sample("scale", dist.Gamma(1, 0.5))

    with pyro.plate("data", len(data)):
        pyro.sample("obs", dist.Weibull(scale, shape), obs=data)


def fit_weibull(ws):
    """
    Fits a Weibull distribution to wind speed data using Pyro and MCMC.

    Args:
        ws (List[float]): A list of wind speed data points.

    Returns:
        arviz.InferenceData: The fitted Weibull distribution in an `arviz.InferenceData` object.
    """
    ws_t = torch.tensor(ws)

    # Set up the MCMC sampler and run it
    nuts_kernel = NUTS(weibull_model)
    mcmc = MCMC(nuts_kernel, num_chains=2, num_samples=1000, warmup_steps=500)
    mcmc.run(ws_t)

    # Generate prior and posterior predictive distributions using `Predictive`
    prior_predictive = Predictive(weibull_model, num_samples=1000).forward(ws_t)
    posterior_predictive = Predictive(weibull_model, posterior_samples=mcmc.get_samples()).forward(ws_t)

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
        idata_wb (arviz.InferenceData): An `arviz.InferenceData` object containing the fitted Weibull distribution.

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


def plot_prior_samples(idata_wb):
    fig, ax = plt.subplots()
    ax.set_title("Weibull prior distributions")

    shapes = idata_wb.prior.shape[0, :10]
    scales = idata_wb.prior.scale[0, :10]

    x = np.linspace(0, 5, 500)

    def weib(x, scale, shape):
        return (shape / scale) * (x / scale)**(shape - 1) * np.exp(-(x / scale)**shape)

    for shape, scale in np.array([shapes, scales]).T:
        ax.plot(x, scale * weib(x, scale, shape))
        
    return fig