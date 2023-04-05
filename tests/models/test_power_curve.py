import torch
import pytest
import gpytorch
from gpytorch.distributions import MultivariateNormal

from REStats.models.power_curve import ExactGPModel, fit, predict
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.distributions import MultivariateNormal


@pytest.fixture
def dummy_data():
    train_x = torch.linspace(0, 1, 100)
    train_y = torch.sin(train_x * (2 * 3.1416)) + torch.randn(train_x.size()) * 0.2
    return train_x, train_y



def test_exact_gp_model_init():
    train_x = torch.randn(10)
    train_y = torch.randn(10)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    
    model = ExactGPModel(train_x, train_y, likelihood)
    
    assert isinstance(model.mean_module, gpytorch.means.ConstantMean)
    assert isinstance(model.covar_module, gpytorch.kernels.ScaleKernel)
    assert isinstance(model.covar_module.base_kernel, gpytorch.kernels.RBFKernel)


def test_exact_gp_model_forward():
    train_x = torch.randn(10)
    train_y = torch.randn(10)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(train_x, train_y, likelihood)

    # Set the model to evaluation mode
    model.eval()

    x = torch.randn(5)
    output = model(x)

    assert isinstance(output, MultivariateNormal)
    assert output.mean.shape == (5,)
    assert output.covariance_matrix.shape == (5, 5)


def test_prior_predictive_samples():
    train_x = torch.randn(10)
    train_y = torch.randn(10)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(train_x, train_y, likelihood)

    x = torch.linspace(-5, 5, 100)
    samples = model.prior_predictive_samples(x, n_samples=5)

    assert samples.shape == (5, 100)


def test_fit(dummy_data):
    train_x, train_y = dummy_data
    model, likelihood = fit(train_x, train_y)
    assert isinstance(model, ExactGPModel)
    assert isinstance(likelihood, GaussianLikelihood)


def test_predict(dummy_data):
    train_x, train_y = dummy_data
    model, likelihood = fit(train_x, train_y)
    test_x = torch.linspace(0, 1, 51)
    pred = predict(model, likelihood, test_x)
    assert isinstance(pred, MultivariateNormal)
    assert pred.mean.shape == test_x.shape
    assert pred.covariance_matrix.shape == (len(test_x), len(test_x))
