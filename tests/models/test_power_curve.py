import torch
import pytest
from REStats.models.power_curve import ExactGPModel, fit, predict
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.distributions import MultivariateNormal


@pytest.fixture
def dummy_data():
    train_x = torch.linspace(0, 1, 100)
    train_y = torch.sin(train_x * (2 * 3.1416)) + torch.randn(train_x.size()) * 0.2
    return train_x, train_y


def test_exact_gp_model(dummy_data):
    train_x, train_y = dummy_data
    likelihood = GaussianLikelihood()
    model = ExactGPModel(train_x, train_y, likelihood)
    assert isinstance(model, ExactGPModel)


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
