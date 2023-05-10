import pyro
import numpy as np
import torch
import pytest

# from pyro.infer import MCMC, NUTS, Predictive
from pyro.distributions import Weibull

from REStats.models.weibull import fit, calc_m, get_params, weibull_model


@pytest.fixture
def synthetic_data():
    torch.manual_seed(42)
    true_shape = 2.0
    true_scale = 5.0
    data = Weibull(true_shape, true_scale).sample((100,))
    return data.tolist()


def test_weibull_model(synthetic_data):
    data = torch.tensor(synthetic_data, dtype=torch.float32)
    trace = pyro.poutine.trace(weibull_model).get_trace(data)
    assert "shape" in trace
    assert "scale" in trace
    assert "obs" in trace


def test_fit(synthetic_data):
    idata = fit(synthetic_data)
    assert idata is not None
    assert hasattr(idata, "posterior")
    assert hasattr(idata, "prior")
    assert hasattr(idata, "posterior_predictive")


def test_get_params(synthetic_data):
    idata = fit(synthetic_data)
    shape, scale = get_params(idata)
    assert isinstance(shape, float)
    assert isinstance(scale, float)
    assert shape > 0
    assert scale > 0


def test_calc_m():
    shape = 2.0
    m = calc_m(shape)
    assert isinstance(m, float)
    assert np.isclose(m, shape / 3.6)
