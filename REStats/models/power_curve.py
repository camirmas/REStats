import torch
import gpytorch
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.means import ConstantMean
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.distributions import MultivariateNormal


class ExactGPModel(gpytorch.models.ExactGP):
    """
    The simplest form of GP model that uses exact inference.

    Args:
        train_x (torch.Tensor): Training input data.
        train_y (torch.Tensor): Training output data.
        likelihood (gpytorch.likelihoods.Likelihood): Likelihood function.
    """
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(RBFKernel())

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            gpytorch.distributions.MultivariateNormal: Multivariate normal distribution.
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


def fit(X_train, y_train):
    """
    Fits the GP model to the training data.

    Args:
        X_train (torch.Tensor): Training input data.
        y_train (torch.Tensor): Training output data.

    Returns:
        tuple: Tuple of the trained model and the likelihood function.
    """
    likelihood = GaussianLikelihood()
    model = ExactGPModel(X_train, y_train, likelihood)

    training_iter = 100

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = ExactMarginalLogLikelihood(likelihood, model)

    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(X_train)
        # Calc loss and backprop gradients
        loss = -mll(output, y_train)
        loss.backward()
        print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
            i + 1, training_iter, loss.item(),
            model.covar_module.base_kernel.lengthscale.item(),
            model.likelihood.noise.item()
        ))
        optimizer.step()

    return model, likelihood


def predict(model, likelihood, data):
    """
    Generates predictions for the input data.

    Args:
        model (ExactGPModel): Trained GP model.
        likelihood (gpytorch.likelihoods.Likelihood): Likelihood function.
        data (torch.Tensor): Input data for which predictions are to be made.

    Returns:
        gpytorch.distributions.MultivariateNormal: Multivariate normal distribution.
    """
    # Get into evaluation (predictive posterior) mode
    model.eval()
    likelihood.eval()

    # Test points are regularly spaced along [0,1]
    # Make predictions by feeding model through likelihood
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = likelihood(model(data))
    
    return pred
