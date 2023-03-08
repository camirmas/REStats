import pymc as pm

def fit_weibull(ws):
    with pm.Model() as model_wb:
        alpha = pm.Normal("alpha", 2, 2) # fairly informative priors
        sigma = pm.Normal("sigma", 10, 3)
        y = pm.Weibull("y", alpha, sigma, observed=ws.values)
        
        idata_wb = pm.sample()
        pm.sample_posterior_predictive(idata_wb, extend_inferencedata=True)
        pm.compute_log_likelihood(idata_wb)
    
    return idata_wb