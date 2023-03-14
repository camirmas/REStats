import pymc as pm

def fit_weibull(ws):
    with pm.Model() as model_wb:
        alpha = pm.HalfNormal("alpha", 2)
        beta = pm.HalfNormal("beta", 2)
        y = pm.Weibull("y", alpha, beta, observed=ws.values)
        
        idata_wb = pm.sample()
        pm.sample_posterior_predictive(idata_wb, extend_inferencedata=True)
        pm.compute_log_likelihood(idata_wb)
    
    return idata_wb