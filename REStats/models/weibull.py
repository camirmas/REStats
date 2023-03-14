import pymc as pm


def fit_weibull(ws):
    with pm.Model():
        alpha = pm.HalfNormal("alpha", 1)
        beta = pm.HalfNormal("beta", 1)
        y_ = pm.Weibull("y", alpha, beta, observed=ws.values)
        
        idata_wb = pm.sample(random_seed=100)
        pm.sample_posterior_predictive(idata_wb, extend_inferencedata=True, random_seed=100)
        pm.compute_log_likelihood(idata_wb)
    
    return idata_wb


def get_params(idata_wb):
    shape = idata_wb.posterior.alpha.mean(["chain", "draw"]).item(0)
    scale = idata_wb.posterior.beta.mean(["chain", "draw"]).item(0)

    return shape, scale


def calc_m(shape):
    return shape / 3.6