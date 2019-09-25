import numpy as np
import scipy as sp
from scipy import integrate
import warnings

def optimize_qstar_sigmaw_sigmab(L):
    warnings.simplefilter("ignore")
    tanh = np.tanh
    sech = lambda x: 1.0 / np.cosh(x)
    g = lambda h, q: np.sqrt(1 / (2 * np.pi)) * np.exp(-(h ** 2) / 2) * ((sech(np.sqrt(q) * h)) ** 2) ** 2

    c = (L - 1) * (L / (L - 1)) ** L
    C = c / (1 + c)

    def objective(QQ):
        DPhi, _ = sp.integrate.quad(g, -np.inf, np.inf, args=(QQ))
        return 0.5 * (DPhi - C) ** 2

    res = sp.optimize.minimize_scalar(objective, args=(), method='bounded', bounds=[0, 3], tol=None,
                                      options={'maxiter': 1000})
    Q = res.x
    sigma_w, sigma_b = optimize_sigmaw_sigmab(Q)
    return Q, sigma_w, sigma_b

def optimize_sigmaw_sigmab(Q):
    warnings.simplefilter("ignore")
    tanh = np.tanh
    sech = lambda x: 1.0 / np.cosh(x)
    g = lambda h, q: np.sqrt(1 / (2 * np.pi)) * np.exp(-(h ** 2) / 2) * ((sech(np.sqrt(q) * h)) ** 2) ** 2
    f = lambda h, q: np.sqrt(1 / (2 * np.pi)) * np.exp(-(h ** 2) / 2) * (tanh(np.sqrt(q) * h)) ** 2
    gamma, abserr2 = integrate.quad(f, -np.inf, np.inf, args=(Q))
    GAMMA, abserr2 = integrate.quad(g, -np.inf, np.inf, args=(Q))
    sigma_w = GAMMA ** -.5
    sigma_b = np.sqrt(Q - sigma_w ** 2 * gamma)

    return sigma_w, sigma_b

def get_glm_std(Q, qglm):
    warnings.simplefilter("ignore")
    tanh = np.tanh
    f = lambda h, q: np.sqrt(1 / (2 * np.pi)) * np.exp(-(h ** 2) / 2) * (tanh(np.sqrt(q) * h)) ** 2
    gamma, abserr2 = integrate.quad(f, -np.inf, np.inf, args=(Q))
    sigma_glm =  np.sqrt( qglm/gamma)
    return sigma_glm