import numpy as np


def zero_mean_garch_1_1_scenario(sim_std_resid, fitted_conditional_volatility, fitted_residuals,  w, alpha, beta):
    r_t = np.zeros(sim_std_resid.shape)
    sigma_2 = np.zeros(sim_std_resid.shape)
    eps_t = np.zeros(sim_std_resid.shape)

    for t in range(sim_std_resid.shape):
        if t == 0:
            sigma_2[t] = w + alpha * np.power(fitted_residuals[-1], 2) + \
                             beta * np.power(fitted_conditional_volatility[-1], 2)
            eps_t[t] = np.sqrt(sigma_2[t])
            r_t[t] = eps_t[t]
        else:
            sigma_2[t] = w + alpha * np.power(eps_t[t - 1], 2) + \
                             beta * np.power(sigma_2[t - 1], 2)
            eps_t[t] = np.sqrt(sigma_2[t])
            r_t[t] = eps_t[t]

        return r_t, sigma_2, eps_t
