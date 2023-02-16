import numpy as np
from .AbstractClass import RiskBudgetingAbstract


class PortfolioVolatility(RiskBudgetingAbstract):
    @classmethod
    def risk_measure(cls, x: np.array, sigma: np.ndarray, squeeze: bool = True, *args, **kwargs):
        """
        Gaussian VolRisk Measure

        :param x: vector of weights
        :type x: numpy.array
        :param sigma: Covariance Matrix
        :type sigma: numpy.ndarray (dim=2)
        :param squeeze: if true, squeezes the return into a numpy array. Else, returs a numpy ndarray.
        :type squeeze: bool
        :return: Risk Contribution
        :rtype: numpy.array
        """
        return np.sqrt(x @ sigma @ np.expand_dims(x, axis=1))

    @classmethod
    def risk_contribution(cls, x: np.array, sigma: np.ndarray, squeeze: bool = True, *args, **kwargs):
        """
        Gaussian Risk Contribution

        :param x: vector of weights
        :type x: numpy.array
        :param sigma: Covariance Matrix
        :type sigma: numpy.ndarray (dim=2)
        :param squeeze: if true, squeezes the return into a numpy array. Else, returs a numpy ndarray.
        :type squeeze: bool
        :return: Risk Contribution
        :rtype: numpy.array
        """

        return cls.marginal_risk_contribution(x, sigma=sigma) * x

    @classmethod
    def marginal_risk_contribution(cls, x: np.array, sigma: np.ndarray, squeeze: bool = True, *args, **kwargs):
        """
        Gaussian Marginal Risk Contribution

        :param x: vector of weights
        :type x: numpy.array
        :param sigma: Covariance Matrix
        :type sigma: numpy.ndarray (dim=2)
        :param squeeze: if true, squeezes the return into a numpy array. Else, returs a numpy ndarray.
        :type squeeze: bool
        :return: Marginal Risk Contribution
        :rtype: numpy.array
        """
        x2dim = np.expand_dims(x, axis=1)
        out2dim = 0.5 * np.power(x2dim.transpose() @ sigma @ x2dim, -0.5) * (x2dim.transpose() @ sigma.transpose() +
                                                                             x2dim.transpose() @ sigma)
        if squeeze:
            return np.squeeze(out2dim, axis=0)
        else:
            return out2dim

    @classmethod
    def jacobian(cls, x: np.array, sigma: np.ndarray, *args, **kwargs):
        return cls.marginal_risk_contribution(x, sigma, False)

    @classmethod
    def hessian(cls, x: np.array, sigma: np.ndarray, *args, **kwargs):

        x2dim = np.expand_dims(x, axis=1)

        v1 = - 0.25 * np.power(x2dim.transpose() @ sigma @ x2dim, -1.5)
        v2 = x2dim.transpose() @ sigma.transpose() + x2dim.transpose() @ sigma
        v3 = 0.5 * np.power(x2dim.transpose() @ sigma @ x2dim, -0.5) * (sigma + sigma.transpose())

        return v1 * v2.transpose() @ v2 + v3
