from abc import ABC, abstractmethod


class RiskBudgetingAbstract(ABC):
    """A Risk Budgeting Abstract Class"""

    @classmethod
    @abstractmethod
    def risk_measure(cls, *args, **kwargs):
        pass

    @classmethod
    @abstractmethod
    def risk_contribution(cls, *args, **kwargs):
        pass

    @classmethod
    @abstractmethod
    def marginal_risk_contribution(cls, *args, **kwargs):
        pass

    @classmethod
    @abstractmethod
    def jacobian(cls, *args, **kwargs):
        pass

    @classmethod
    @abstractmethod
    def hessian(cls, *args, **kwargs):
        pass