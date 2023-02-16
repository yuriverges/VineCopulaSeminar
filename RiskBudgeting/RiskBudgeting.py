import RiskBudgeting.resources as resources


class RiskBudgeting:
    def __new__(cls, risk_measure_type=None, *args, **kwargs):
        if risk_measure_type is None:
            risk_measure_type = 'PortfolioVolatility'

        try:
            obj = getattr(resources, risk_measure_type)()
        except:
            ValueError(f"Risk Measure {risk_measure_type} not recognized. Verify")

        return obj
