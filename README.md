# EVT_CVaR_SK


We are considering the problem of evaluating risk for a system that is modeled by a complex stochastic
simulation with many possible input parameter values. Naturally, two sources of computational
burden can be identified: the effort associated with extensive simulation runs required to accurately
represent the tail of the loss distribution for each set of parameter values, and the computational cost
of evaluating multiple candidate parameter values. The former concern can be addressed by using
Extreme Value Theory (EVT) estimations, which specifically concentrate on the tails. Meta-modeling
approaches are often used to tackle the latter concern. In this paper, we propose a framework for
constructing a particular meta-modeling framework, Stochastic Kriging, that is based on EVT-based
estimation of a popular measure of risk – Conditional-Value-at-Risk – used as the underlying signal
of interest. Specifically, combining the two requires an efficient estimator of the intrinsic variance, for
which we derive an EVT-based expression, which allows us to avoid multiple replications of CVaR
in each design point, which was required in previously proposed approaches. We then perform a
case study, outlining promising use cases, and conditions when the EVT-based approach outperforms
simpler empirical estimators.
