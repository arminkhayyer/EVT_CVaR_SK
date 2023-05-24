import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import math
from smt.sampling_methods import LHS
import sys
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from scipy.stats import norm
import os
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import math
import matplotlib.tri as tri
# from thresholdmodeling import thresh_modeling #importing package
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go

from rpy2.robjects.packages import importr
import rpy2.robjects.packages as rpackages

base = importr('base')
utils = importr('utils')
utils.chooseCRANmirror(ind=1)
utils.install_packages('POT') #installing POT package

from thresholdmodeling import thresh_modeling #importing package

desired_width=320

pd.set_option('display.width', desired_width)

np.set_printoptions(linewidth=desired_width)

pd.set_option('display.max_columns',10)


limits = [[-np.pi, np.pi], [-np.pi, np.pi]]
xlimits = np.array(limits)
sampling = LHS(xlimits=xlimits, criterion="ese", random_state= 41)
# number of desing points


x = sampling(40)
x = np.array(x)
df = pd.DataFrame(x, columns=["x1", "x2"])
# df["emperical_var"] = 0
# df["estimated_var"] = 0

plt.scatter(df.x1, df.x2)
plt.grid(linewidth=0.1)
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.xlabel("x")
plt.ylabel("y")
plt.title("latin hyper cube")
plt.show()
new_list = []

for index, row in df.iterrows():
    cvar = []
    cvar_emperical = []
    var_cvar = []
    alpha = 0.99
    thresh = .9
    for run in range(1, 20+ 1):
        data = []
        for reps in range(1, 1000 + 1):
            mu = row["x1"] + row["x2"]
            sigma = math.sqrt(row["x1"] ** 2 + row["x2"] ** 2)
            temp = (row["x1"] * math.sin(np.pi * row["x2"])) + (row["x2"] * math.sin(np.pi * row["x1"]))
            noise = np.random.normal(0, sigma + 1)
            temp = noise + temp
            data.append(temp)
            # inside_list = [index, row["x1"], row["x2"], run, reps, temp]
            # new_list.append(inside_list)
        threshold = np.quantile(data, thresh)
        print(threshold)
        threshold = float(threshold)

        mle = thresh_modeling.gpdfit(data, threshold, 'mle')
        N_u = len(mle[4])
        excedence = np.array(mle[4])
        N = len(data)
        scale = mle[1]
        shape = mle[0]

        value_at_risk = threshold + scale * (((N_u / (N * (1 - alpha))) ** shape - 1) / shape)
        Conditional_var = (value_at_risk + scale - (shape * threshold)) / (1 - shape)
        cvar.append(Conditional_var)

        gradients_of_h = np.zeros((2,))
        gradients_of_h[0] = (value_at_risk + scale - threshold) / ((1 - shape) ** 2)
        gradients_of_h[1] = 1 / (1 - shape)

        I_11 = 0
        I_12 = 0
        I_22 = 0
        for observation in excedence:
            z = observation - threshold
            I_11 += ((-(2 / (shape ** 3)) * math.log(1 + ((z * shape) / scale))) + ((2 / (shape ** 2)) * (z / (scale + shape * z))) +
                     ((1 / shape + 1) * ((z ** 2) / ((scale + shape * z) ** 2))))
            I_12 += (1 / scale * ((z / (scale + shape * z)) - (((z ** 2) * (shape + 1)) / ((scale + shape * z) ** 2))))
            I_22 += (((-1 / (scale ** 2)) * (((z * (shape + 1)) / (scale + shape * z)) - 1)) - (
                        (z * (shape + 1)) / (scale * ((scale + shape * z) ** 2))))
        I_11 = (-1 / N_u) * (I_11)
        I_12 = (-1 / N_u) * (I_12)
        I_22 = (-1 / N_u) * (I_22)

        fisher = np.array([[I_11, I_12], [I_12, I_22]])
        fisher_inverse = np.linalg.inv(fisher)
        var = 1 / N_u * (gradients_of_h.dot(fisher_inverse).dot(gradients_of_h))
        var_cvar.append(var)
        N_over_value_at_risk = np.count_nonzero(np.where(np.array(excedence)>value_at_risk))
        var_samp_est = np.var(excedence[excedence>value_at_risk])
        cvar_emp = np.average(excedence[excedence>value_at_risk])
        cvar_emperical.append(cvar_emp)

        # thresh_modeling.gpdpdf(data, thresh, 'mle', 'sturges', 0.05)
        # thresh_modeling.gpdcdf(data, thresh, 'mle', 0.05)
        # thresh_modeling.qqplot(data,thresh, 'mle', 0.05)
        # thresh_modeling.ppplot(data, thresh, 'mle', 0.05)
    df.at[index, "emperical_var"] = np.var(cvar)
    df.at[index, "estimated_var"] = var_cvar[0]
    df.at[index, "sample_estimate"] = scale / N_over_value_at_risk
    df.at[index, "emp_var_cvar"] = np.average(cvar_emperical)
    df.at[index, "sample_var_emp"] = np.average(cvar_emperical)

    df.at[index, "Emp_cvar"] = np.average(cvar)
    df.at[index, "POT_cvar"] = cvar[-1]

    print(index,np.var(cvar), "emperical")
    print(var_cvar, "estimate")
print(df)

app = dash.Dash(__name__)

# Create traces
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df.emperical_var,
                    mode='lines+markers',
                    name='Emp-EST'))
fig.add_trace(go.Scatter(x=df.index, y=df.estimated_var,
                    mode='lines+markers',
                    name='EVT-EST'))
fig.add_trace(go.Scatter(x=df.index, y=df.sample_estimate,
                    mode='lines+markers', name='Sample-EST'))

fig.update_layout(
    title="Estimated Variance of CVaR, Run = 20, Replication = 1000",
    xaxis_title="Design Point",
    yaxis_title="Variance",
    legend_title="Method",
)

fig.show()