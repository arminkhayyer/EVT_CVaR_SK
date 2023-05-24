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
from thresholdmodeling import thresh_modeling #importing package
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go



desired_width=320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns',10)


limits = [[-np.pi, np.pi], [-np.pi, np.pi]]
xlimits = np.array(limits)
# number of desing points


dict_design = {1:[50, 10, 200],2:[50, 5, 400],3:[50, 1, 2000], 4:[100,5,200],5:[100, 1,1000],
               6:[50, 20, 1000], 7:[50, 10, 2000],8:[100, 10, 1000], 9:[100,5,2000],10:[100, 1,10000],
               11:[50, 100, 2000], 12:[100, 50, 2000], 13:[100, 10, 10000], 14:[100, 5, 20000],15:[100, 1, 100000]}


def True_cvar(x1, x2, alpha):
    mu = (x1 * math.sin(np.pi * x2)) + (x2 * math.sin(np.pi * x1))
    sigma = math.sqrt((x1 ** 2) + (x2 ** 2))
    true_val = mu + ((2 + sigma) * 2) / math.sqrt(1-alpha)
    return true_val


alphas = [0.95, 0.99, 0.995]
thresh = .9
# sampling = LHS(xlimits=xlimits, criterion="ese", random_state=12)
# x = sampling(50)
for alpha in alphas:
    for tot_rep in range(10):
        print(tot_rep)
        for key , value in dict_design.items():
            sampling = LHS(xlimits=xlimits, criterion="ese", random_state=41)
            x = sampling(value[0])
            x = np.array(x)
            df = pd.DataFrame(x, columns=["x1", "x2"])

            # plt.scatter(df.x1, df.x2)
            # plt.grid(linewidth=0.1)
            # plt.minorticks_on()
            # plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
            # plt.xlabel("x")
            # plt.ylabel("y")
            # plt.title("latin hyper cube")
            # plt.show()
            new_list = []

            for index, row in df.iterrows():
                cvar = []
                cvar_emperical = []
                var_cvar = []
                var_CVAR_chen = []
                true_CVaR = True_cvar(row["x1"], row["x2"], alpha)
                for run in range(value[1]):
                    data = []
                    mu = row["x1"] + row["x2"]
                    sigma = math.sqrt(row["x1"] ** 2 + row["x2"] ** 2)
                    deterministic = (row["x1"] * math.sin(np.pi * row["x2"])) + (row["x2"] * math.sin(np.pi * row["x1"]))
                    for reps in range(1, value[2] +1):
                        noise = (np.random.pareto(2)+ 1) * (sigma + 2)
                        temp = noise + deterministic
                        data.append(temp)
                    threshold = np.quantile(data, thresh)
                    threshold = float(threshold)
                    mle = thresh_modeling.gpdfit(data, threshold, 'mle')
                    N_u = len(mle[4])
                    excedence = np.array(mle[4])

                    N = len(data)
                    scale = mle[1]
                    shape = mle[0]
                    value_at_risk = threshold + scale * (((N_u / (N * (1 - alpha))) ** shape - 1) / shape)
                    var_emp = np.quantile(data, alpha)
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
                    N_over_value_at_risk = np.count_nonzero(np.where(np.array(excedence)>=var_emp))
                    var_samp_est = np.var(excedence[excedence>value_at_risk])
                    cvar_emp = np.average(excedence[excedence>=var_emp])
                    cvar_emperical.append(cvar_emp)

                    sumation = 0
                    for i in data:
                        wi = var_emp + ((1 / (1 - alpha)) * max(i - var_emp, 0))
                        sumation += (wi - cvar_emp) ** 2
                    estimated_var = (1 / (N * (N - 1))) * sumation
                    var_CVAR_chen.append(estimated_var)

                    # thresh_modeling.gpdpdf(data, thresh, 'mle', 'sturges', 0.05)
                    # thresh_modeling.gpdcdf(data, thresh, 'mle', 0.05)
                    # thresh_modeling.qqplot(data,thresh, 'mle', 0.05)
                    # thresh_modeling.ppplot(data, thresh, 'mle', 0.05)
                df.at[index, "Emperical_variance_of_cvar"] = np.var(cvar)
                df.at[index, "POT_conditional_va_risk"] = np.average(cvar)
                df.at[index, "Estimated_variance_of_cvar"] = np.average(var_cvar)
                df.at[index, "Estimated_variance_of_cvar_EMP_CHEN"] = np.average(var_CVAR_chen)
                df.at[index, "Sample_Estimate_CLT_scale"] = scale / N_over_value_at_risk
                df.at[index, "Sample_varinave_Estimate_CLT"] = var_samp_est / N_over_value_at_risk
                df.at[index, "Emp_variance_of_EMP_cvar"] = np.var(cvar_emperical)
                df.at[index, "EMP_conditional_vaRisk"] = np.average(cvar_emperical)
                df.at[index, "True_cvar"] = true_CVaR
                # df_evt = df
                # for rep in range( value[1]):
                #     col_pot = "POT_CVAR" + str(rep)
                #     col_var = "var"+ str(rep)
                #     df_evt.at[index, col_pot] = cvar[rep]
                #     df_evt.at[index, col_var] = var_cvar[rep]
                #
            outdir = "pareto" + str(alpha) + "_"+ str(tot_rep)
            if not os.path.exists(outdir):
                os.mkdir(outdir)

            filename= "pareto{}.csv".format(key)
            fullname = os.path.join(outdir, filename)
            df.to_csv(fullname)



    sampling = LHS(xlimits=xlimits, criterion="ese", random_state=11)
    x = sampling(100)
    x = np.array(x)
    df = pd.DataFrame(x, columns=["x1", "x2"])
    for index, row in df.iterrows():
        cvar = []
        cvar_emperical = []
        var_cvar = []
        var_CVAR_chen = []
        true_CVaR = True_cvar(row["x1"], row["x2"], alpha)
        for run in range(value[1]):
            data = []
            mu = row["x1"] + row["x2"]
            sigma = math.sqrt(row["x1"] ** 2 + row["x2"] ** 2)
            deterministic = (row["x1"] * math.sin(np.pi * row["x2"])) + (row["x2"] * math.sin(np.pi * row["x1"]))
            for reps in range(1, value[2] +1):
                noise = (np.random.pareto(2)+ 1) * (sigma + 2)
                temp = noise + deterministic
                data.append(temp)
            threshold = np.quantile(data, thresh)
            threshold = float(threshold)
            mle = thresh_modeling.gpdfit(data, threshold, 'mle')
            N_u = len(mle[4])
            excedence = np.array(mle[4])

            N = len(data)
            scale = mle[1]
            shape = mle[0]
            value_at_risk = threshold + scale * (((N_u / (N * (1 - alpha))) ** shape - 1) / shape)
            var_emp = np.quantile(data, alpha)
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
            N_over_value_at_risk = np.count_nonzero(np.where(np.array(excedence)>var_emp))
            var_samp_est = np.var(excedence[excedence>value_at_risk])
            cvar_emp = np.average(excedence[excedence>= var_emp])
            cvar_emperical.append(cvar_emp)

            sumation = 0
            for i in data:
                wi = var_emp + ((1 / (1 - alpha)) * max(i - var_emp, 0))
                sumation += (wi - cvar_emp) ** 2
            estimated_var = (1 / (N * (N - 1))) * sumation
            var_CVAR_chen.append(estimated_var)

            # thresh_modeling.gpdpdf(data, thresh, 'mle', 'sturges', 0.05)
            # thresh_modeling.gpdcdf(data, thresh, 'mle', 0.05)
            # thresh_modeling.qqplot(data,thresh, 'mle', 0.05)
            # thresh_modeling.ppplot(data, thresh, 'mle', 0.05)
        df.at[index, "Emperical_variance_of_cvar"] = np.var(cvar)
        df.at[index, "POT_conditional_va_risk"] = np.average(cvar)
        df.at[index, "Estimated_variance_of_cvar"] = np.average(var_cvar)
        df.at[index, "Estimated_variance_of_cvar_EMP_CHEN"] = np.average(var_CVAR_chen)
        df.at[index, "Sample_Estimate_CLT_scale"] = scale / N_over_value_at_risk
        df.at[index, "Sample_varinave_Estimate_CLT"] = var_samp_est / N_over_value_at_risk
        df.at[index, "Emp_variance_of_EMP_cvar"] = np.var(cvar_emperical)
        df.at[index, "EMP_conditional_vaRisk"] = np.average(cvar_emperical)
        df.at[index, "True_cvar"] = true_CVaR

    outdir = "pareto_test" + str(alpha)
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    filename = "pareto.csv"
    fullname = os.path.join(outdir, filename)
    df.to_csv(fullname)
