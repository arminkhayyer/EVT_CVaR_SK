import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import seaborn as sns
import os
from scipy.stats import norm
from smt.surrogate_models import KPLS, KRG
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF, ConstantKernel as C


files = ["pareto{}.csv".format(i) for i in range(1, 16)]
vars = [ "Estimated_variance_of_cvar", "Emperical_variance_of_cvar", "Estimated_variance_of_cvar_EMP_CHEN"]


#Once the design points are found by running the last cell, each design point is enterd into LCOM and desirable
#statistics are collected for each design point, in this example we only collect A3 statistc. The following lines of
#in this cell grab a dataframe from a csv file. The final dataframe has 6 columns, the first 4 represent the parts
#and the last two columns are the mean and variance of A3 statistic for 30 replications
alphas = [.95, .99, .995]
for alpha in alphas:
    for tot_rep in range(10):
        for var in vars:

            y_lable = "POT_conditional_va_risk"
            if var == "Estimated_variance_of_cvar_EMP_CHEN":
                y_lable = "EMP_conditional_vaRisk"

            res = pd.DataFrame(columns=[])

            dict_design = {1: [50, 10, 200], 2: [50, 5, 400], 3: [50, 1, 2000], 4: [100, 5, 200], 5: [100, 1, 1000],
                           6: [50, 20, 1000], 7: [50, 10, 2000], 8: [100, 10, 1000], 9: [100, 5, 2000], 10: [100, 1, 10000],
                           11: [50, 100, 2000], 12: [100, 50, 2000], 13: [100, 10, 10000], 14: [100, 5, 20000],
                           15: [100, 1, 100000]}

            plot_results = pd.DataFrame()

            for i in files:
                try:
                    print(i)
                    num = i.strip("pareto")
                    num = num.strip(".csv")
                    ind = int(num)
                    num_rep = dict_design[ind][1]
                    num_identifier = ind
                    print(tot_rep,i, ind, var)
                    filename ="pareto" +str(alpha) + "_"+ str(tot_rep) + "/"+ i
                    df = pd.read_csv(filename)
                    df = df.drop(["Unnamed: 0"], 1)

                    filename_test = "pareto_test"+ str(alpha) + "/pareto.csv"
                    df_test = pd.read_csv(filename_test)
                    df_test = df_test.drop(["Unnamed: 0"], 1)
                    df_test = df_test.sort_values(by = ["True_cvar"])

                    plot_results["x1"] = df_test["x1"]
                    plot_results["x2"] = df_test["x2"]
                    plot_results[str(files.index(i)) + "true"] = df_test["True_cvar"]


                    #in this cell we preproce.ss the data and split it into two sets, train and test set. The test set is used for
                    #validation of the fited stochastic kriging model.

                    x_data = (np.array(df[["x1", "x2"]]))
                    x_data_test = (np.array(df_test[["x1", "x2"]]))
                    scaler = MinMaxScaler(feature_range=(0, 1))
                    x_data = scaler.fit_transform(x_data)
                    x_data_test = scaler.transform(x_data_test)
                    y_data = (np.array(df[y_lable]))
                    y_data_test = (np.array(df_test['True_cvar']))
                    y_true = (np.array(df_test['True_cvar']))
                    y_emp = (np.array(df_test['EMP_conditional_vaRisk']))

                    intrinsic_var = np.array(df[var])/num_rep

                    # x_train , x_test, y_train, y_test, intrinsic_var_train, intrinsic_var_test, y_true_train, y_true_test= train_test_split(x_data, y_data, intrinsic_var,y_true,  test_size=.2, random_state=42 )

                    #this cell creates a diagnal matrix of the variance measures of each design point. this matrix
                    # is then used as the intrinsic covariance matrix.
                    intrinsic_cov =  np.diag(intrinsic_var)
                    x_train = x_data
                    y_train = y_data
                    x_test = x_data_test
                    y_test = y_data_test

                    sm = KRG(theta0=[1], print_prediction=False)
                    sm.set_training_values(x_train, y_train)
                    sm.train()
                    y_pred = sm.predict_values(x_test)
                    y_pred = y_pred.flatten()
                    def mean_absolute_percentage_error(y_true, y_pred):
                        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
                    MAPE_KRG = mean_absolute_percentage_error(y_test, y_pred)
                    MAE_KRG = mean_absolute_error(y_test, y_pred)
                    print("error ", np.linalg.norm(y_pred - y_test) / np.linalg.norm(y_test) * 100)
                    print("MAPE", MAPE_KRG)
                    print("MAE",MAE_KRG )
                    print("R2:", r2_score(y_test, y_pred))
                    print("MSE", mean_squared_error(y_test, y_pred))


                    print("MAPE_true", mean_absolute_percentage_error(y_true, y_pred))
                    print("MAE_true", mean_absolute_error(y_true, y_pred))
                    #
                    # plt.plot(y_pred, color='red')
                    # plt.plot(y_test)
                    # plt.legend(["Prediction", "test data"])
                    # plt.title("Ordinary KRG " + str(num_identifier))
                    # plt.show()

                    thou2 = (y_train - y_train.mean()).var()

                    kernel = C(thou2, (1e-3, 1e3)) * RBF([.10, .10], (1e-2, 1e2))
                    # Instantiate a Gaussian Process model
                    gp = GaussianProcessRegressor(kernel=kernel, alpha=intrinsic_var, n_restarts_optimizer=10)

                    # Fit to data using Maximum Likelihood Estimation of the parameters
                    gp.fit(x_train, y_train)

                    # Make the prediction on the meshed x-axis (ask for MSE as well)
                    y_pred, sigma = gp.predict(x_test, return_std=True)
                    print("error ", (np.linalg.norm((y_pred - y_test)) / np.linalg.norm(y_test)) * 100)
                    MAPE_SKSK = mean_absolute_percentage_error(y_test, y_pred)

                    print("MAPE", MAPE_SKSK)
                    print("MAE", mean_absolute_error(y_test, y_pred))
                    print("R2:", r2_score(y_test, y_pred))
                    print("MSE", mean_squared_error(y_test, y_pred))
                    # plt.plot(y_pred)
                    # plt.plot(y_test)
                    # plt.show()



                    res = res.append({"filename":i, "MAPE_pot":MAPE_SKSK, "MAPE_SKSK":MAPE_SKSK,
                                      "MAPE_KRG": MAPE_KRG, "MAE_KRG":MAE_KRG }, ignore_index=True)
                    res = res[["filename", "MAPE_pot","MAPE_SKSK","MAPE_KRG", "MAE_KRG"]]
                    result_dir = "pareto{}_{}/".format(alpha,tot_rep)+"res{}.csv".format(vars.index(var))
                    res.to_csv(result_dir, sep=",", index=False)
                    print("-------------------------------------------------")
                except:
                    res = res.append({"filename": i, "MAPE_pot": np.nan, "MAPE_SKSK": np.nan,
                                      "MAPE_KRG": np.nan, "MAE_KRG": np.nan}, ignore_index=True)
                    res = res[["filename", "MAPE_pot", "MAPE_SKSK", "MAPE_KRG", "MAE_KRG"]]
                    result_dir = "pareto{}_{}/".format(alpha,tot_rep)+"res{}.csv".format(vars.index(var))
                    res.to_csv(result_dir, sep=",", index=False)
                    print("-------------------------------------------------")
                    pass
                plot_results.to_csv("plot_results.csv")

                #
                # data_a = [np.array(plot_results[i+"true"]-plot_results[i+"pred"]) for i in test_files ]
                # data_b = [np.array(plot_results[i+"pot"]-plot_results[i+"pred"]) for i in test_files ]
                #
                # ticks = test_files
                #
                # def set_box_color(bp, color):
                #     plt.setp(bp['boxes'], color=color)
                #     plt.setp(bp['whiskers'], color=color)
                #     plt.setp(bp['caps'], color=color)
                #     plt.setp(bp['medians'], color=color)
                #
                # plt.figure(figsize=(20, 10))
                #
                # bpl = plt.boxplot(data_a, positions=np.array(range(len(data_a)))*2.0-0.4, sym='', widths=0.6)
                # bpr = plt.boxplot(data_b, positions=np.array(range(len(data_b)))*2.0+0.4, sym='', widths=0.6)
                # set_box_color(bpl, '#D7191C') # colors are from http://colorbrewer2.org/
                # set_box_color(bpr, '#2C7BB6')
                #
                # # draw temporary red and blue lines and use them to create a legend
                # plt.plot([], c='#D7191C', label='true')
                # plt.plot([], c='#2C7BB6', label='simulated')
                # plt.legend()
                #
                # plt.xticks(range(0, len(ticks) * 2, 2), ticks, rotation=90)
                # plt.xlim(-2, len(ticks)*2)
                # # plt.ylim(0, 8)
                # plt.tight_layout()
                # plt.savefig('boxcompare.png')
