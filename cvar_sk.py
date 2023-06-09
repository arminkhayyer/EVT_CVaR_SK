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


files = sorted(os.listdir("normal"))
# files = ["normal4.csv","normal9.csv"]
test_files = sorted(os.listdir("normal_test"))


#Once the design points are found by running the last cell, each design point is enterd into LCOM and desirable
#statistics are collected for each design point, in this example we only collect A3 statistc. The following lines of
#in this cell grab a dataframe from a csv file. The final dataframe has 6 columns, the first 4 represent the parts
#and the last two columns are the mean and variance of A3 statistic for 30 replications
res = pd.DataFrame(columns=[])

dict_design = {1:[50, 20, 1000], 2:[100, 10, 1000], 3:[100,5,2000], 4: [100, 1, 10000], 5:[50, 100, 1000],
               6:[100, 50, 1000], 7:[100, 10, 5000], 8:[100, 5, 10000], 9:[100, 1, 50000] }

plot_results = pd.DataFrame()
def True_cvar(row):
    alpha = .05
    mu = (row["x1"] * math.sin(np.pi * row["x2"])) + (row["x2"] * math.sin(np.pi * row["x1"]))
    sigma = math.sqrt(row["x1"] ** 2 + row["x2"] ** 2)
    cdf = norm.ppf(alpha)
    pdf = norm.pdf(cdf)
    true_val = mu + (sigma * pdf / (alpha))
    return true_val



for i in files:
    num_rep = dict_design[int(files.index(i)) + 1][1]
    print(i, files.index(i)+ 1)
    filename ="normal/"+ i
    df = pd.read_csv(filename)
    df = df.drop(["Unnamed: 0"], 1)

    filename_test = "normal_test/" + "normal.csv"
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
    y_data = (np.array(df['POT_conditional_va_risk']))
    y_data_test = (np.array(df_test['True_cvar']))
    y_true = (np.array(df_test['True_cvar']))
    y_emp = (np.array(df_test['EMP_conditional_vaRisk']))
    intrinsic_var = np.array(df["Emperical_variance_of_cvar"])/num_rep

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
    print("error ", np.linalg.norm(y_pred - y_test) / np.linalg.norm(y_test) * 100)
    print("MAPE", mean_absolute_percentage_error(y_test, y_pred))
    print("MAE", mean_absolute_error(y_test, y_pred))
    print("R2:", r2_score(y_test, y_pred))
    print("MSE", mean_squared_error(y_test, y_pred))

    plt.plot(y_pred, color='red')
    plt.plot(y_test)
    plt.legend(["Prediction", "test data"])
    plt.title("Ordinary KRG" + str(files.index(i)+ 1))
    plt.show()


    def Cov_extrinsic(theta):
        '''
        this function returns the spatial covariance matrix amonge all the design points. it takes thetha as input
        and outputs a k*k matrix.
        '''
        cov_ex_mat = np.zeros(shape=(len(y_train), len(y_train) ))
        for i in range(len(x_train)):
            for j in range(len(x_train)):
                sumation = - sum(theta[d]*(x_train[i][d] - x_train[j][d])**2 for d in range(len(x_train[i])))
                cov_ex_mat[i, j] = math.exp(sumation)
        return(cov_ex_mat)

    def get_lowe_tri_cholesky(cov):
        return np.linalg.cholesky(cov, lower=True)

    mean = y_train.mean()
    thou2 = (y_train-  y_train.mean()).var()
    print (mean)
    print(thou2)

    def liklihood(x):
        """
        This function calculate the above-mentioned liklihood function assuming that mean, tou2, and thetha are given
        """
        theta = x[2:]
        cov_exter = x[1] * Cov_extrinsic(theta) + intrinsic_cov
        cov_exter_det = np.linalg.det(cov_exter)

        cov_exter_inverse = np.linalg.inv(cov_exter)
        resid = y_train - x[0] * np.ones((len(y_train)))
        weights = resid.transpose().dot(cov_exter_inverse).dot(resid)
        lik =  - .5* math.log2(cov_exter_det + .0002) - .5*(weights) -(.5 * len(y_train)* math.log2(2*math.pi))
        # print(-lik)
        return -lik

    # In the following lines of code, it is tried to maximize the liklihood function by using SLSQP algorithm to
    # find the best set of parameters of the liklihood function namely mean, thou2, theta

    cons = ({'type': 'ineq', 'fun': lambda x:  x[1] + 0},
           {'type': 'ineq', 'fun': lambda x: x[2] +0})
    thetha = [100,100]
    x0 = [mean, thou2] + thetha
    # print(x0)
    sol = minimize(liklihood, x0, method='SLSQP', constraints=cons, options={"maxiter":200, "disp": True})
    x = sol.x
    print (x)

    # once
    cov_exter = x[1] * Cov_extrinsic(x[2:]) + intrinsic_cov
    cov_exter_inverse = np.linalg.inv(cov_exter)
    resid = y_train - x[0] * np.ones(len(y_train))

    def predict(B0, thou2, theta, point):
        cov_ex_mat = np.zeros(len(y_train))
        for i in range(len(x_train)):
                sumation = - sum(theta[d]*(x_train[i][d] - point[d])**2 for d in range(len(point)))
                cov_ex_mat[i] = math.exp(sumation)
        cov_ex_mat= thou2 * cov_ex_mat
        weights = cov_ex_mat.dot(cov_exter_inverse).dot(resid)
        value = B0 + weights
        return value, cov_ex_mat


    def MSE(B0, thou2, theta, point):
        cov_ex_matrix = np.zeros(len(y_train))
        for i in range(len(x_train)):
                sumation = - sum(theta[d]*(x_train[i][d] - point[d])**2 for d in range(len(point)))
                cov_ex_matrix[i] = math.exp(sumation)
        cov_ex_mat= (thou2 **2) * cov_ex_matrix
        weights = np.transpose(cov_ex_mat).dot(cov_exter_inverse).dot(cov_ex_matrix)
        sigma = 1 - np.ones(len(y_train)).dot(cov_exter_inverse).dot(cov_ex_matrix)*thou2
        righthandside_term = np.ones(len(y_train)).dot(cov_exter_inverse).dot(np.ones(len(y_train)))
        # righthandside_term = np.linalg.inv(righthandside_term)
        B0_variability = sigma * sigma/righthandside_term
        value = thou2 - weights + B0_variability
        return value
#
# print(predict(x[0], x[1], x[2:], x_test[6])[0])
# print(y_test[6])

# once the parameters are estimated, they can be used to interpolate values for other points with
# unknown response value.

    pred = [predict(x[0], x[1], x[2:], i)[0] for i in x_test]
    plot_results[str(files.index(i))+ "pred"] = pred


    mse = [MSE(x[0], x[1], x[2:], i) for i in x_test]
    mse = np.average(mse)
    print("mse", mse)

    plt.plot( pred)
    plt.plot( y_test)
    # plt.plot(x_test[:, 0], y_true)
    plt.title("SK" + str(files.index(i)+ 1))
    plt.show()

    mae_pot = mean_absolute_error(y_test, pred)
    mean_se_pot = mean_squared_error(y_test, pred)
    print("MEAN ABSOLUTE ERROR: ",mae_pot  )
    print("MEAN sqaured ERROR: ",mean_se_pot )
    def mean_absolute_percentage_error(y_true, y_pred):
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    mape_pot = mean_absolute_percentage_error(y_test, pred)
    print("MAPE",mape_pot)

    mae_true = mean_absolute_error(y_true, pred)
    mean_se_true = mean_squared_error(y_true, pred)
    mape_true = mean_absolute_percentage_error(y_true, pred)

    mae_emp = mean_absolute_error(y_emp, pred)
    mean_se_emp = mean_squared_error(y_emp, pred)
    mape_emp = mean_absolute_percentage_error(y_emp, pred)


    res = res.append({"filename":files.index(i), "MAPE_pot":mape_pot, "mse_hat":mse,"mse_pot":mean_se_pot, "mea_pot": mae_pot, "mape_true":mape_true,
                      "mse_true":mean_se_true, "mae_true":mae_true, "mape_emp":mape_emp,
                      "mse_emp":mean_se_emp, "mae_emp":mae_emp}, ignore_index=True)
    res = res[["filename", "mse_hat", "MAPE_pot","mape_true","mea_pot","mae_true","mse_pot","mse_true"]]
    res.to_csv("res.csv", sep=",", index=False)
    print("-------------------------------------------------")
# except:
    #     pass
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