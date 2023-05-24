import pandas as pd
import numpy as np
import scipy.stats
from matplotlib.lines import Line2D
import scipy.stats as st
import matplotlib.pyplot as plt

dict_design = {1: [50, 10, 200], 2: [50, 5, 400], 3: [50, 1, 2000], 4: [100, 5, 200], 5: [100, 1, 1000],
               6: [50, 20, 1000], 7: [50, 10, 2000], 8: [100, 10, 1000], 9: [100, 5, 2000], 10: [100, 1, 10000],
               11: [50, 100, 2000], 12: [100, 50, 2000], 13: [100, 10, 10000], 14: [100, 5, 20000],
               15: [100, 1, 100000]}

alphas = [.95, .99, .995]
df_paper = pd.DataFrame()
df_pvals = pd.DataFrame()
for alpha in alphas:
    dirs = []
    for rep in range(10):
        dirs.append("pareto{}_{}".format(alpha, rep))

    color = ["black", "blue", "gray", "red"]
    x_ticks = ("Budget{}".format(i) for i in range(1, 10))

    df = pd.DataFrame()
    for dir in dirs:
        df_est = pd.read_csv(dir + "/res0.csv")
        df_emp = pd.read_csv(dir + "/res1.csv")
        df_emp_CHEN = pd.read_csv(dir + "/res2.csv")
        print( len(df_est), len(df_emp), len(df_emp_CHEN))
        df[dir+"EMP_VAR"] = df_emp["MAPE_SKSK"]
        df[dir+"EST_VAR"] = df_est["MAPE_SKSK"]
        df[dir + "Chenvar"] = df_emp_CHEN["MAPE_SKSK"]
        df[dir + "KRG"] = df_est["MAPE_KRG"]
    df.to_csv("all_res_pareto{}.csv".format(alpha))
    df_final = pd.DataFrame()


    EMP_cols = [i for i in df.columns if i.endswith("EMP_VAR")]
    df_sample = df[EMP_cols]
    confidence = []
    Mean=[]

    EMP_all_data = []
    EMP_all = []
    for i in range(len(df_sample)):
        data = list(df_sample.iloc[i])
        EMP_all.append(data)
        temp = [round(i, 2) for i in data if i<=50]
        if i + 1 in [3, 5, 10, 15]:
            temp = []
        cfd = st.t.interval(alpha=0.95, df=len(temp) - 1, loc=np.mean(temp), scale=st.sem(temp))
        cdf_round = [round(i, 2) for i in cfd]
        Mean.append(np.median(temp))
        confidence.append(str(cdf_round))
        EMP_all_data.append(temp)
    df_final["EMP_VAR"] = confidence
    df_final["EMP_VAR-mean"] = Mean

    EMP_cols = [i for i in df.columns if i.endswith("EST_VAR")]
    df_sample = df[EMP_cols]
    confidence = []
    EST_VAR_ALL_DATA = []
    EST_VAR_ALL = []
    Mean = []
    for i in range(len(df_sample)):
        data = list(df_sample.iloc[i])
        EST_VAR_ALL.append(data)
        temp = [round(i, 2) for i in data if i<=50]
        cfd = st.t.interval(alpha=0.95, df=len(temp) - 1, loc=np.mean(temp), scale=st.sem(temp))
        cdf_round = [round(i, 2) for i in cfd]
        confidence.append(str(cdf_round))
        Mean.append(np.median(temp))
        EST_VAR_ALL_DATA.append(temp)
    df_final["EST_VAR"] = confidence
    df_final["EST_VAR-mean"] = Mean



    EMP_cols = [i for i in df.columns if i.endswith("Chenvar")]
    df_sample = df[EMP_cols]
    confidence = []
    Mean = []
    CHEN_ALL_DATA = []
    CHEN_ALL = []
    for i in range(len(df_sample)):
        data = list(df_sample.iloc[i])
        CHEN_ALL.append(data)
        temp = [round(i, 2) for i in data if i<=50]
        cfd = st.t.interval(alpha=0.95, df=len(temp) - 1, loc=np.mean(temp), scale=st.sem(temp))
        cdf_round = [round(i, 2) for i in cfd]
        confidence.append(str(cdf_round))
        Mean.append(np.median(temp))
        CHEN_ALL_DATA.append(temp)
    df_final["Chen_VAR"] = confidence
    df_final["Chen_VAR-mean"] = Mean




    EMP_cols = [i for i in df.columns if i.endswith("KRG")]
    df_sample = df[EMP_cols]
    confidence = []
    Mean = []
    KRG_ALL_DATA = []
    KRG_ALL = []

    for i in range(len(df_sample)):
        data = list(df_sample.iloc[i])
        KRG_ALL.append(data)
        temp = [round(i, 2) for i in data if i<=200]
        cfd = st.t.interval(alpha=0.95, df=len(temp) - 1, loc=np.mean(temp), scale=st.sem(temp))
        cdf_round = [round(i, 2) for i in cfd]
        confidence.append(str(cdf_round))
        Mean.append(np.median(temp))
        KRG_ALL_DATA.append(temp)
    df_final["KRG"] = confidence
    df_final["KRG-mean"] = Mean

    def set_box_color(bp, color):
        plt.setp(bp['boxes'], color=color)
        plt.setp(bp['whiskers'], color=color, linestyle='-', linewidth=3)
        plt.setp(bp['caps'], color=color)
        plt.setp(bp['medians'], color=color, linewidth=5)
        plt.setp(bp['fliers'], mfc=color,  marker="o")
        plt.setp(bp['means'], mfc=color)



    pos = np.array([i*6 for i in range(1, 16)])
    plt.figure(figsize=(10, 5))

    # plt.scatter(pos,, color="#D7191C")

    p_value_EST = []
    p_value_EST2 = []
    p_value_EST3 = []
    for i in range(len(EST_VAR_ALL_DATA)):
        est = EST_VAR_ALL[i]
        chen = CHEN_ALL[i]
        emp = EMP_all[i]
        krg = KRG_ALL[i]
        print(len(est), len(chen))
        dif1 = []
        dif2 = []
        for i in range(10):
            try:
                x = est[i]
                y = chen[i]
                temp = x - y
                if not np.isnan(temp):
                    dif1.append(temp)
                    dif2.append(-temp)
                else:
                    dif1.append(0)
                    dif2.append(0)
            except:
                dif1.append(0)
                dif2.append(0)

        w, p = scipy.stats.wilcoxon(dif1, alternative="less")
        w2, p2 = scipy.stats.wilcoxon(dif2)
        w3, p3 = scipy.stats.wilcoxon(dif2, alternative="less")

        print(w, p, w2, p2)
        if i in [3, 5, 10, 15]:
            continue
        p_value_EST.append(p)
        p_value_EST2.append(p2)
        p_value_EST3.append(p3)
    df_pvals[str(alpha)] = p_value_EST
    df_pvals[str(alpha) + "equal"] = p_value_EST2
    df_pvals[str(alpha) + "reverse"] = p_value_EST3


    # bp_KRG = plt.boxplot(KRG_ALL_DATA,notch=False, positions=pos, labels=["KRG" for i in range(15)], showbox=False, manage_ticks=False, showfliers=False)

    bp_EMP= plt.boxplot(EMP_all_data,notch=False, positions=pos +1, labels=["EMP" for i in range(15)], showbox=False, manage_ticks=False, showfliers=False)
    # plt.scatter(pos, m_EMP,marker="s", color= '#D7191C')
    # plt.scatter(pos+1, m_EST,marker="s", color= '#2C7BB6')
    bp_CHEN = plt.boxplot(CHEN_ALL_DATA,notch=False, positions=pos+2, labels=["CHEN" for i in range(15)], showbox=False, manage_ticks=False, showfliers=False)
    # plt.scatter(pos+2, m_EST_EMP,marker="s", color= 'gray')
    bp_EST = plt.boxplot(EST_VAR_ALL_DATA,notch=False, positions=pos+3, labels=["Budget {}".format(i) for i in range(1, 16)], showbox=False, showfliers=False)

    # plt.show()
    # plt.scatter(pos+3, m_KRG,marker="s", color= 'orange')

    plt.xticks(rotation=45)

    set_box_color(bp_EMP, 'red') # colors are from http://colorbrewer2.org/
    set_box_color(bp_EST, 'orange')
    set_box_color(bp_CHEN, 'gray')
    # set_box_color(bp_KRG, 'blue')



    colors = [ 'red', 'gray', "orange"]
    lines = [Line2D([0], [0], color=c, linewidth=3, linestyle='--') for c in colors]
    # labels = ["EMP", "EST", "EST-EMP", "KRG"]
    labels = [ "POT-EMP", "EMP-EMP", "POT-EVT"]
    plt.legend(lines, labels)


    # plt.xticks(x_ticks, rotation=90)
    # plt.legend(["EMP", "EST", "EST-EMP", "KRG"], loc='upper right', numpoints=1)
    plt.tight_layout()
    plt.savefig("pareto{}.pdf".format(alpha))
    plt.show()


    print(df_final)
    df_final.to_csv("final_result_pareto{}.csv".format(alpha))

    empty_list = []
    for row in range(len(df_final)):
        empty_list.append([df_final.iloc[row]["KRG"], df_final.iloc[row]["KRG-mean"], "KRG"])
        empty_list.append([df_final.iloc[row]["EMP_VAR"], df_final.iloc[row]["EMP_VAR-mean"], "EMP_KRG"])
        empty_list.append([df_final.iloc[row]["Chen_VAR"], df_final.iloc[row]["Chen_VAR-mean"], "Chen_VAR"])
        empty_list.append([df_final.iloc[row]["EST_VAR"], df_final.iloc[row]["EST_VAR-mean"], "EST_VAR"])
    df_paper[[str(alpha), str(alpha) + "mean", "method{}".format(alpha)]] = empty_list

df_paper.to_csv("final_df_paper_format.csv")
df_pvals.to_csv("pvals.csv")