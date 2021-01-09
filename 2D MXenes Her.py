from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import AdaBoostRegressor


def guji(pre_y, test_y):
    pre = pre_y
    pre_pd = pd.DataFrame(pre)
    test_y_g = test_y
    mae = mean_squared_error(test_y_g, pre)
    r2 = r2_score(test_y_g, pre)
    error = round((mae+r2)/2, 2)
    print('r2=', r2)
    print('error=', error)
    return error


def map_plot(test_x,test_y,pre_y):
    test_x_m = test_x['I_D']
    test_y_m = test_y
    pre_y = pre_y
    plt.plot(test_x_m,pre_y,'r-')
    plt.scatter(test_x_m,test_y)
    plt.show()



def model_AdaBoost(train_x, train_y, test_x, test_y):
    print('运行AdaBoost模型')
    lar = AdaBoostRegressor()
    lar.fit(train_x, y=train_y)
    pre_y = lar.predict(test_x)
    # print('pre_y = ', pre_y)
    map_plot(test_x,test_y,pre_y)
    guji(pre_y,test_y)
    print('************************************')
    # print()
    return pre_y

def model_rfr(train_x, train_y, test_x, test_y):
    print('运行RFR模型')
    dfr = DecisionTreeRegressor()
    dfr.fit(train_x, train_y)
    pre_y = dfr.predict(test_x)
    # print('pre_y = ', pre_y)
    map_plot(test_x,test_y,pre_y)
    guji(pre_y,test_y)
    print('************************************')
    # print()
    return pre_y




def fil_open():
    data_all = pd.read_csv(filepath_or_buffer=r'D:\ML-2020.12.29\sx_MBene-master\Figure7\SVR\ML_Figure7.csv',
                           header=0,
                           encoding='utf-8',
                           error_bad_lines=False)
    data_list = list(data_all)
    drop_name_unname = data_list[0]
    # print('drop_name_unname = ', drop_name_unname)
    data_use = data_all.drop(drop_name_unname, axis=1)
    use_list = list(data_use)
    # print('use_list =', use_list)
    # print('data_use = ', data_use)
    y = data_use['ΔG']
    x = data_use.drop(['ΔG'], axis=1, inplace=False)
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.25, random_state=14)
    train_y = train_y.values
    test_y = test_y.values
    return train_x, train_y, test_x, test_y




if __name__ == '__main__':
    train_x, train_y, test_x, test_y = fil_open()
    model_AdaBoost(train_x, train_y, test_x, test_y)
    model_rfr(train_x, train_y, test_x, test_y)


