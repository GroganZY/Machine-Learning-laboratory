import numpy as np
import scipy as sp
from scipy.optimize import leastsq #最小二乘法
import matplotlib.pyplot as plt #可视化库
# from sklearn.linear_model import LinearRegression, Lasso, Ridge

#目标函数
def real_func(x):
    return np.sin(2*np.pi*x)

#多项式
def fit_func(p,x):
    """
    eg:p = np.poly1d([2,3,5,7])   

　　　print(p)    ==>>2x3 + 3x2 + 5x + 7   
    """
    f = np.poly1d(p)
    return f(x)

#残差
def residuals_func(p, x, y):
    ret = fit_func(p, x) - y
    return ret

# lasso_poly4 = Lasso()

def fitting(M=0):
    """
    n 为 多项式的次数
    """    
    # 随机初始化多项式参数
    #numpy.random.rand(d0)的随机样本位于[0, 1)之间。d0表示返回多少个
    p_init = np.random.rand(M+1) #生成M+1个随机数的列表
    # 最小二乘法
    p_lsq = leastsq(residuals_func, p_init, args=(x, y)) # 三个参数：误差函数、函数参数列表、数据点
    print('Fitting Parameters:', p_lsq[0])
    return p_lsq
	


def residuals_func_regularization_l1(p, x, y):
    ret = fit_func(p, x) - y
    ret = np.append(ret, np.sqrt(regularization*abs(p))) # L1范数作为正则化项
    return ret


def residuals_func_regularization_l2(p, x, y):
    ret = fit_func(p, x) - y
    ret = np.append(ret, np.sqrt(0.5*regularization*np.square(p))) # L2范数作为正则化项
    return ret



def draw(M=9):
    p_lsq_M = fitting(M)
    # 最小二乘法,加正则化项
    p_init = np.random.rand(M+1)
    p_lsq_regularization_l1 = leastsq(residuals_func_regularization_l1, p_init, args=(x, y))
    p_lsq_regularization_l2 = leastsq(residuals_func_regularization_l2, p_init, args=(x, y))
    plt.plot(x_points, real_func(x_points), label='real')
    plt.plot(x_points, fit_func(p_lsq_M[0], x_points), label='fitted curve')
    plt.plot(x_points, fit_func(p_lsq_regularization_l1[0], x_points), label='regularization_l1')
    plt.plot(x_points, fit_func(p_lsq_regularization_l2[0], x_points), label='regularization_l2')
    plt.plot(x, y, 'bo', label='noise')
    plt.legend()
    plt.show()


if __name__=="__main__":
    #  正则化参数
    regularization = 0.0001
    # 十五个点
    x = np.linspace(-1, 1, 15)
    x_points = np.linspace(-1, 1, 1000)
    # 加上正态分布噪音的目标函数的值
    y_ = real_func(x)
    y = [np.random.normal(0, 0.1)+y1 for y1 in y_]
    draw(M=5)