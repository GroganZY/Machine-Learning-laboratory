# import numpy as np
# import matplotlib.pyplot as plt
# import scipy.optimize as opt
# def func(x):
#   return np.sin(2*np.pi*x)
# x = np.linspace(0, 1, 10)
# y = func(x)
# xi = np.linspace(0, 1, 30)

# #多项式
# def fit_func(p,x):
#     """
#     eg:p = np.poly1d([2,3,5,7])
# 　　　print(p)==>>2x3 + 3x2 + 5x + 7
#     """
#     f = np.poly1d(p)
#     return f(x)



# #残差
# def residuals_func(p, x, y):
#     ret = fit_func(p, x) - y
#     return ret


# def fitting(M=0):
#     """
#     n 为 多项式的次数
#     """    
#     # 随机初始化多项式参数
#     #numpy.random.rand(d0)的随机样本位于[0, 1)之间。d0表示返回多少个
#     p_init = np.random.rand(M+1) #生成M+1个随机数的列表
#     print(p_init)
#     # 最小二乘法
#     p_lsq = opt.leastsq(residuals_func, p_init, args=(x, y)) # 三个参数：误差函数、函数参数列表、数据点
#     print('Fitting Parameters:', p_lsq)
    
#     # 可视化
#     plt.plot(xi, func(xi), label='real')  #dui de
#     # plt.show()
#     plt.plot(xi, fit_func(p_lsq[0], xi), label='fitted curve')
#     # plt.show()
#     plt.plot(x, y, 'bo', label='noise')
#     # plt.show()
#     plt.legend()
#     plt.show()
#     return p_lsq
    
# # M=0
# p_lsq = fitting(M=4)

import numpy as np
import scipy as sp
from scipy.optimize import leastsq #最小二乘法
import matplotlib.pyplot as plt #可视化库
# %matplotlib inline

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
	
	
	
# 十个点
x = np.linspace(0, 1, 10)
x_points = np.linspace(0, 1, 1000)

# 加上正态分布噪音的目标函数的值
y_ = real_func(x)
y = [np.random.normal(0, 0.1)+y1 for y1 in y_]

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
    
    # 可视化
    plt.plot(x_points, real_func(x_points), label='real')
    plt.plot(x_points, fit_func(p_lsq[0], x_points), label='fitted curve')
    plt.plot(x, y, 'bo', label='noise')
    plt.legend()
    plt.show()
    return p_lsq
	
# M=0
p_lsq = fitting(M=2)

p_lsq_9 = fitting(M=8)