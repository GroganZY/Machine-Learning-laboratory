
import numpy as np
import matplotlib.pyplot as plt #可视化库
import math

# print("make_regression：")
# print(X)
# print()
# print(y)

#  之前的是生成10个随机值，加进sin中进行拟合
#  这次是随机生成10个点，有x有y，只要把这个xy换成上次的sin上的点即可

from sklearn.datasets import make_regression
X,y=make_regression(n_samples=10,n_features=1,noise=20)

def shuffle_data(X,y,seed=None):
    "将X和y的数据进行随机排序/乱序化"
    if seed:
        np.random.seed(seed)
    idx=np.arange(X.shape[0])
    print(type(idx))
    np.random.shuffle(idx)
    return X[idx],y[idx] #对于np.array，idx作为index数组可以改变array的顺序

def train_test_split(X,y,test_size=0.5,shuffle=True,seed=None):
    '将数据集根据test_size分成训练集和测试集，可以指定是否随机洗牌'
    if shuffle:
        X,y=shuffle_data(X,y,seed)
    split_i=len(y)-int(len(y)//(1/test_size)) #//号保留它的int值
    #split_i=len(y)-int(len(y)*test_size)
    #分割点确定X，y都确定
    X_train,X_test=X[:split_i],X[split_i:]
    y_train,y_test=y[:split_i],y[split_i:]
    
    return X_train, X_test, y_train, y_test

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
print("X_train")
print(X_train)
print("y_train")
print(y_train)


print("===============================================")


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

x =( [i] for i in np.linspace(0, 1, 10))
print(x)
# x_points = np.linspace(0, 1, 1000)
# y_points = real_func(x_points)
y_points = real_func(x)
y = [np.random.normal(0, 0.1)+y1 for y1 in y_points]

print(x)
print(y_points)

# 数据类型错误