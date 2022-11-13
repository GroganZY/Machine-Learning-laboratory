"""首先，这是没问题的。可以这么写，不过不可以实现高次你和，只能是一次函数"""
import numpy as np
import matplotlib.pyplot as plt #可视化库
import warnings
warnings.filterwarnings("ignore")
# from sklearn.datasets import make_regression
# X,y=make_regression(n_samples=10,n_features=1,noise=20)
# print("make_regression：")
# print(X)
# print()
# print(y)

#  之前的是生成10个随机值，加进sin中进行拟合
#  这次是随机生成10个点，有x有y，只要把这个xy换成上次的sin上的点即可

import math

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

# #残差
# def residuals_func(p, x, y):
#     ret = fit_func(p, x) - y
#     return ret
	
	
	
# # 十个点
# x = np.linspace(0, 1, 10)
# x_points = np.linspace(0, 1, 1000)



# # 加上正态分布噪音的目标函数的值
# y_points = real_func(x)
# y = [np.random.normal(0, 0.1)+y1 for y1 in y_points]



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

class Regression(object):
    
    """
        基础线性回归模型，使用输入的X和y进行参数回归
        超参：
        n_iterations:int 训练的步数，迭代多少次
        learning_rate:float 学习率
        
        内部函数:
        initialize_weights:初始化参数
        fit:开始训练
        predict:预测
        
        内部的数据:
        n_iterations
        learning_rate
        regularization:正则化参数
        regularization.grad:正则化的梯度函数
    """
    
    def __init__(self, n_iterations, learning_rate):
        self.n_iterations=n_iterations
        self.learning_rate=learning_rate
        self.regularization=lambda x:0
        self.regularization.grad=lambda x:0
    
    def initialize_weights(self, n_features):
        """初始化系数，输入是feature的个数，输出是一个随机初始化好的参数矩阵,[-1/sqrt(N),1/sqrt(N)]"""
        # 实验得出经典算法，随机分布初始化系数
        limit=1/math.sqrt(n_features)
        self.w=np.random.uniform(-limit,limit,(n_features,))
        #Uniform Distribution/Xavier/MSRA/Gaussian 高斯初始化
    
    def fit(self, X, y):
        #插入偏置列1到X中
        X = np.insert(X,0,1,axis=1)#给每一行的第0列增加一个1
        self.training_errors = []#保存每一次步长的训练Loss
        self.initialize_weights(n_features=X.shape[1])#初始化参数w
        
        #进行梯度下降迭代
        for i in range(self.n_iterations):
            y_pred=X.dot(self.w)#进行预测
            #计算Loss
            mse=np.mean(0.5*(y-y_pred)**2+self.regularization(self.w))
            self.training_errors.append(mse)#将Loss加入到training_errors的数组中
            #计算带有正则化项的梯度
            g_w=-(y-y_pred).T.dot(X)/len(X)+self.regularization.grad(self.w)
            #根据梯度下降的算法更新参数
            self.w-=self.learning_rate*g_w
            
    def predict(self,X):
        #通过输入X预测一个样本
        X=np.insert(X,0,1,axis=1)
        pred=X.dot(self.w)
        return pred
    


class l1_regularization():
    """L1正则化类/函数
    参数:
    
    alpha--L1正则化系数
    """
    def __init__(self, alpha):
        self.alpha=alpha
    def __call__(self,w):
        return self.alpha*np.linalg.norm(w,ord=1)
    def grad(self,w):
        #w>0->w`=1;w<0->w`=0;w==0->w`=0
        return self.alpha*np.sign(w)

class l2_regularization():
    """L2正则化参数
    参数：
    
    alpha 正则化系数
    """
    def __init__(self,alpha):
        self.alpha=alpha
    
    def __call__(self,w):
        return self.alpha*0.5*w.T.dot(w)
    
    def grad(self,w):
        return self.alpha*w


class LassoLinearRegression(Regression):
    def __init__(self,alpha,n_iterations=1000,learning_rate=0.01):
        self.regularization=l1_regularization(alpha=alpha)
        super(LassoLinearRegression,self).__init__(n_iterations,learning_rate)
        
    def fit(self,X,y):
        super(LassoLinearRegression,self).fit(X,y)
    def predict(self,X):
        return super(LassoLinearRegression,self).predict(X)



def test_and_draw(model,X_test,y_test):
    y_pred=model.predict(X_test)
    # mse=mean_squared_error(y_test,y_pred)
    # print("方差:",mse)
    plt.plot(X_test,y_test,'k.')
    plt.plot(X_test,y_pred,'y')
    plt.show()


model=LassoLinearRegression(alpha=120, n_iterations=1000,learning_rate=0.1)
model.fit(X_train,y_train)
test_and_draw(model,X_train,y_train)
