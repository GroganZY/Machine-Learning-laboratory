import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron

# load data
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['label'] = iris.target

df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
"""发现有3各种类，一共150条数据"""

data = np.array(df.iloc[:100, [0, 1, -1]])  # 前100条数据，取第一列 第二列 第三列
X, y = data[:,:-1], data[:,-1]  # 切片 取前两列   取后一列
y = np.array([1 if i == 1 else -1 for i in y])  # 转换成1和-1两类


"""数据预处理结束"""

clf = Perceptron()  # 默认参数
clf.fit(X, y)
# Weights assigned to the features.
print(clf.coef_)
# 截距 Constants in decision function.
print(clf.intercept_)
x_ponits = np.arange(4, 8)
y_ = -(clf.coef_[0][0]*x_ponits + clf.intercept_)/clf.coef_[0][1]
plt.plot(x_ponits, y_)

plt.plot(data[:50, 0], data[:50, 1], 'bo', color='blue', label='0')
plt.plot(data[50:100, 0], data[50:100, 1], 'bo', color='orange', label='1')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()
plt.show()