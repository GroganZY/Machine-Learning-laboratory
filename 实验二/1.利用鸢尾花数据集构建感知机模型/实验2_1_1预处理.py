import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
# %matplotlib inline

# load data
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['label'] = iris.target

df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
print(df)
print()
print(df.label.value_counts())
"""发现有3各种类，一共150条数据"""
# print(df)
def draw_data():
    plt.scatter(df[:50]['sepal length'], df[:50]['sepal width'], label='0')
    plt.scatter(df[50:100]['sepal length'], df[50:100]['sepal width'], label='1')
    plt.scatter(df[100:150]['sepal length'], df[100:150]['sepal width'], label='2')
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')
    plt.legend()
    plt.show()

# draw_data()

data = np.array(df.iloc[:100, [0, 1, -1]])  # 前100条数据，取第一列 第二列 第三列
X, y = data[:,:-1], data[:,-1]  # 切片 取前两列   取后一列
y = np.array([1 if i == 1 else -1 for i in y])  # 转换成1和-1两类


"""数据预处理结束"""