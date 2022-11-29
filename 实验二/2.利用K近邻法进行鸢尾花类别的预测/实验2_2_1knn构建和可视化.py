import math
from itertools import combinations

# p = 1 曼哈顿距离
# p = 2 欧氏距离
# p = inf 明式距离minkowski_distance
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from collections import Counter

# def L(x, y, p=2):
#     # x1 = [1, 1], x2 = [5,1]
#     if len(x) == len(y) and len(x) > 1:
#         sum = 0
#         for i in range(len(x)):
#             sum += math.pow(abs(x[i] - y[i]), p)
#         return math.pow(sum, 1/p)
#     else:
#         return 0


# # 课本例3.1
# x1 = [1, 1]
# x2 = [5, 1]
# x3 = [4, 4]

# # x1, x2
# for i in range(1, 5):
#     r = { '1-{}'.format(c):L(x1, c, p=i) for c in [x2, x3]}
#     print(min(zip(r.values(), r.keys())))


# data
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['label'] = iris.target
df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
# data = np.array(df.iloc[:100, [0, 1, -1]])

"""预处理结束"""

plt.scatter(df[:50]['sepal length'], df[:50]['sepal width'], label='0')
plt.scatter(df[50:100]['sepal length'], df[50:100]['sepal width'], label='1')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()
plt.show()

"""可视化结束"""

