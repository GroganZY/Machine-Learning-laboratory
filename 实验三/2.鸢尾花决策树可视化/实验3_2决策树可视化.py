from sklearn.tree import DecisionTreeClassifier

from sklearn.tree import export_graphviz
import graphviz
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from collections import Counter
import math
from math import log

import pprint

# data
def create_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    data = np.array(df.iloc[:100, [0, 1, -1]])
    # print(data)
    return data[:,:2], data[:,-1]

X, y = create_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
'''数据处理完成'''

datasets, labels = create_data()
train_data = pd.DataFrame(datasets, columns=labels)



clf = DecisionTreeClassifier()
clf.fit(X_train, y_train,)
print(clf.score(X_test, y_test))
tree_pic = export_graphviz(clf, out_file="mytree.pdf")
with open('mytree.pdf') as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)