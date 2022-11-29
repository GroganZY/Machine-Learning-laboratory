from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")


def classify_2():  
    """二分类"""
    # data
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    # data = np.array(df.iloc[:100, [0, 1, -1]])

    """预处理结束"""

    data = np.array(df.iloc[:100, [0, 1, -1]])   # 第一 第二 倒数第一
    X, y = data[:,:-1], data[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    clf_sk = KNeighborsClassifier()
    clf_sk.fit(X_train, y_train)
    clf_sk.score(X_test, y_test)

    # 预测 两个数据

    test_point = [[6.0, 3.0],[2.0,6.0]]
    print('Test Point: {}'.format(clf_sk.predict(test_point)))

    # 画出分布图

    plt.scatter(df[:50]['sepal length'], df[:50]['sepal width'], label='0')
    plt.scatter(df[50:100]['sepal length'], df[50:100]['sepal width'], label='1')
    plt.plot(test_point[0][0], test_point[0][1], 'bo', label='test_point')
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')
    plt.legend()
    plt.show()


def classify_3():
    """三分类，二维"""
        # data
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    # data = np.array(df.iloc[:100, [0, 1, -1]])

    """预处理结束"""

    data = np.array(df.iloc[:150, [0, 1, -1]])
    print(data)
    X, y = data[:,:-1], data[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    print(X)
    print()
    print(y)
    clf_sk = KNeighborsClassifier()
    clf_sk.fit(X_train, y_train)
    clf_sk.score(X_test, y_test)

    # # 预测 三个数据

    test_point = [[6.0, 3.0],[5.0,2.0],[6,4]]
    print('Test Point: {}'.format(clf_sk.predict(test_point)))

    # # 画出分布图

    plt.scatter(df[:50]['sepal length'], df[:50]['sepal width'], label='0')
    plt.scatter(df[50:100]['sepal length'], df[50:100]['sepal width'], label='1')
    plt.scatter(df[100:150]['sepal length'], df[100:150]['sepal width'], label='2')
    num=0
    for [j,k] in test_point :
        num+=1
        plt.plot(j, k, 'bo', label='test_point'+str(num))
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')
    plt.legend()
    plt.show()


def classify_3_4D():
    """三分类，四维"""
        # data
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    # data = np.array(df.iloc[:100, [0, 1, -1]])

    """预处理结束"""

    data = np.array(df.iloc[:150, [0, 1, 2, 3, 4]])
    # print(data)
    X, y = data[:,:-1], data[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # print(X)
    # print()
    # print(y)
    clf_sk = KNeighborsClassifier()
    clf_sk.fit(X_train, y_train)
    print("预测准确度为：{}".format(clf_sk.score(X_test, y_test)))
    # 四维预测没问题，但是画图咱这三维空间还是算了，如果硬是要画图，可以先聚类，四类变3类或者2类，再分类并画图
    # # 预测 三个数据

    test_point = [[6.0, 3.0,3.0,2.0],[5.0,2.0,3.0,3.0],[6,4,2,2]]
    print('Test Point: {}'.format(clf_sk.predict(test_point)))

if __name__=="__main__":
    # classify_2()
    # classify_3()
    classify_3_4D()
    pass