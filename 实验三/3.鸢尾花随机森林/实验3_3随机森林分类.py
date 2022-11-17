import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import matplotlib as mpl
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
 
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False
RF = RandomForestClassifier(n_estimators=100, n_jobs=4, oob_score=True)
iris = load_iris()
X1 = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size=0.25, random_state=0)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
RF.fit(X_train, y_train)
y_pred = RF.predict(X_test)
print("随机森林准确率:", accuracy_score(y_test, y_pred))
print("其他评估指标：\n", classification_report(y_test, y_pred, target_names=['0', '1', '2']))
# 混淆矩阵在二分类和多分类中的使用 https://blog.csdn.net/Orange_Spotty_Cat/article/details/80520839
 
 
# 画图
x = iris.data[:, :2]
RF.fit(x, y)
N = 50
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
# for weight in ['uniform', 'distance']:
x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, N), np.linspace(y_min, y_max, N))
z_show = np.stack((xx.flat, yy.flat), axis=1)  # 测试点
# z = RF.predict(np.c_[xx.ravel(), yy.ravel()])
z = RF.predict(z_show)
plt.figure()
plt.pcolormesh(xx, yy, z.reshape(xx.shape), shading='auto', cmap=cmap_light)
plt.scatter(x[:, 0], x[:, 1], c=y, cmap=cmap_bold, edgecolors='k', s=20)
plt.xlim(xx.min(), xx.max())
plt.title('RandomForestClassifier')
plt.show()
# print('RandomForestClassifier:', RF.score(x, y))