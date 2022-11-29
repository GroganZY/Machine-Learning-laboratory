# 
# 实验一 线性回归（黑体三号）
## 一、实验目的
1.搭建机器学习开发平台。
2.掌握线性回归分析的基本思想和基本方法（难点）。
3.掌握最小二乘法原理及实现（重点）。
## 二、实验原理及说明
线性回归是在已有数据集上通过构建一个线性的模型来拟合该数据集特征向量的各个分量之间的关系，对于需要预测结果的新数据，我们利用已经拟合好的线性模型来预测其结果。最小二乘法是用的比较广泛的一种方法。
高斯于 1823年在误差独立同分布的假定下，证明了最小二乘方法的一个最优性质:在所有无偏的线性估计类中, 最小二乘方法是其中方差最小的！ 对于数据(𝑥𝑥𝑖𝑖,𝑦𝑦𝑖𝑖)(𝑖𝑖=1,2,3...,n)，拟合出函数 ℎ(𝑥)有误差，即残差：𝑟𝑖=ℎ(𝑥𝑖)−𝑦𝑖，此时 L2 范数(残差平方和) 最小时， h(x) 和 y相似度最高， 更拟合一般的 H(x) 为 n次的多项式：
𝐻𝐻(𝑥𝑥)=𝑤0+𝑤1𝑥+𝑤2𝑥2+...𝑤𝑛𝑥𝑛，其中w(w0,w1,w2,...,wn)为参数，最小二乘法就是要找到一组w(w0,w1,w2,...,wn)，使得残差平方和最小。
## 三、实验内容
### 1．搭建机器学习开发平台。
#### （1）安装 Anaconda开发平台，使用 jupyter notebook进行编辑。
#### （2）建立虚拟环境：
创建虚拟环境：使用 conda create -n your_env_name python=X.X（3.6、3.8 等），anaconda命令创建 python版本为 X.X、名字为 your_env_name 的虚拟环境。your_env_name 文件可以在 Anaconda 安装目录 envs 文件下找到。
激 活 虚 拟 环 境 ： 使 用 如 下 命 令 即 可 激 活 创 建 的 虚 拟 环 境activate your_env_name(虚拟环境名称)，此时使用 python --version可以检查当前 python版本是否为想要的（即虚拟环境的 python 版本）。
退出虚拟环境：使用如下命令即可退出创建的虚拟环境 deactivate env_name，
也可以使用“activate root”切回 root环境。
删除虚拟环境：使用命令 conda remove -n your_env_name(虚拟环境名称) --al
即可删除。
![image.png](https://cdn.nlark.com/yuque/0/2022/png/32555890/1668700137806-417cd7f2-2fbb-4d17-a7a9-b72caa6d3624.png#averageHue=%23171717&clientId=u8c3edc07-3917-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=222&id=uba04a727&margin=%5Bobject%20Object%5D&name=image.png&originHeight=489&originWidth=1172&originalType=binary&ratio=1&rotation=0&showTitle=false&size=48755&status=done&style=none&taskId=u7b42c2fd-3a92-4282-b9ef-c8a132e6272&title=&width=532.7272611807203)
#### （3）  在虚拟环境中安装相应的包文件，如：pandas、numpy、matplotlib、scipy、sklearn等，具体实验可具体安装。可以使用  conda list   命令查看安装了哪些包。
![image.png](https://cdn.nlark.com/yuque/0/2022/png/32555890/1668700269547-49670a6f-212b-4de1-84d4-c10db143ecbd.png#averageHue=%23181818&clientId=u8c3edc07-3917-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=512&id=u5b82a078&margin=%5Bobject%20Object%5D&name=image.png&originHeight=1126&originWidth=1301&originalType=binary&ratio=1&rotation=0&showTitle=false&size=126490&status=done&style=none&taskId=udca60ee5-72e1-4338-a234-3513c8eff3f&title=&width=591.3636235461751)
### 2．最小二乘法实现。
#### （1）   最小二乘法的 python实现：我们用目标函数𝑦=𝑠𝑖𝑛2𝜋𝑥,加上一个正态分布的噪音干扰，再用多项式去拟合（分别取 0 阶、1 阶、3 阶、9 阶进行拟合）。程序流程如下：
![image.png](https://cdn.nlark.com/yuque/0/2022/png/32555890/1668331157533-c923f3a5-792a-470f-a742-483e3ef5771f.png#averageHue=%23ebebeb&clientId=u3ecb4fef-37b1-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=133&id=u4a22f111&margin=%5Bobject%20Object%5D&name=image.png&originHeight=292&originWidth=933&originalType=binary&ratio=1&rotation=0&showTitle=false&size=53898&status=done&style=none&taskId=ue7821089-07fb-4890-b4b6-511d84e0a2c&title=&width=424.0908998989864)

写出代码和可视化结果。
目标函数：代入生成的x，生成对应的y
```python
def func(x):
return np.sin(2*np.pi*x)
```
随机生成10个x进行实验：
```python
x = np.linspace(0, 1, 10)
```
构造多项式拟合函数：
```python
#多项式
def fit_func(p,x):
"""
eg:p = np.poly1d([2,3,5,7])
print(p)==>>2x3 + 3x2 + 5x + 7
"""
f = np.poly1d(p)
return f(x)
```
计算误差：
```python
#残差
def residuals_func(p, x, y):
ret = fit_func(p, x) - y
return ret
```
leastsq 是 scipy 库 进行最小二乘法计算的函数，也就是通过误差函数以及数据点进行我们前面讲的对参数进行求导操作，最后得出我们拟合出来的函数。
```python
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
	plt.plot(x_points, func(x_points), label='real')
	plt.plot(x_points, fit_func(p_lsq[0], x_points), label='fitted curve')
	plt.plot(x, y, 'bo', label='noise')
	plt.legend()
	return p_lsq

   # M=0
p_lsq = fitting(M=0)
```
我们从一次函数依次增加项式，找到最合适的拟合曲线。到9次的时候，已经过拟合这些点了 。
![image.png](https://cdn.nlark.com/yuque/0/2022/png/32555890/1668331267551-67477c92-7fe1-41b9-b3c2-7718821b4075.png#averageHue=%23faf9f8&clientId=u3ecb4fef-37b1-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=218&id=u855b1cf5&margin=%5Bobject%20Object%5D&name=image.png&originHeight=480&originWidth=644&originalType=binary&ratio=1&rotation=0&showTitle=false&size=99139&status=done&style=none&taskId=u4d58a56a-b980-4331-9775-9295cc3a910&title=&width=292.7272663825801)
#### （2）   过拟合的情况下，引入正则化项进行优化实现，写出代码和可视化结果。
```python
def residuals_func_regularization_l1(p, x, y):
    """L1正则化"""
    ret = fit_func(p, x) - y
    ret = np.append(ret, np.sqrt(regularization*abs(p))) # L1范数作为正则化项
    return ret


def residuals_func_regularization_l2(p, x, y):
    """L2正则化"""
    ret = fit_func(p, x) - y
    ret = np.append(ret, np.sqrt(0.5*regularization*np.square(p))) # L2范数作为正则化项
    return ret

```

加入了正则化的最小二乘法
```python
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

```

![image.png](https://cdn.nlark.com/yuque/0/2022/png/32555890/1668700964247-c8c1368d-7d0d-48cc-95a1-7b4b59cad3f4.png#averageHue=%23fcfcfb&clientId=u8c3edc07-3917-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=348&id=u13f9c854&margin=%5Bobject%20Object%5D&name=image.png&originHeight=1826&originWidth=3072&originalType=binary&ratio=1&rotation=0&showTitle=false&size=171379&status=done&style=none&taskId=u6a9182ce-9a07-4a9d-9859-3fed98ef1c3&title=&width=584.6448364257812)
可视化结果
## 四、实验安全事项
实验过程中注意用电安全。


## 五、实验提交方式
√ 实验报告      □现场打分      □线上平台提交









# 实验二 线性分类模型
## 一、实验目的
1. 掌握感知机原理及实现。
2. 掌握 K 近邻基本思想和基本方法（重难点）。
3. 了解朴素贝叶斯原理及实现。
## 二、实验原理及说明
### 1. 感知机
感知机是根据输入实例的特征向量𝑥𝑥x 对其进行二类分类的线性分类模型：
$𝑓(𝑥)=sign(𝑤⋅𝑥+𝑏)$
感知机模型对应于输入空间（特征空间）中的分离超平面𝑤⋅𝑥+𝑏=0
感知机学习的策略是极小化损失函数：
$min𝐿(𝑤,𝑏)=−∑y_i(w⋅x_i+b)$，其中损失函数对应于误分类点到分离超平面的总距离。感知机学习算法是基于随机梯度下降法的对损失函数的最优化算法，有原始形式和对偶形式。算法简单且易于实现。原始形式中，首先任意选取一个超平面，然后用梯度下降法不断极小化目标函数。在这个过程中一次随机选取一个误分类点使其梯度下降。
$𝑤=𝑤+𝜂y_𝑖𝑥_i$
$b=b+𝜂y_i$
当实例点被误分类，即位于分离超平面的错误侧，则调整 w, b 的值，使分离超平面向该无分类点的一侧移动，直至误分类点被正确分类。
### 2. K 近邻
k 近邻法是基本且简单的分类与回归方法。k 近邻法的基本做法是：对给定的训练实例点和输入实例点，首先确定输入实例点的 k 个最近邻训练实例点，然后利用这 k个训练实例点的类的多数来预测输入实例点的类。
k 近邻模型对应于基于训练数据集对特征空间的一个划分。k 近邻法中，当训练集、距离度量、k 值及分类决策规则确定后，其结果唯一确定。
k 近邻法三要素：距离度量、k 值的选择和分类决策规则。常用的距离度量是欧氏距离及更一般的 pL 距离。k 值小时，k 近邻模型更复杂；k 值大时，k 近邻模型更简单。k 值的选择反映了对近似误差与估计误差之间的权衡，通常由交叉验证选择最优的 k。常用的分类决策规则是多数表决，对应于经验风险最小化。
k 近邻法的实现需要考虑如何快速搜索 k 个最近邻点。kd 树是一种便于对 k 维空间中的数据进行快速检索的数据结构。kd 树是二叉树，表示对 k 维空间的一个划分，其每个结点对应于 k 维空间划分中的一个超矩形区域。利用 kd 树可以省去对大部分数据点的搜索，从而减少搜索的计算量。
### 3. 朴素贝叶斯
朴素贝叶斯法是典型的生成学习方法。生成方法由训练数据学习联合概率分
布 P(X,Y)，然后求得后验概率分布 P(Y|X)。具体来说，利用训练数据学习 P(X|Y)和 P(Y)的估计，得到联合概率分布：P(X,Y)＝P(Y)P(X|Y)其中，概率估计方法可以是极大似然估计或贝叶斯估计。朴素贝叶斯法的基本假设是条件独立性:
$P(X=x|Y=c_k)=P(X^{(1)}=x^{(1)},...,X^{(n)}=x^{(n)}|Y=c_k)=\prod \limits_{j=0}^nP(X^{(j)}=x^{(j)}|Y=c_k)$
 这是一个较强的假设。由于这一假设，模型包含的条件概率的数量大为减少，朴素 贝叶斯法的学习与预测大为简化。因而朴素贝叶斯法高效，且易于实现。其缺点是分类 的性能不一定很高。 朴素贝叶斯法利用贝叶斯定理与学到的联合概率模型进行分类预测  :
$P(Y|X)= \frac{P(Y)P(X|Y)}{\sum_YP(Y)P(X|Y)}$
 将输入 x 分到后验概率最大的类 y：  
$y=\arg\max_{c_k}{P(Y=c_k)}\prod \limits_{j=i}^n{P(X_j=x^{(j)}|Y=c_k)}$
 后验概率最大等价于 0-1 损失函数时的期望风险最小化。  
## 三、实验内容
### 1．利用鸢尾花数据集构建感知机模型。
#### （1）IRIS 数据集也称作鸢尾花数据集，整个数据集共有 150 条数据，分为三类，每类50 条数据，每一条数据都有四个属性：花萼长度，花萼宽度，花瓣长度，花瓣宽度，标签数据共有三种，分别是 Setosa，Versicolour，Virginica。学会导入数据集，并作数据的预处理。

```python
# load data
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['label'] = iris.target

df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']

```
导入数据并选取数据
```python
def draw_data():
    plt.scatter(df[:50]['sepal length'], df[:50]['sepal width'], label='0')
    plt.scatter(df[50:100]['sepal length'], df[50:100]['sepal width'], label='1')
    plt.scatter(df[100:150]['sepal length'], df[100:150]['sepal width'], label='2')
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')
    plt.legend()
    plt.show()
```
查看数据
```python
data = np.array(df.iloc[:100, [0, 1, -1]])  # 前100条数据，取第一列 第二列 第三列
X, y = data[:,:-1], data[:,-1]  # 切片 取前两列   取后一列
y = np.array([1 if i == 1 else -1 for i in y])  # 转换成1和-1两类

```
划分数据
#### （2）用 python 构建感知机模型，并进行可视化。
```python
class Model:
    def __init__(self):
        self.w = np.ones(len(data[0])-1, dtype=np.float32)
        self.b = 0
        self.l_rate = 0.1
        # self.data = data
    
    def sign(self, x, w, b):
        y = np.dot(x, w) + b
        return y
    
    # 随机梯度下降法
    def fit(self, X_train, y_train):
        is_wrong = False
        while not is_wrong:
            wrong_count = 0
            for d in range(len(X_train)):
                X = X_train[d]
                y = y_train[d]
                if y * self.sign(X, self.w, self.b) <= 0:
                    self.w = self.w + self.l_rate*np.dot(y, X)
                    self.b = self.b + self.l_rate*y
                    wrong_count += 1
            if wrong_count == 0:
                is_wrong = True
        return 'Perceptron Model!'
        
    def score(self):
        pass

```
感知机模型构建

```python
perceptron = Model() # 建立模型
perceptron.fit(X, y)  # 模型拟合数据、训练


x_points = np.linspace(4, 7,10)
y_ = -(perceptron.w[0]*x_points + perceptron.b)/perceptron.w[1]
# 预测结束
```
利用感知机进行分类
```python

plt.plot(x_points, y_)

plt.plot(data[:50, 0], data[:50, 1], 'bo', color='blue', label='0')
plt.plot(data[50:100, 0], data[50:100, 1], 'bo', color='orange', label='1')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()
plt.show()

```
可视化
![image.png](https://cdn.nlark.com/yuque/0/2022/png/32555890/1668731853891-e917a63b-00d6-4445-895a-9b94743ac4e8.png#averageHue=%23faf9f7&clientId=u8c3edc07-3917-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=496&id=u6292edd5&margin=%5Bobject%20Object%5D&name=image.png&originHeight=1092&originWidth=1284&originalType=binary&ratio=1&rotation=0&showTitle=false&size=58563&status=done&style=none&taskId=uf2b4e368-bf4a-4a08-bb18-cfaefcf51cc&title=&width=583.6363509863864)
可视化结果
#### （3）利用 sklearn 库进行感知机模型的构建和可视化。
调包比较简单，只需要引用sklearn库中的感知机模型建立模型即可
```python
from sklearn.linear_model import Perceptron
clf = Perceptron()  # 使用默认参数
clf.fit(X, y)
print(clf.coef_) # 截距 Constants in decision function.
print(clf.intercept_)
x_ponits = np.arange(4, 8)
y_ = -(clf.coef_[0][0]*x_ponits + clf.intercept_)/clf.coef_[0][1]
# 得到结果

```
可视化结果：
![image.png](https://cdn.nlark.com/yuque/0/2022/png/32555890/1668732065047-f5d562ff-c503-4a56-9443-452a2568de90.png#averageHue=%23faf9f7&clientId=u8c3edc07-3917-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=496&id=u5d2c2738&margin=%5Bobject%20Object%5D&name=image.png&originHeight=1092&originWidth=1284&originalType=binary&ratio=1&rotation=0&showTitle=false&size=56099&status=done&style=none&taskId=u3ecfc9ad-5624-48f7-b894-56593411425&title=&width=583.6363509863864)
### 2．利用 K 近邻法进行鸢尾花类别的预测。
#### （1）K 近邻的 python 实现：建立一个类 KNN，进行模型的构建和可视化。
构建k近邻模型
```python
class KNN:
    def __init__(self, X_train, y_train, n_neighbors=3, p=2):
        """
        parameter: n_neighbors 临近点个数
        parameter: p 距离度量
        """
        self.n = n_neighbors
        self.p = p
        self.X_train = X_train
        self.y_train = y_train
    
    def predict(self, X):
        # 取出n个点
        knn_list = []
        for i in range(self.n):
            dist = np.linalg.norm(X - self.X_train[i], ord=self.p)
            knn_list.append((dist, self.y_train[i]))
            
        for i in range(self.n, len(self.X_train)):
            max_index = knn_list.index(max(knn_list, key=lambda x: x[0]))
            dist = np.linalg.norm(X - self.X_train[i], ord=self.p)
            if knn_list[max_index][0] > dist:
                knn_list[max_index] = (dist, self.y_train[i])
                
        # 统计
        knn = [k[-1] for k in knn_list]
        count_pairs = Counter(knn)
        max_count = sorted(count_pairs, key=lambda x:x)[-1]
        return max_count
    
    def score(self, X_test, y_test):
        right_count = 0
        n = 10
        for X, y in zip(X_test, y_test):
            label = self.predict(X)
            if label == y:
                right_count += 1
        return right_count / len(X_test)

clf = KNN(X_train, y_train)

clf.score(X_test, y_test)
```
测试模型：
```python
test_point = [6.0, 3.0]
print('Test Point: {}'.format(clf.predict(test_point)))
```
可视化：
```python
plt.scatter(df[:50]['sepal length'], df[:50]['sepal width'], label='0')
plt.scatter(df[50:100]['sepal length'], df[50:100]['sepal width'], label='1')
plt.plot(test_point[0], test_point[1], 'bo', label='test_point')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()
plt.show()
```
#### （2）利用 sklearn 库实现 KNN 模型的构建和可视化。
导入包
```python
from sklearn.neighbors import KNeighborsClassifier
```

1. 先选取两个特征，和两个类进行二分类：（原本数据集里花的特征有四个这里取两个，原本数据集里花有三类，这里取两类）
```python
def classify_2():  
    """二分类"""
    # data
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    # data = np.array(df.iloc[:100, [0, 1, -1]])

    """预处理结束"""

    data = np.array(df.iloc[:100, [0, 1, -1]])   # 取第一 第二 （前两个是特征）倒数第一（这个是标签，即是训练集的label）
    X, y = data[:,:-1], data[:,-1] # X是两个特征   y是0 1这样的label信息
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) # 划分训练集和验证集（这里应该是作为测试集）

    clf_sk = KNeighborsClassifier() # 构建模型
    clf_sk.fit(X_train, y_train)
    clf_sk.score(X_test, y_test)  # 评估模型

    # 预测 我自己随便给的两个数据

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

```
输出结果：
> Test Point: 1.0

即是分类的结果为第二类。
可视化结果：
![image.png](https://cdn.nlark.com/yuque/0/2022/png/32555890/1668734197057-d60c9a6a-a358-45e8-b727-7824de6aaa3f.png#averageHue=%23f9f8f7&clientId=u8c3edc07-3917-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=496&id=uc4c253bc&margin=%5Bobject%20Object%5D&name=image.png&originHeight=1092&originWidth=1284&originalType=binary&ratio=1&rotation=0&showTitle=false&size=48055&status=done&style=none&taskId=ubcd98b71-4aeb-4eef-8566-026744551c8&title=&width=583.6363509863864)

2. 然后尝试增加一个类预测：（原本数据集里花有三类，这里取三类）
```python
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
    X, y = data[:,:-1], data[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    clf_sk = KNeighborsClassifier()
    clf_sk.fit(X_train, y_train)
    clf_sk.score(X_test, y_test)

    # # 预测 两个数据

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
```
输出结果：
> Test Point: [1. 1. 0.]

可视化结果：
![image.png](https://cdn.nlark.com/yuque/0/2022/png/32555890/1668734274567-305c1849-65eb-42bd-925a-4a0caa9d1779.png#averageHue=%23f8f7f7&clientId=u8c3edc07-3917-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=496&id=u202606d5&margin=%5Bobject%20Object%5D&name=image.png&originHeight=1092&originWidth=1284&originalType=binary&ratio=1&rotation=0&showTitle=false&size=58662&status=done&style=none&taskId=ue5a7a5da-d49e-4650-bb40-39f7bab51e8&title=&width=583.6363509863864)

3. 取数据集里的四个特征，用knn进行三分类：
```python
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
    print(data)
    X, y = data[:,:-1], data[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    clf_sk = KNeighborsClassifier()
    clf_sk.fit(X_train, y_train)
    print("预测准确度为：{}".format(clf_sk.score(X_test, y_test)))
    # 四维预测没问题，但是画图咱这三维空间还是算了，如果硬是要画图，可以先聚类，四类变3类或者2类，再分类并画图
    # # 预测 三个数据

    test_point = [[6.0, 3.0,3.0,2.0],[5.0,2.0,3.0,3.0],[6,4,2,2]]
    print('Test Point: {}'.format(clf_sk.predict(test_point)))
    # 四维预测没问题，但是画图咱这三维空间还是算了，如果硬是要画图，可以先聚类，四类变3类或者2类，再分类并画图
    
```
预测结果：
> 模型预测准确度为：0.9666666666666667
> Test Point: [1. 1. 0.]

在 sklearn 库中，KNeighborsClassifier 是实现 K 近邻算法的一个类，一般都使用欧
式距离进行测量。这个类的结构如下： class sklearn.neighbors. KNeighborsClassifier 
( n_neighbors=5, weights=’uniform’, algorithm=’auto’, leaf_size=30, p=2, 
metric=’minkowski’, metric_params=None, n_jobs=1, **kwargs ) n_neighbors：就是选取最
近的点的个数：k leaf_size：这个是构造树的大小，值一般选取默认值即可，太大会影
响速度。 n_jobs ：默认值 1，选取-1 占据 CPU 比重会减小，但运行速度也会变慢，所
有的 core 都会运行。 algorithm: 近邻算法，可选{'auto', 'ball_tree', 'kd_tree', 'brute'}。
#### （3）构造平衡 Kd 树，并实现书中例题 3.2。
```python
class KdNode(object):
    def __init__(self, dom_elt, split, left, right):
        self.dom_elt = dom_elt  # k维向量节点(k维空间中的一个样本点)
        self.split = split      # 整数（进行分割维度的序号）
        self.left = left        # 该结点分割超平面左子空间构成的kd-tree
        self.right = right      # 该结点分割超平面右子空间构成的kd-tree
```
创建kd树代码：
```python
class KdTree(object):
    def __init__(self, data):
 
        def CreateNode(split, data_set):  # 按第split维划分数据集exset创建KdNode
            if not data_set:              # 数据集为空
                return None
            # key参数的值为一个函数，此函数只有一个参数且返回一个值用来进行比较
            # operator模块提供的itemgetter函数用于获取对象的哪些维的数据，参数为需要获取的数据在对象中的序号
            data_set.sort(key=operator.itemgetter(split)) # 按要进行分割的那一维数据排序  从小到大排序
            #data_set.sort(key=lambda x: x[split])
            split_pos = len(data_set) // 2     # //为Python中的整数除法
            median = data_set[split_pos]       # 中位数分割点
            leftdata=data_set[:split_pos]
            rightdata=data_set[split_pos+1:]
            split_next1=np.argmax(np.var(leftdata,0))
            split_next2=np.argmax(np.var(rightdata,0))
            # 递归的创建kd树
            return KdNode(median, split, CreateNode(split_next1,leftdata), CreateNode(split_next2, rightdata))
                                           
 
        self.root = CreateNode(0, data)  # 从第0维分量开始构建kd树,返回根节点
```
书上题3.2：
给定一个二维空间的数据集：
T={(2,3)T,(5,4)T,(9,6)T,(4,7)T,(8,1)T,(7,2)T}构造一个平衡kd树
```python
# KDTree的前序遍历
def preorder(root):
    print(root.dom_elt)
    if root.left:  # 节点不为空
        preorder(root.left)
    if root.right:
        preorder(root.right)
 
 
if __name__ == "__main__":
    data = [[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]]
    kd = KdTree(data)
    preorder(kd.root)
```
构造完成
![image.png](https://cdn.nlark.com/yuque/0/2022/png/32555890/1669707442831-e3b5293f-7228-4a47-9e62-d3f5dde4462c.png#averageHue=%230e0021&clientId=u1554ef67-f7a1-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=131&id=u14b2b9fa&margin=%5Bobject%20Object%5D&name=image.png&originHeight=289&originWidth=1093&originalType=binary&ratio=1&rotation=0&showTitle=false&size=21175&status=done&style=none&taskId=u43683d98-7613-4747-9aa4-cc5cc79d30b&title=&width=496.818171049938)
### 3. 在鸢尾花数据集上用高斯朴素贝叶斯实现分类，并进行预测。直接使用 sklearn 方法。
手写代码（高斯朴素贝叶斯）：
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
# 这个不算是包的内容，只是用来获取数据集和划分数据集用的
from collections import Counter
import math

```
数据预处理：
```python
# data
def create_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    data = np.array(df.iloc[:100, :])
    # print(data)
    return data[:,:-1], data[:,-1]


X, y = create_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
X_test[0], y_test[0]
```

```python
class NaiveBayes:
    def __init__(self):
        self.model = None

    # 数学期望
    @staticmethod
    def mean(X):
        return sum(X) / float(len(X))

    # 标准差（方差）
    def stdev(self, X):
        avg = self.mean(X)
        return math.sqrt(sum([pow(x-avg, 2) for x in X]) / float(len(X)))

    # 概率密度函数
    def gaussian_probability(self, x, mean, stdev):
        exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
        return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent

    # 处理X_train
    def summarize(self, train_data):
        summaries = [(self.mean(i), self.stdev(i)) for i in zip(*train_data)]
        return summaries

    # 分类别求出数学期望和标准差
    def fit(self, X, y):
        labels = list(set(y))
        data = {label:[] for label in labels}
        for f, label in zip(X, y):
            data[label].append(f)
        self.model = {label: self.summarize(value) for label, value in data.items()}
        return 'gaussianNB train done!'

    # 计算概率
    def calculate_probabilities(self, input_data):
        # summaries:{0.0: [(5.0, 0.37),(3.42, 0.40)], 1.0: [(5.8, 0.449),(2.7, 0.27)]}
        # input_data:[1.1, 2.2]
        probabilities = {}
        for label, value in self.model.items():
            probabilities[label] = 1
            for i in range(len(value)):
                mean, stdev = value[i]
                probabilities[label] *= self.gaussian_probability(input_data[i], mean, stdev)
        return probabilities

    # 类别
    def predict(self, X_test):
        # {0.0: 2.9680340789325763e-27, 1.0: 3.5749783019849535e-26}
        label = sorted(self.calculate_probabilities(X_test).items(), key=lambda x: x[-1])[-1][0]
        return label

    def score(self, X_test, y_test):
        right = 0
        for X, y in zip(X_test, y_test):
            label = self.predict(X)
            if label == y:
                right += 1

        return right / float(len(X_test))

odel = NaiveBayes()
# define
model.fit(X_train, y_train)
# fit
print(model.predict([4.4,  3.2,  1.3,  0.2]))
# 预测
model.score(X_test, y_test)
```
输出：
```python

0
```
即是第一类。
同上，调包实现会更简单：
```python
from sklearn.naive_bayes import GaussianNB
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from collections import Counter
import math

# data
def create_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    data = np.array(df.iloc[:100, :])
    # print(data)
    return data[:,:-1], data[:,-1]

X, y = create_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
X_test[0], y_test[0]
"""数据预处理结束"""


clf = GaussianNB()
clf.fit(X_train, y_train)
clf.score(X_test, y_test)
print(clf.predict([[4.4,  3.2,  1.3,  0.2]]))
```
输出：
```python
[0,]
```
即是第一类。
## 四、实验安全事项
实验过程中注意用电安全。 
## 五、实验提交方式 
√ 实验报告 □ 现场打分 □ 线上平台提交  

# 实验三 决策树与随机森林
## 一、实验目的

1. 掌握决策树原理及实现。
2. 掌握 ID3 算法实现（重点）。
3. 掌握 sklearn 库中决策树的实现方法（CART 树算法）。
4. 了解随机森林原理及实现（加分项）。
## 二、实验原理及说明

1. 决策树
决策树（decision tree）：是一种基本的分类与回归方法，此处主要讨论分类的决策树。在分类问题中，表示基于特征对实例进行分类的过程，可以认为是 if-then 的集合，也可以认为是定义在特征空间与类空间上的条件概率分布。
决策树通常有三个步骤：特征选择、决策树的生成、决策树的修剪。
用决策树分类：从根节点开始，对实例的某一特征进行测试，根据测试结果将实例分配到其子节点，此时每个子节点对应着该特征的一个取值，如此递归的对实例进行测试并分配，直到到达叶节点，最后将实例分到叶节点的类中。
决策树的构造
决策树学习的算法通常是一个递归地选择最优特征，并根据该特征对训练数据进行分割，使得各个子数据集有一个最好的分类的过程。这一过程对应着对特征空间的划分，也对应着决策树的构建。
1） 开始：构建根节点，将所有训练数据都放在根节点，选择一个最优特征，按着这一特征将训练数据集分割成子集，使得各个子集有一个在当前条件下最好的分类。
2） 如果这些子集已经能够被基本正确分类，那么构建叶节点，并将这些子集分到所对应的叶节点去。
3）如果还有子集不能够被正确的分类，那么就对这些子集选择新的最优特征，继续对其进行分割，构建相应的节点，如果递归进行，直至所有训练数据子集被基本正确的分类，或者没有合适的特征为止。
4）每个子集都被分到叶节点上，即都有了明确的类，这样就生成了一颗决策树。

2. 随机森林
尽管有剪枝等方法，一棵树的生成肯定还是不如多棵树，因此就有了随机森林，解决决策树泛化能力弱的缺点。
而同一批数据，用同样的算法只能产生一棵树，这时 Bagging 策略可以帮助我们产生不同的数据集。Bagging 策略来源于 bootstrap aggregation：从样本集（假设样本集 N个数据点）中重采样选出 Nb 个样本（有放回的采样，样本数据点个数仍然不变为 N），在所有样本上，对这 n 个样本建立分类器（ID3\C4.5\CART\SVM\LOGISTIC），重复以上两步 m 次，获得 m 个分类器，最后根据这 m 个分类器的投票结果，决定数据属于哪一类。
随机森林在 bagging 的基础上更进一步：
3. 样本的随机：从样本集中用 Bootstrap 随机选取 n 个样本
4. 特征的随机：从所有属性中随机选取 K 个属性，选择最佳分割属性作为节点建立 CART 决策树（泛化的理解，这里面也可以是其他类型的分类器，比如 SVM、Logistics）
5. 重复以上两步 m 次，即建立了 m 棵 CART 决策树
6. 这 m 个 CART 形成随机森林，通过投票表决结果，决定数据属于哪一类（投票机制有一票否决制、少数服从多数、加权多数）
## 三、实验内容
### 1. ID3 算法实现。
#### （1）编写代码计算信息增益，数据集为教材 71 页表 5.1。
数据集构造：
```python

datasets = [['青年', '否', '否', '一般', '否'],
               ['青年', '否', '否', '好', '否'],
               ['青年', '是', '否', '好', '是'],
               ['青年', '是', '是', '一般', '是'],
               ['青年', '否', '否', '一般', '否'],
               ['中年', '否', '否', '一般', '否'],
               ['中年', '否', '否', '好', '否'],
               ['中年', '是', '是', '好', '是'],
               ['中年', '否', '是', '非常好', '是'],
               ['中年', '否', '是', '非常好', '是'],
               ['老年', '否', '是', '非常好', '是'],
               ['老年', '否', '是', '好', '是'],
               ['老年', '是', '否', '好', '是'],
               ['老年', '是', '否', '非常好', '是'],
               ['老年', '否', '否', '一般', '否'],
               ]
labels = [u'年龄', u'有工作', u'有自己的房子', u'信贷情况', u'类别']
```
按照公式写出计算信息增益的代码：
```python

# 熵
def calc_ent(datasets):
    data_length = len(datasets)
    label_count = {}
    for i in range(data_length):
        label = datasets[i][-1]
        if label not in label_count:
            label_count[label] = 0
        label_count[label] += 1
    ent = -sum([(p/data_length)*log(p/data_length, 2) for p in label_count.values()])
    return ent

# 经验条件熵
def cond_ent(datasets, axis=0):
    data_length = len(datasets)
    feature_sets = {}
    for i in range(data_length):
        feature = datasets[i][axis]
        if feature not in feature_sets:
            feature_sets[feature] = []
        feature_sets[feature].append(datasets[i])
    cond_ent = sum([(len(p)/data_length)*calc_ent(p) for p in feature_sets.values()])
    return cond_ent

# 信息增益
def info_gain(ent, cond_ent):
    return ent - cond_ent

```
获取信息增益：
```python
def info_gain_train(datasets):
    count = len(datasets[0]) - 1
    ent = calc_ent(datasets)
    best_feature = []
    for c in range(count):
        c_info_gain = info_gain(ent, cond_ent(datasets, axis=c))
        best_feature.append((c, c_info_gain))
        print('特征({}) - info_gain - {:.3f}'.format(labels[c], c_info_gain))
    # 比较大小
    best_ = max(best_feature, key=lambda x: x[-1])
    return '特征({})的信息增益最大，选择为根节点特征'.format(labels[best_[0]])


info_gain_train(np.array(datasets))
```
输出结果：
![image.png](https://cdn.nlark.com/yuque/0/2022/png/32555890/1669707672040-1ad03759-9479-4b6f-82bf-dd110dc6dcb3.png#averageHue=%230f0123&clientId=u1554ef67-f7a1-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=71&id=u8b028dea&margin=%5Bobject%20Object%5D&name=image.png&originHeight=157&originWidth=624&originalType=binary&ratio=1&rotation=0&showTitle=false&size=31089&status=done&style=none&taskId=ua1d9388f-7e35-427d-b266-1703c5007bc&title=&width=283.63635748871116)
#### （2）用 python 编写 ID3 算法。
书接上回：
```python
class DTree:
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon
        self._tree = {}

    # 熵
    @staticmethod
    def calc_ent(datasets):
        data_length = len(datasets)
        label_count = {}
        for i in range(data_length):
            label = datasets[i][-1]
            if label not in label_count:
                label_count[label] = 0
            label_count[label] += 1
        ent = -sum([(p/data_length)*log(p/data_length, 2) for p in label_count.values()])
        return ent

    # 经验条件熵
    def cond_ent(self, datasets, axis=0):
        data_length = len(datasets)
        feature_sets = {}
        for i in range(data_length):
            feature = datasets[i][axis]
            if feature not in feature_sets:
                feature_sets[feature] = []
            feature_sets[feature].append(datasets[i])
        cond_ent = sum([(len(p)/data_length)*self.calc_ent(p) for p in feature_sets.values()])
        return cond_ent

    # 信息增益
    @staticmethod
    def info_gain(ent, cond_ent):
        return ent - cond_ent

    def info_gain_train(self, datasets):
        count = len(datasets[0]) - 1
        ent = self.calc_ent(datasets)
        best_feature = []
        for c in range(count):
            c_info_gain = self.info_gain(ent, self.cond_ent(datasets, axis=c))
            best_feature.append((c, c_info_gain))
        # 比较大小
        best_ = max(best_feature, key=lambda x: x[-1])
        return best_

    def train(self, train_data):
        """
        input:数据集D(DataFrame格式)，特征集A，阈值eta
        output:决策树T
        """
        _, y_train, features = train_data.iloc[:, :-1], train_data.iloc[:, -1], train_data.columns[:-1]
        # 1,若D中实例属于同一类Ck，则T为单节点树，并将类Ck作为结点的类标记，返回T
        if len(y_train.value_counts()) == 1:
            return Node(root=True,
                        label=y_train.iloc[0])

        # 2, 若A为空，则T为单节点树，将D中实例树最大的类Ck作为该节点的类标记，返回T
        if len(features) == 0:
            return Node(root=True, label=y_train.value_counts().sort_values(ascending=False).index[0])

        # 3,计算最大信息增益 同5.1,Ag为信息增益最大的特征
        max_feature, max_info_gain = self.info_gain_train(np.array(train_data))
        max_feature_name = features[max_feature]

        # 4,Ag的信息增益小于阈值eta,则置T为单节点树，并将D中是实例数最大的类Ck作为该节点的类标记，返回T
        if max_info_gain < self.epsilon:
            return Node(root=True, label=y_train.value_counts().sort_values(ascending=False).index[0])

        # 5,构建Ag子集
        node_tree = Node(root=False, feature_name=max_feature_name, feature=max_feature)

        feature_list = train_data[max_feature_name].value_counts().index
        for f in feature_list:
            sub_train_df = train_data.loc[train_data[max_feature_name] == f].drop([max_feature_name], axis=1)

            # 6, 递归生成树
            sub_tree = self.train(sub_train_df)
            node_tree.add_node(f, sub_tree)

        # pprint.pprint(node_tree.tree)
        return node_tree

    def fit(self, train_data):
        self._tree = self.train(train_data)
        return self._tree

    def predict(self, X_test):
        return self._tree.predict(X_test)

datasets, labels = create_data()
data_df = pd.DataFrame(datasets, columns=labels)
dt = DTree()
tree = dt.fit(data_df)
print(tree)

print(dt.predict(['老年', '否', '否', '一般']))

```
输出结果：
![image.png](https://cdn.nlark.com/yuque/0/2022/png/32555890/1669712673931-01fe4dc3-9a84-4708-92e8-0f37a1d9d189.png#averageHue=%230e0022&clientId=u1554ef67-f7a1-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=93&id=ud51320c2&margin=%5Bobject%20Object%5D&name=image.png&originHeight=205&originWidth=1124&originalType=binary&ratio=1&rotation=0&showTitle=false&size=24844&status=done&style=none&taskId=u6eaea2c1-7e4d-4ddf-b9c9-9a31c5b0704&title=&width=510.90907983543485)
### 2. 使用 sklearn 库，对鸢尾花数据建立决策树，并进行可视化。
#### （1）scikit-learn 决策树算法类库内部实现是使用了调优过的 CART 树算法，既可以做分类，又可以做回归。分类决策树的类对应的是 DecisionTreeClassifier，而回归决策树的类对应的是 DecisionTreeRegressor。两者的参数定义几乎完全相同，但是意义不全相同。在 [https://graphviz.org/download/](https://graphviz.org/download/) 上下载 graphviz,安装时注意勾选加环境变量，在 anaconda prompt 中 pip install graphviz ，再 pip install pydotplus。
这个略了，安装就完事儿了
#### （2）使用 DecisionTreeClassifier()构建决策树并进行可视化。
包的调用
```python
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

```

数据预处理：
```python
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
```
决策树可视化：
```python
datasets, labels = create_data()
train_data = pd.DataFrame(datasets, columns=[0,1])

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train,)
print(clf.score(X_test, y_test))
tree_pic = export_graphviz(clf, out_file="mytree.pdf")
with open('mytree.pdf') as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)
```
输出结果显示：
![image.png](https://cdn.nlark.com/yuque/0/2022/png/32555890/1669713848073-3fe3aa92-ab1a-4053-bc22-f17784ce9bf9.png#averageHue=%23100224&clientId=u1554ef67-f7a1-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=16&id=uc9e71f0d&margin=%5Bobject%20Object%5D&name=image.png&originHeight=35&originWidth=336&originalType=binary&ratio=1&rotation=0&showTitle=false&size=2362&status=done&style=none&taskId=u3e6a5db2-6b8a-4a21-8035-f1b432d6619&title=&width=152.72726941699833)
![image.png](https://cdn.nlark.com/yuque/0/2022/png/32555890/1669713827478-3163388f-ef96-47d8-8b88-3c39964b422d.png#averageHue=%230e0022&clientId=u1554ef67-f7a1-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=310&id=ue9984adf&margin=%5Bobject%20Object%5D&name=image.png&originHeight=682&originWidth=1121&originalType=binary&ratio=1&rotation=0&showTitle=false&size=102969&status=done&style=none&taskId=ue4168c02-c206-431a-a62b-7f4c4ea7f25&title=&width=509.5454435013545)

### 3. 在鸢尾花数据集上使用随机森林进行分类，有条件的可进行可视化。
```python
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

```
输出结果：
![image.png](https://cdn.nlark.com/yuque/0/2022/png/32555890/1669714005610-d9b9c262-0f05-40a9-8792-18ea973acced.png#averageHue=%230e0021&clientId=u1554ef67-f7a1-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=179&id=u60861647&margin=%5Bobject%20Object%5D&name=image.png&originHeight=393&originWidth=960&originalType=binary&ratio=1&rotation=0&showTitle=false&size=43071&status=done&style=none&taskId=u6baaecd3-17f5-405b-a825-e9652e80633&title=&width=436.36362690570945)
```python
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
z_show = np.stack((xx.flat, yy.flat), axis=1)  # 测试点
# z = RF.predict(np.c_[xx.ravel(), yy.ravel()])
z = RF.predict(z_show)
plt.figure()
plt.pcolormesh(xx, yy, z.reshape(xx.shape), shading='auto', cmap=cmap_light)
plt.scatter(x[:, 0], x[:, 1], c=y, cmap=cmap_bold, edgecolors='k', s=20)
plt.xlim(xx.min(), xx.max())
plt.title('RandomForestClassifier')
plt.show()
# print('RandomForestClassifier:', RF.score(x, y))
```
输出图像：
![image.png](https://cdn.nlark.com/yuque/0/2022/png/32555890/1669713982282-738c1ab0-056a-49ed-973c-c9039c690382.png#averageHue=%23d0a159&clientId=u1554ef67-f7a1-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=491&id=u99375304&margin=%5Bobject%20Object%5D&name=image.png&originHeight=1081&originWidth=1258&originalType=binary&ratio=1&rotation=0&showTitle=false&size=62162&status=done&style=none&taskId=u3d5ba88e-a875-4ea0-9067-2de2ad44586&title=&width=571.8181694243568)

##  四、实验安全事项 
实验过程中注意用电安全。
##  五、实验提交方式
 √ 实验报告 □ 现场打分 □ 线上平台提交  

# 实验四 支持向量机
## 一、实验目的

### 1. 掌握线性可分支持向量机的原理及实现。
### 2. 掌握 sklearn 库中支持向量机的实现方法。
## 二、实验原理及说明
支持向量机最简单的情况是线性可分支持向量机，或硬间隔支持向量机。构建它的条件是训练数据线性可分。其学习策略是最大间隔法。可以表示为凸二次规划问题，其原始最优化问题为
![image.png](https://cdn.nlark.com/yuque/0/2022/png/32555890/1668350828142-d8440d02-430f-496f-b2da-666d4abf33e3.png#averageHue=%23f9f9f9&clientId=u385df243-09b6-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=80&id=ue7373b88&margin=%5Bobject%20Object%5D&name=image.png&originHeight=177&originWidth=786&originalType=binary&ratio=1&rotation=0&showTitle=false&size=22082&status=done&style=none&taskId=u7a06dfb1-e590-4a14-b3c0-762d8445480&title=&width=357.2727195290496)
 求得最优化问题的解为 w∗ ，𝑏𝑏∗ ，得到线性可分支持向量机，分离超平面是  
![image.png](https://cdn.nlark.com/yuque/0/2022/png/32555890/1668350846487-9845d5ce-83a9-44f5-a93f-20897d2d47ba.png#averageHue=%23f4f4f4&clientId=u385df243-09b6-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=26&id=ueaae46b9&margin=%5Bobject%20Object%5D&name=image.png&originHeight=58&originWidth=272&originalType=binary&ratio=1&rotation=0&showTitle=false&size=5537&status=done&style=none&taskId=uce7d6a5c-fcc5-4d6c-97bc-07d68c323b6&title=&width=123.63636095661768)
分类决策函数是  
![image.png](https://cdn.nlark.com/yuque/0/2022/png/32555890/1668350897772-6b5925d6-b995-4698-ac3c-7a2ba91a36ff.png#averageHue=%23f3f3f3&clientId=u385df243-09b6-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=35&id=u3cfb874d&margin=%5Bobject%20Object%5D&name=image.png&originHeight=76&originWidth=371&originalType=binary&ratio=1&rotation=0&showTitle=false&size=10870&status=done&style=none&taskId=ua3e5b529-a82e-4cd1-bd06-b76801f6d0a&title=&width=168.63635998126898)
 最大间隔法中，函数间隔与几何间隔是重要的概念。 线性可分支持向量机的最优解存在且唯一。位于间隔边界上的实例点为支持向量。最优 分离超平面由支持向量完全决定。 二次规划问题的对偶问题是  
![image.png](https://cdn.nlark.com/yuque/0/2022/png/32555890/1668350922375-fb6e08a6-9501-428a-b8e7-784a2b83e6f0.png#averageHue=%23f6f6f6&clientId=u385df243-09b6-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=133&id=u96ce10bd&margin=%5Bobject%20Object%5D&name=image.png&originHeight=293&originWidth=582&originalType=binary&ratio=1&rotation=0&showTitle=false&size=39573&status=done&style=none&taskId=u55f9ccbd-267f-4441-985c-7764753d1d9&title=&width=264.54544881158637)
 通常，通过求解对偶问题学习线性可分支持向量机，即首先求解对偶问题的最优值 𝑎∗ ，然后求最优值𝑤∗ 和𝑏∗ ，得出分离超平面和分类决策函数。  
##  三、实验内容  
###  1．使用 sklearn 库，使用鸢尾花数据集，对鸢尾花进行分类。 
（1）数据准备，载入数据集，并进行数据集分割； 
（2）模型搭建；
（3）模型训练； 
（4）模型评估； 
（5）可视化。 
### 2. 完成习题 7.2，编程实现。
##  四、实验安全事项 
实验过程中注意用电安全。 
## 五、实验提交方式 
√ 实验报告 □ 现场打分 □ 线上平台提交  
