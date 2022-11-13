#导入工具包
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv("data.csv")
#查看数据集中的列名
data.info()
#发现data['TotalCharges']中有  ' '  数据，无法将其直接转化为浮点型，
#这里用到强制转化，不能转化的设为空值，再用0填充空值
data['TotalCharges'].dtypes
data['TotalCharges'].value_counts()
# 在筛选中发现，TotalCharges对应NaN时，其MonthlyCharges和tenure如下：
set_num = data[data['TotalCharges'].isnull().values==True][['tenure','MonthlyCharges','TotalCharges']]
set_num



data.loc[data['TotalCharges'].isnull(),'TotalCharges']=data[data['TotalCharges'].isnull()]['MonthlyCharges']
set_num = data[data['tenure']==0][['tenure','MonthlyCharges','TotalCharges']]
set_num




#将入网时长的0改为1
data.loc[:,'tenure'].replace(to_replace=0,value=1,inplace=True)
#对数值型变量进行描述统计
data.describe()


plt.figure(figsize=(6, 6))
#按性别统计流失与否占比
gender_data = (data.groupby('gender')['Churn'].value_counts()/len(data)).to_frame()
gender_data.rename(columns = {'Churn': '客户占比'},inplace = True)
gender_data.reset_index(inplace = True)
gender_data
#同样可以生成按照是否为老年用户、是否为伴侣用户、是否为家属用户分类的统计表
def bar_per(feature, orient = 'v', axis_name = '客户占比'):
    ratios = pd.DataFrame()
    p = (data.groupby(feature)['Churn'].value_counts()/len(data)).to_frame()
    p.rename(columns = {'Churn': axis_name}, inplace = True)
    p.reset_index(inplace = True)
    #print(p)
    if orient == 'v':
        ax = sns.barplot(x = feature, y = axis_name,
                         hue = 'Churn', data = p, 
                         orient = orient, palette="plasma_r", alpha=0.5)
        ax.set_yticklabels(['{:,.0%}'.format(y) for y in ax.get_yticks()])
        plt.rcParams.update({'font.size': 13})
        plt.legend(fontsize = 10)
    else:
        ax = sns.barplot(x = axis_name, y = feature, 
                        hue = 'Churn', data = p, orient = orient,
                        palette="plasma_r", alpha=0.5)
        ax.set_xticklabels(['{:,.0%}'.format(x) for x in ax.get_xticks()])
        plt.legend(fontsize = 10)
    #plt.title('按照 {} 分类的客户流失与否的比例'.format(feature))
    plt.show()
bar_per('SeniorCitizen')
bar_per('gender')



#作核密度估计 Kernel density estimaton
#寻找tenure与流失与否的关系
a = data[data['Churn'] == 'No']['tenure']
b = data[data['Churn'] == 'Yes']['tenure']
plt.figure(figsize=(9, 4))
ax0 = sns.kdeplot(a, shade = 'True',color = 'g', label = '未流失')
ax1 = sns.kdeplot(b, shade = 'True',color = 'b', label = '已流失')
plt.xlabel('客户在公司停留的月数')
#plt.title('tenure密度分布图', fontsize = 20)
plt.legend(fontsize = 12)




#对于多线服务
plt.figure(figsize=(9, 4))
bar_per('MultipleLines', orient = 'h')
#对于上网服务
plt.figure(figsize=(9, 4))
bar_per('InternetService', orient = 'h')





#对于其他的服务属性都是建立在上网服务不为No的基础上，于是我们将剩下属性全部提取，并将宽数据转为长数据
cols = ["PhoneService","MultipleLines","OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
df1 = pd.melt(data[data['InternetService'] != 'No'][cols],
             value_name = 'Has service')
plt.figure(figsize=(20, 8))
ax = sns.countplot(data = df1, x = 'variable', hue = 'Has service',palette="plasma_r",alpha = 0.5)
plt.rcParams.update({'font.size':20})
ax.set(xlabel = '网络附加服务', ylabel = '客户的数量')
plt.legend(labels = ['无服务', '有服务'], fontsize = 20)
#plt.title('按照有无网络附加服务分类的客户的人数', fontsize = 30)
plt.show()
#对于其他的服务属性都是建立在上网服务不为No的基础上，于是我们将剩下属性全部提取，并将宽数据转为长数据
cols = ["PhoneService","MultipleLines","OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
df1 = pd.melt(data[(data['InternetService'] != 'No') & (data['Churn'] == 'No')][cols],
             value_name = 'Has service')
plt.figure(figsize=(20, 8))
ax = sns.countplot(data = df1, x = 'variable', hue = 'Has service',palette="plasma_r",alpha = 0.5)
plt.rcParams.update({'font.size':20})
ax.set(xlabel = '网络附加服务', ylabel = '在网客户的数量')
plt.legend(labels = ['无服务', '有服务'], fontsize = 20)
#plt.title('按照有无网络附加服务分类的在网客户的人数', fontsize = 30)
plt.show()




#对客户是否有纸质化账单和客户的合同期限作分类
g = sns.FacetGrid(data, col = 'PaperlessBilling', height = 6, aspect = .9)
ax = g.map(sns.barplot, 'Contract','churn_rate', palette="plasma_r",alpha = 0.5,
          order = ['Month-to-month', 'One year', 'Two year'])
plt.rcParams.update({'font.size': 20})
plt.show()




#客户的付款方式
plt.figure(figsize = (10, 5))
bar_per('PaymentMethod', orient = 'h')


#每月以及总收取金额的密度图
a = data[data['Churn'] == 'No']['MonthlyCharges']
b = data[data['Churn'] == 'Yes']['MonthlyCharges']
plt.figure(figsize=(9, 4))
ax0 = sns.kdeplot(a, shade = 'True',color = 'g', label = '未流失')
ax1 = sns.kdeplot(b, shade = 'True',color = 'b', label = '已流失')
plt.xlabel('每月向客户收取的金额')
#plt.title('每月收取金额密度分布图', fontsize = 20)
plt.legend(fontsize = 12)
a = data[data['Churn'] == 'No']['TotalCharges']
b = data[data['Churn'] == 'Yes']['TotalCharges']
plt.figure(figsize=(9, 4))
ax0 = sns.kdeplot(a, shade = 'True',color = 'g', label = '未流失')
ax1 = sns.kdeplot(b, shade = 'True',color = 'b', label = '已流失')
plt.xlabel('向客户收取总金额')
#plt.title('收取总金额密度分布图', fontsize = 20)
plt.legend(fontsize = 12)



#去除数据框中的ID信息和rate
cus_ID = data['customerID']
data.drop(['customerID', 'churn_rate'], axis = 1, inplace = True)
#分离出离散特征
cate_cols = [c for c in data.columns if data[c].dtype == 'object' or c == 'SeniorCitizen']
data_cate = data[cate_cols].copy()
data_cate.head(5)
for col in cate_cols:
    if data_cate[col].nunique() == 2:
        data_cate[col] = pd.factorize(data_cate[col])[0]
    else:
        data_cate = pd.get_dummies(data_cate, columns=[col])
data_cate
#添上连续变量
data_cate['tenure'] = data[['tenure']]
data_cate['MonthlyCharges'] = data[['MonthlyCharges']]
data_cate['TotalCharges'] = data[['TotalCharges']]
#查看相关性关系
plt.figure(figsize = (16, 8))
data_cate.corr()['Churn'].sort_values(ascending = False).plot(kind = 'bar')
plt.show()



from sklearn.model_selection import train_test_split,GridSearchCV #划分数据集，网络搜索
from sklearn.neighbors import NearestNeighbors
#from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier #随机森林
from sklearn.linear_model import LogisticRegression #逻辑回归
from sklearn.model_selection import cross_val_score
from sklearn.tree import  DecisionTreeClassifier #分类树
from sklearn.metrics import precision_score, accuracy_score,recall_score, classification_report,f1_score
from sklearn.model_selection import learning_curve
import random
from sklearn.linear_model import  SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import  SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BaseNB
from sklearn.neighbors import KNeighborsClassifier #k近邻算法
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
#将data数据中含有No .. service的项换成No
data['MultipleLines'].replace('No phone service', 'No', inplace = True)
col = ['InternetService','OnlineSecurity','DeviceProtection','TechSupport',
      'StreamingMovies','StreamingTV']
for i in col:
    data[i].replace('No internet service', 'No', inplace = True)
clos = data[data.columns[data.dtypes == object]].copy()  #将离散型的特征提出
for i in clos.columns:
    if clos[i].nunique() == 2:
        clos[i] = pd.factorize(clos[i])[0]
    else:
        clos = pd.get_dummies(clos, columns = [i])
clos['tenure'] = data[['tenure']]
clos['MonthlyCharges'] = data[['MonthlyCharges']]
clos['TotalCharges'] = data[['TotalCharges']]
#划分特征和标签
X = clos.iloc[:, clos.columns != 'Churn']
y = clos.iloc[:, clos.columns == 'Churn'].values.ravel()
from imblearn.over_sampling import SMOTE
# 样本平衡处理：采用的是过采样SMOTE算法，其主要原理是通过插值的方式插入近邻的数据点。
print('样本个数:{},1占:{:.2%},0占:{:.2%}'.format(X.shape[0],y.sum()/X.shape[0],(X.shape[0]-y.sum())/X.shape[0]))
smote=SMOTE(random_state=10)
over_X,over_y=smote.fit_sample(X,y)
print('样本个数:{},1占:{:.2%},0占:{:.2%}'.format(over_X.shape[0],over_y.sum()/over_X.shape[0],(over_X.shape[0]-over_y.sum())/over_X.shape[0]))
#运行结果：
# 样本个数:7043,1占:26.54%,0占:73.46%
# 样本个数:10348,1占:50.00%,0占:50.00%





#划分训练集、测试集
# 按照3:7的比例选择测试集与训练集
# random_state = 1 表示重复试验随机得到的数据集始终不变
# stratify = target 表示按标识的类别，作为训练数据集、测试数据集内部的分配比例
Xtrain, Xtest, ytrain, ytest = train_test_split(over_X, over_y, test_size = 0.30, random_state = 10)
# 采用决策树、随机森林、K近邻、逻辑回归四个模型。观察模型的准确度、精确度、召回率、f1。
model=[DecisionTreeClassifier(random_state = 120)
      ,RandomForestClassifier(random_state = 120)
      ,KNeighborsClassifier()
      ,LogisticRegression(max_iter = 1000)]
for clf in model:
    clf.fit(Xtrain, ytrain)
    #用建好的model在测试集进行验证
    y_pre = clf.predict(Xtest)
    precision = precision_score(ytest, y_pre) 
    accuracy = accuracy_score(ytest, y_pre)
    print(clf,'\n \n',classification_report(ytest,y_pre)
         ,'\n \n Precision Score:',precision
         ,'\n Accuracy Score::',accuracy
          ,'\n \n')
#运行结果：




DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='gini',
                       max_depth=None, max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, presort='deprecated',
                       random_state=120, splitter='best') 
 

RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=100,
                       n_jobs=None, oob_score=False, random_state=120,
                       verbose=0, warm_start=False) 
 

KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                     metric_params=None, n_jobs=None, n_neighbors=5, p=2,
                     weights='uniform') 
 

LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=1000,
                   multi_class='auto', n_jobs=None, penalty='l2',
                   random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
                   warm_start=False) 
figure,ax=plt.subplots(1,4,figsize=(30,15))
for i in range(4):
    train_sizes, train_scores, valid_scores=learning_curve(model[i],over_X,over_y, cv=5,random_state=10)
    train_std=train_scores.mean(axis=1)
    test_std=valid_scores.mean(axis=1)
    ax[i].plot(train_sizes,train_std,color='red',label='train_scores')
    ax[i].plot(train_sizes,test_std,color='blue',label='test_scores')
plt.legend()




param_c, param_grid=param_grid,cv=5)
gridsearch.fit(Xtrain,ytrain)
print('best_params:',gridsearch.best_params_
      ,'\n \nbest_score: ',gridsearch.best_score_)grid  = { 
                'n_estimators' : [500,1200],
#                'min_samples_split': [2,5,10,15],
#                'min_samples_leaf': [1,2,5,10],
                'max_depth': range(1,10,2),
#                 'max_features' : ('log2', 'sqrt'),
              }
rfc = RandomForestClassifier(random_state = 120)
gridsearch = GridSearchCV(estimator =rf
rfc = RandomForestClassifier(random_state = 120,
                            max_depth = 9,
                            n_estimators = 500)
rfc.fit(Xtrain, ytrain)
y_pre = rfc.predict(Xtest)
f1_score(ytest, y_pre)



# 查看特征的重要性并排序。
fea_import=pd.DataFrame(rfc.feature_importances_)
fea_import['feature']=list(Xtrain)
fea_import.sort_values(0,ascending=False,inplace=True)
fea_import.reset_index(drop=True,inplace=True)
fea_import
figuer,ax=plt.subplots(figsize=(12,8))
g=sns.barplot(0,'feature',data=fea_import)
g.set_xlabel('Weight')
g.set_title('Random Forest')


#使用上述得到的最优模型
model = RandomForestClassifier(random_state = 120,
                            max_depth = 9,
                            n_estimators = 500)
model.fit(Xtrain, ytrain)

#提取customerID
pred_id = cus_ID.tail(10)
#提取预测数据集特征
pred_x = X.tail(10)
 #预测值
pred_y = model.predict(pred_x)

#预测结果
pred_data = pd.DataFrame({'customerID':pred_id, 'Churn':pred_y})
print(pred_data)





