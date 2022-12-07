import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split #数据集划分
from sklearn.ensemble import RandomForestClassifier # 随机森林
from sklearn.metrics import confusion_matrix, classification_report #报告
from sklearn.tree import DecisionTreeClassifier #决策树
from sklearn.linear_model import LogisticRegression #逻辑回归
from sklearn.metrics import accuracy_score #精确度

def cal_is_agree(x):  # x 为每个用户的三个月值
    # 如果三个月不全为1，用第三个月值减去前两个月均值；三个月的值都为1，取值为1.5。
    # 所有取值情况为-1、-0.5、0、0.5、1、1.5
    x = np.array(x)
    if x.sum() == 3:
        return 1.5
    else:
        return x[2] - x[:2].mean()

data = pd.read_csv("USER_INFO_M.csv", encoding='gbk')
data.drop_duplicates(inplace=True)  # 数据去重
data.drop(['MANU_NAME', 'MODEL_NAME', 'OS_DESC', 'CONSTELLATION_DESC'], axis=1, inplace=True)
cleardata = data;
data_group = cleardata.groupby('USER_ID')  # 分组


label = data_group[['USER_ID', 'IS_LOST']].tail(1)  # 取用户id、标记（每组的最后一个值）
label.set_index('USER_ID', inplace=True)  # 将“USER_ID”设为索引
label = data_group[['USER_ID', 'IS_LOST']].tail(1)   # 取用户id、标记（每组的最后一个值）
label.set_index('USER_ID', inplace=True)             # 将“USER_ID”设为索引

data_1 = data_group[['CUST_SEX', 'CERT_AGE', 'TERM_TYPE']].first() 
data_2 = data_group['INNET_MONTH'].last()
data_3 = pd.DataFrame(data_group['IS_AGREE'].agg(cal_is_agree))#agg是一个聚合函数，聚合函数操作始终是在轴（默认是列轴，也可设置行轴）上执行，
date = data_group['AGREE_EXP_DATE'].last()  # 取第3个月的"合约计划到期时长"
num_mon = (pd.to_datetime(date, format='%Y%m') - pd.to_datetime('2016-03')).dt.days/30  # 时长以“月”为单位
data_4 = pd.DataFrame(num_mon).fillna(-1)    #用-1填充缺失值
data_5 = pd.DataFrame(data_group['CREDIT_LEVEL'].agg('mean'))    # 信用等级
# 3.7 VIP等级
data_6 = data_group['VIP_LVL'].last().fillna(0)    # 取最后一个值
# 3.8 本月费用(取三个月的平均值)特征构建
data_7 = pd.DataFrame(data_group['ACCT_FEE'].mean())
# 3.9 平均每次通话时长
# 总通话
data_8_1 = pd.DataFrame(data_group['CALL_DURA'].sum()/data_group['CDR_NUM'].sum(),
                        columns=['Total_mean'])
# 本地通话
data_8_2 = pd.DataFrame(data_group['NO_ROAM_LOCAL_CALL_DURA'].sum()/data_group['NO_ROAM_LOCAL_CDR_NUM'].sum(),
                         columns=['Local_mean'])
# 国内长途通话
data_8_3 = pd.DataFrame(data_group['NO_ROAM_GN_LONG_CALL_DURA'].sum() / data_group['NO_ROAM_GN_LONG_CDR_NUM'].sum(),
                         columns=['GN_Long_mean'])
# 国内漫游通话
data_8_4 = pd.DataFrame(data_group['GN_ROAM_CALL_DURA'].sum() / data_group['GN_ROAM_CDR_NUM'].sum(),
                         columns=['GN_Roam_mean'])
# 数据拼接
data_8 = pd.concat([data_8_1, data_8_2, data_8_3, data_8_4], axis=1).fillna(0)
# 3.10 其他变量
# 非漫游通话次数（次）、短信发送数（条）、上网流量(MB)、本地非漫游上网流量(MB)、国内漫游上网流量(MB)、
# 有通话天数、有主叫天数、有被叫天数  （主叫 + 被叫 ≠ 总通话）
# 语音呼叫圈、主叫呼叫圈、被叫呼叫圈
data_9 = data_group[['NO_ROAM_CDR_NUM', 'P2P_SMS_CNT_UP', 'TOTAL_FLUX', 'LOCAL_FLUX','GN_ROAM_FLUX',
                      'CALL_DAYS', 'CALLING_DAYS', 'CALLED_DAYS',
                      'CALL_RING','CALLING_RING', 'CALLED_RING']].agg('mean')


# 对所有特征&标签按索引重新排序，以保证数据拼接时索引一致
label.sort_index(inplace=True)
data_1.sort_index(inplace=True)
data_2.sort_index(inplace=True)
data_3.sort_index(inplace=True)
data_4.sort_index(inplace=True)
data_5.sort_index(inplace=True)
data_6.sort_index(inplace=True)
data_7.sort_index(inplace=True)
data_8.sort_index(inplace=True)
data_9.sort_index(inplace=True)
# 拼接所有特征&标记
data_new = pd.concat([data_1, data_2, data_3, data_4,
           data_5, data_6, data_7, data_8, data_9, label], axis=1)
print(data_new.head() )

#缺失值处理
print("6 isnull \n",data_new.isnull().sum())    # 查看缺失值
data_new = data_new.fillna(method='ffill').fillna(method='bfill')      # 近邻值填充(向下填充+向上填充)

data_new.to_csv('clear_data.csv', index=True, encoding='utf-8-sig')
data = pd.read_csv('clear_data.csv', index_col=0)
corr = data.corr()    # 皮尔逊相关系数 矩阵
# 以0.08作为筛选阈值
feature_index = corr['IS_LOST'].drop('IS_LOST').abs() > 0.08    # 取出与"标记"的相关系数
feature_name = feature_index.loc[feature_index].index           # 选出的重要特征名

# 提取特征与标记
X = data.loc[:, feature_name]    # 样本自变量
y = data.loc[:, 'IS_LOST']       # 样本目标变量
# 样本不平衡 
y.value_counts()

index_positive = y.index[y == 1]          # 正样本的索引
index_negative = np.random.choice(a=y.index[y == 0].tolist(), size=y.value_counts()[1])   # 负样本的索引，对负样本进行下采样操作

X_positive = X.loc[index_positive, :]     # 正样本自变量
X_negative = X.loc[index_negative, :]     # 负样本自变量

y_positive = y.loc[index_positive]        # 正样本标签
y_negative = y.loc[index_negative]        # 负样本标签

X = pd.concat([X_positive, X_negative], axis=0)    # 处理后的正样本
y = pd.concat([y_positive, y_negative], axis=0)    # 处理后的负样本



##################################
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y) # 数据划分

rfc = RandomForestClassifier()    # 初始化随机森林模型
rfc.fit(X_train, y_train)         # 模型训练
y_pre = rfc.predict(X_test)       # 调用模型对测试样本进行预测
print(classification_report(y_test, y_pre))    # 打印分类报告（包含了各模型性能评价指标）
rfc_acc = round(accuracy_score(y_pre,y_test)*100,2)
print(f"logistic accuracy is: {rfc_acc}%")

######
# 创建决策树模型
dtc = DecisionTreeClassifier()
# 训练模型
dtc.fit(X_train,y_train)
# 预测训练集和测试集结果
dtc_pred = dtc.predict(X_test)
# 计算精确度
dtc_acc = round(accuracy_score(dtc_pred,y_test)*100,2)
print(f"decision tree accuracy is: {dtc_acc}%")

######
# 创建逻辑回归模型
lr = LogisticRegression()
# 训练模型
lr.fit(X_train,y_train)
# 预测训练集和测试集结果
lr_pred = lr.predict(X_test)
# 计算精确度
lr_acc = round(accuracy_score(lr_pred,y_test)*100,2)
print(f"logistic accuracy is: {lr_acc}%")


print("the best acc is :" ,(max(lr_acc,dtc_acc,rfc_acc)),"%")
