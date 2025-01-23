import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 导入数据
df = pd.read_excel('气象地点+虫情地点数据/Step7_固始-徽州/Step7_固始-徽州.xlsx')
df_5 = pd.read_excel('气象地点+虫情地点数据/Step7_固始-徽州/Step7_固始-徽州(气象数据前移5天).xlsx')
df_10 = pd.read_excel('气象地点+虫情地点数据/Step7_固始-徽州/Step7_固始-徽州(气象数据前移10天).xlsx')
df_15 = pd.read_excel('气象地点+虫情地点数据/Step7_固始-徽州/Step7_固始-徽州(气象数据前移15天).xlsx')
df_pred = pd.read_csv('cmip_Nor/SSP1/江淮固始ssp126.csv')


missing_dates = df[df['date'].isna()]
print(missing_dates)
df['date'] = df['date'].apply(lambda x: x.timestamp())
# 查找 NaT 值

missing_dates = df_5[df_5['date'].isna()]
print(missing_dates)
df_5['date'] = df_5['date'].apply(lambda x: x.timestamp())

df_10['date'] = df_10['date'].apply(lambda x: x.timestamp())
missing_dates = df_10[df_10['date'].isna()]
print(missing_dates)
df_15['date'] = df_15['date'].apply(lambda x: x.timestamp())
missing_dates = df_15[df_15['date'].isna()]
print(missing_dates)

df['白背飞虱'] = np.log(df['白背飞虱'] + 1)
df_5['白背飞虱'] = np.log(df_5['白背飞虱'] + 1)
df_10['白背飞虱'] = np.log(df_10['白背飞虱'] + 1)
df_15['白背飞虱'] = np.log(df_15['白背飞虱'] + 1)

df['褐飞虱'] = np.log(df['褐飞虱'] + 1)
df_5['褐飞虱'] = np.log(df_5['褐飞虱'] + 1)
df_10['褐飞虱'] = np.log(df_10['褐飞虱'] + 1)
df_15['褐飞虱'] = np.log(df_15['褐飞虱'] + 1)


print(df.columns)


# 相关性矩阵
correlation_matrix = df.corr()
# 绘制热力图，查看各特征与目标变量的相关性
selected_rows_cols = correlation_matrix.loc[['date', 'year', 'month', 'hou', '1000hPa_air', '850hPa_air', '1000hPa_rhum', '850hPa_rhum', '1000hPa_wind', '850hPa_wind', '1000hPa_azimuth', '850hPa_azimuth', '1000hPa_number', '850hPa_number', '850hPa_omega'], ['白背飞虱', '褐飞虱']]
plt.figure(figsize=(20, 12))
sns.heatmap(selected_rows_cols, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Correlation Matrix")
plt.show()

correlation_matrix1 = df_5.corr()
# 绘制热力图，查看各特征与目标变量的相关性
selected_rows_cols = correlation_matrix1.loc[['date', 'year', 'month', 'hou', '1000hPa_air', '850hPa_air', '1000hPa_rhum', '850hPa_rhum', '1000hPa_wind', '850hPa_wind', '1000hPa_azimuth', '850hPa_azimuth', '1000hPa_number', '850hPa_number', '850hPa_omega'], ['白背飞虱', '褐飞虱']]
plt.figure(figsize=(20, 12))
sns.heatmap(selected_rows_cols, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Correlation Matrix")
plt.show()


correlation_matrix2 = df_10.corr()
# 绘制热力图，查看各特征与目标变量的相关性
selected_rows_cols = correlation_matrix2.loc[['date', 'year', 'month', 'hou', '1000hPa_air', '850hPa_air', '1000hPa_rhum', '850hPa_rhum', '1000hPa_wind', '850hPa_wind', '1000hPa_azimuth', '850hPa_azimuth', '1000hPa_number', '850hPa_number', '850hPa_omega'], ['白背飞虱', '褐飞虱']]
plt.figure(figsize=(20, 12))
sns.heatmap(selected_rows_cols, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Correlation Matrix")
plt.show()

correlation_matrix3 = df_15.corr()
# 绘制热力图，查看各特征与目标变量的相关性
selected_rows_cols = correlation_matrix3.loc[['date', 'year', 'month', 'hou', '1000hPa_air', '850hPa_air', '1000hPa_rhum', '850hPa_rhum', '1000hPa_wind', '850hPa_wind', '1000hPa_azimuth', '850hPa_azimuth', '1000hPa_number', '850hPa_number', '850hPa_omega'], ['白背飞虱', '褐飞虱']]
plt.figure(figsize=(20, 12))
sns.heatmap(selected_rows_cols, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Correlation Matrix")
plt.show()



# --------------------------------------------------------------------------------------------------------------
# 选择和目标变量"白背飞虱"和"褐飞虱"相关性较强的特征
target = "白背飞虱"
cor_target = correlation_matrix[target].sort_values(ascending=False) #这里需要修改！！！！！！！！！！1
print(cor_target)

# 选择和目标变量相关性较高的特征
# 可以选择相关性高于某个阈值的特征，假设阈值为0.1
# 直接在筛选相关性绝对值大于0.1的特征索引时，排除指定的特征
selected_features = cor_target[(cor_target > 0.1) & (~cor_target.index.isin(['白背飞虱', '褐飞虱']))].index.tolist()
print("Selected features:", selected_features)

# 目标和特征值
# X = df_5[selected_features].drop(['白背飞虱', '褐飞虱'], axis=1)
X = df[selected_features] #这里需要修改！！！！！！！！11
y = df[target] #这里需要修改！！！！！！！！！！！！

# 特征标准化
# ss = StandardScaler()
# X_std = ss.fit_transform(X)

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

mes_white = []

# 随机森林回归
rf_white = RandomForestRegressor(n_estimators=150, max_depth=10, random_state=0)
rf_white.fit(X_train, y_train)

y_pred_rf = rf_white.predict(X_test)
mes_white.append(mean_squared_error(y_test, y_pred_rf))
print("Random Forest R2:", r2_score(y_test, y_pred_rf))
print("Random Forest MSE:", mean_squared_error(y_test, y_pred_rf))
print('\n')


from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 初始化KNN模型
knn_model_white = KNeighborsRegressor(n_neighbors=5)
knn_model_white.fit(X_train, y_train)

# 预测并评估
y_pred_knn = knn_model_white.predict(X_test)
mes_white.append(mean_squared_error(y_test, y_pred_knn))
print("KNN R2:", r2_score(y_test, y_pred_knn))
print("KNN MSE:", mean_squared_error(y_test, y_pred_knn))
print('\n')

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 初始化线性回归模型
lr_model_white = LinearRegression()
lr_model_white.fit(X_train, y_train)

# 预测并评估
y_pred_lr = lr_model_white.predict(X_test)
mes_white.append(mean_squared_error(y_test, y_pred_lr))
print("Linear Regression R2:", r2_score(y_test, y_pred_lr))
print("Linear Regression MSE:", mean_squared_error(y_test, y_pred_lr))
print('\n')


# --------------------------------------------------------------------------------------------------------------



target1 = "褐飞虱"
cor_target1 = correlation_matrix[target1].sort_values(ascending=False) #这里需要修改！！！！！！！！！！！！！！！！！！
print(cor_target1)

# 选择和目标变量相关性较高的特征
# 可以选择相关性高于某个阈值的特征，假设阈值为0.1
# 直接在筛选相关性绝对值大于0.1的特征索引时，排除指定的特征
selected_features1 = cor_target1[(cor_target1 > 0.1) & (~cor_target1.index.isin(['白背飞虱', '褐飞虱']))].index.tolist()
print("Selected features:", selected_features1)


# 目标和特征值
# X = df_5[selected_features1].drop(['白背飞虱', '褐飞虱'], axis=1)
X = df[selected_features1] #这里需要修改！！！！！！！！！！！！！1
y = df[target1] #这里需要修改！！！！！！！！！！！！！！！！！

# 特征标准化
# ss = StandardScaler()
# X_std = ss.fit_transform(X)

mes_brown = []

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# 随机森林回归
rf = RandomForestRegressor(n_estimators=150, max_depth=10, random_state=0)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
mes_brown.append(mean_squared_error(y_test, y_pred_rf))
print("Random Forest R2:", r2_score(y_test, y_pred_rf))
print("Random Forest MSE:", mean_squared_error(y_test, y_pred_rf))
print('\n')


from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 初始化KNN模型
knn_model = KNeighborsRegressor(n_neighbors=5)
knn_model.fit(X_train, y_train)

# 预测并评估
y_pred_knn = knn_model.predict(X_test)
mes_brown.append(mean_squared_error(y_test, y_pred_knn))
print("KNN R2:", r2_score(y_test, y_pred_knn))
print("KNN MSE:", mean_squared_error(y_test, y_pred_knn))
print('\n')

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 初始化线性回归模型
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# 预测并评估
y_pred_lr = lr_model.predict(X_test)
mes_brown.append(mean_squared_error(y_test, y_pred_lr))
print("Linear Regression R2:", r2_score(y_test, y_pred_lr))
print("Linear Regression MSE:", mean_squared_error(y_test, y_pred_lr))
print('\n')

print(mes_white)
print(mes_brown)


# ----------------------------------------------------------------------------------

# 进行数据预测


# 预测白背飞虱
X_white = df_pred[selected_features]
print(X_white.columns)

if mes_white[0] < mes_white[1] and mes_white[0] < mes_white[2]:
    white_num = rf_white.predict(X_white)
elif mes_white[1] < mes_white[2] and mes_white[1] < mes_white[0]:
    white_num = knn_model_white.predict(X_white)
elif mes_white[2] < mes_white[0] and mes_white[2] < mes_white[1]:
    white_num = lr_model_white.predict(X_white)

df_pred['白背飞虱'] = white_num

# 预测褐飞虱
X_brown = df_pred[selected_features1]
print(X_brown.columns)

if mes_white[0] < mes_white[1] and mes_white[0] < mes_white[2]:
    brown_num = rf.predict(X_brown)
elif mes_white[1] < mes_white[0] and mes_white[1] < mes_white[2]:
    brown_num = knn_model.predict(X_brown)
elif mes_white[2] < mes_white[0] and mes_white[2] < mes_white[1]:
    brown_num = lr_model.predict(X_brown)

df_pred['褐飞虱'] = brown_num

df_pred.to_excel('pred_SSP126_Step7_固始-徽州.xlsx', index=False)



# --------------------------------------------------------------------------------------------------------------
# 进行气象数据前移5天地方模型训练和预测
# 选择和目标变量"白背飞虱"和"褐飞虱"相关性较强的特征
target = "白背飞虱"
cor_target = correlation_matrix1[target].sort_values(ascending=False) #这里需要修改！！！！！！！！！！1
print(cor_target)

# 选择和目标变量相关性较高的特征
# 可以选择相关性高于某个阈值的特征，假设阈值为0.1
# 直接在筛选相关性绝对值大于0.1的特征索引时，排除指定的特征
selected_features = cor_target1[(cor_target > 0.1) & (~cor_target.index.isin(['白背飞虱', '褐飞虱']))].index.tolist()
print("Selected features:", selected_features)

# 目标和特征值
# X = df_5[selected_features].drop(['白背飞虱', '褐飞虱'], axis=1)
X = df_5[selected_features] #这里需要修改！！！！！！！！11
y = df_5[target] #这里需要修改！！！！！！！！！！！！

# 特征标准化
# ss = StandardScaler()
# X_std = ss.fit_transform(X)

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

mes_white = []

# 随机森林回归
rf_white = RandomForestRegressor(n_estimators=150, max_depth=10, random_state=0)
rf_white.fit(X_train, y_train)

y_pred_rf = rf_white.predict(X_test)
mes_white.append(mean_squared_error(y_test, y_pred_rf))
print("Random Forest R2:", r2_score(y_test, y_pred_rf))
print("Random Forest MSE:", mean_squared_error(y_test, y_pred_rf))
print('\n')


from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 初始化KNN模型
knn_model_white = KNeighborsRegressor(n_neighbors=5)
knn_model_white.fit(X_train, y_train)

# 预测并评估
y_pred_knn = knn_model_white.predict(X_test)
mes_white.append(mean_squared_error(y_test, y_pred_knn))
print("KNN R2:", r2_score(y_test, y_pred_knn))
print("KNN MSE:", mean_squared_error(y_test, y_pred_knn))
print('\n')

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 初始化线性回归模型
lr_model_white = LinearRegression()
lr_model_white.fit(X_train, y_train)

# 预测并评估
y_pred_lr = lr_model_white.predict(X_test)
mes_white.append(mean_squared_error(y_test, y_pred_lr))
print("Linear Regression R2:", r2_score(y_test, y_pred_lr))
print("Linear Regression MSE:", mean_squared_error(y_test, y_pred_lr))
print('\n')


# --------------------------------------------------------------------------------------------------------------



target1 = "褐飞虱"
cor_target1 = correlation_matrix1[target1].sort_values(ascending=False) #这里需要修改！！！！！！！！！！！！！！！！！！
print(cor_target1)

# 选择和目标变量相关性较高的特征
# 可以选择相关性高于某个阈值的特征，假设阈值为0.1
# 直接在筛选相关性绝对值大于0.1的特征索引时，排除指定的特征
selected_features1 = cor_target1[(cor_target1 > 0.1) & (~cor_target1.index.isin(['白背飞虱', '褐飞虱']))].index.tolist()
print("Selected features:", selected_features1)


# 目标和特征值
# X = df_5[selected_features1].drop(['白背飞虱', '褐飞虱'], axis=1)
X = df_5[selected_features1] #这里需要修改！！！！！！！！！！！！！1
y = df_5[target1] #这里需要修改！！！！！！！！！！！！！！！！！

# 特征标准化
# ss = StandardScaler()
# X_std = ss.fit_transform(X)

mes_brown = []

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# 随机森林回归
rf = RandomForestRegressor(n_estimators=150, max_depth=10, random_state=0)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
mes_brown.append(mean_squared_error(y_test, y_pred_rf))
print("Random Forest R2:", r2_score(y_test, y_pred_rf))
print("Random Forest MSE:", mean_squared_error(y_test, y_pred_rf))
print('\n')


from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 初始化KNN模型
knn_model = KNeighborsRegressor(n_neighbors=5)
knn_model.fit(X_train, y_train)

# 预测并评估
y_pred_knn = knn_model.predict(X_test)
mes_brown.append(mean_squared_error(y_test, y_pred_knn))
print("KNN R2:", r2_score(y_test, y_pred_knn))
print("KNN MSE:", mean_squared_error(y_test, y_pred_knn))
print('\n')

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 初始化线性回归模型
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# 预测并评估
y_pred_lr = lr_model.predict(X_test)
mes_brown.append(mean_squared_error(y_test, y_pred_lr))
print("Linear Regression R2:", r2_score(y_test, y_pred_lr))
print("Linear Regression MSE:", mean_squared_error(y_test, y_pred_lr))
print('\n')

print(mes_white)
print(mes_brown)


# ----------------------------------------------------------------------------------

# 进行数据预测



# 预测白背飞虱
X_white = df_pred[selected_features]
print(X_white.columns)

if mes_white[0] < mes_white[1] and mes_white[0] < mes_white[2]:
    white_num = rf_white.predict(X_white)
elif mes_white[1] < mes_white[2] and mes_white[1] < mes_white[0]:
    white_num = knn_model_white.predict(X_white)
elif mes_white[2] < mes_white[0] and mes_white[2] < mes_white[1]:
    white_num = lr_model_white.predict(X_white)

df_pred['白背飞虱'] = white_num

# 预测褐飞虱
X_brown = df_pred[selected_features1]
print(X_brown.columns)

if mes_white[0] < mes_white[1] and mes_white[0] < mes_white[2]:
    brown_num = rf.predict(X_brown)
elif mes_white[1] < mes_white[0] and mes_white[1] < mes_white[2]:
    brown_num = knn_model.predict(X_brown)
elif mes_white[2] < mes_white[0] and mes_white[2] < mes_white[1]:
    brown_num = lr_model.predict(X_brown)

df_pred['褐飞虱'] = brown_num

df_pred.to_excel('pred_SSP126_Step7_固始-徽州(气象数据前移5天).xlsx', index=False)


# --------------------------------------------------------------------------------------------------------------
# 进行气象数据前移10天地方模型训练和预测
# 选择和目标变量"白背飞虱"和"褐飞虱"相关性较强的特征
target = "白背飞虱"
cor_target = correlation_matrix2[target].sort_values(ascending=False) #这里需要修改！！！！！！！！！！1
print(cor_target)

# 选择和目标变量相关性较高的特征
# 可以选择相关性高于某个阈值的特征，假设阈值为0.1
# 直接在筛选相关性绝对值大于0.1的特征索引时，排除指定的特征
selected_features = cor_target[(cor_target > 0.1) & (~cor_target.index.isin(['白背飞虱', '褐飞虱']))].index.tolist()
print("Selected features:", selected_features)

# 目标和特征值
# X = df_5[selected_features].drop(['白背飞虱', '褐飞虱'], axis=1)
X = df_10[selected_features] #这里需要修改！！！！！！！！11
y = df_10[target] #这里需要修改！！！！！！！！！！！！

# 特征标准化
# ss = StandardScaler()
# X_std = ss.fit_transform(X)

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

mes_white = []

# 随机森林回归
rf_white = RandomForestRegressor(n_estimators=150, max_depth=10, random_state=0)
rf_white.fit(X_train, y_train)

y_pred_rf = rf_white.predict(X_test)
mes_white.append(mean_squared_error(y_test, y_pred_rf))
print("Random Forest R2:", r2_score(y_test, y_pred_rf))
print("Random Forest MSE:", mean_squared_error(y_test, y_pred_rf))
print('\n')


from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 初始化KNN模型
knn_model_white = KNeighborsRegressor(n_neighbors=5)
knn_model_white.fit(X_train, y_train)

# 预测并评估
y_pred_knn = knn_model_white.predict(X_test)
mes_white.append(mean_squared_error(y_test, y_pred_knn))
print("KNN R2:", r2_score(y_test, y_pred_knn))
print("KNN MSE:", mean_squared_error(y_test, y_pred_knn))
print('\n')

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 初始化线性回归模型
lr_model_white = LinearRegression()
lr_model_white.fit(X_train, y_train)

# 预测并评估
y_pred_lr = lr_model_white.predict(X_test)
mes_white.append(mean_squared_error(y_test, y_pred_lr))
print("Linear Regression R2:", r2_score(y_test, y_pred_lr))
print("Linear Regression MSE:", mean_squared_error(y_test, y_pred_lr))
print('\n')


# --------------------------------------------------------------------------------------------------------------



target1 = "褐飞虱"
cor_target1 = correlation_matrix2[target1].sort_values(ascending=False) #这里需要修改！！！！！！！！！！！！！！！！！！
print(cor_target1)

# 选择和目标变量相关性较高的特征
# 可以选择相关性高于某个阈值的特征，假设阈值为0.1
# 直接在筛选相关性绝对值大于0.1的特征索引时，排除指定的特征
selected_features1 = cor_target1[(cor_target1 > 0.1) & (~cor_target1.index.isin(['白背飞虱', '褐飞虱']))].index.tolist()
print("Selected features:", selected_features1)


# 目标和特征值
# X = df_5[selected_features1].drop(['白背飞虱', '褐飞虱'], axis=1)
X = df_10[selected_features1] #这里需要修改！！！！！！！！！！！！！1
y = df_10[target1] #这里需要修改！！！！！！！！！！！！！！！！！

# 特征标准化
# ss = StandardScaler()
# X_std = ss.fit_transform(X)

mes_brown = []

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# 随机森林回归
rf = RandomForestRegressor(n_estimators=150, max_depth=10, random_state=0)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
mes_brown.append(mean_squared_error(y_test, y_pred_rf))
print("Random Forest R2:", r2_score(y_test, y_pred_rf))
print("Random Forest MSE:", mean_squared_error(y_test, y_pred_rf))
print('\n')


from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 初始化KNN模型
knn_model = KNeighborsRegressor(n_neighbors=5)
knn_model.fit(X_train, y_train)

# 预测并评估
y_pred_knn = knn_model.predict(X_test)
mes_brown.append(mean_squared_error(y_test, y_pred_knn))
print("KNN R2:", r2_score(y_test, y_pred_knn))
print("KNN MSE:", mean_squared_error(y_test, y_pred_knn))
print('\n')

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 初始化线性回归模型
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# 预测并评估
y_pred_lr = lr_model.predict(X_test)
mes_brown.append(mean_squared_error(y_test, y_pred_lr))
print("Linear Regression R2:", r2_score(y_test, y_pred_lr))
print("Linear Regression MSE:", mean_squared_error(y_test, y_pred_lr))
print('\n')

print(mes_white)
print(mes_brown)


# ----------------------------------------------------------------------------------

# 进行数据预测



# 预测白背飞虱
X_white = df_pred[selected_features]
print(X_white.columns)

if mes_white[0] < mes_white[1] and mes_white[0] < mes_white[2]:
    white_num = rf_white.predict(X_white)
elif mes_white[1] < mes_white[2] and mes_white[1] < mes_white[0]:
    white_num = knn_model_white.predict(X_white)
elif mes_white[2] < mes_white[0] and mes_white[2] < mes_white[1]:
    white_num = lr_model_white.predict(X_white)

df_pred['白背飞虱'] = white_num

# 预测褐飞虱
X_brown = df_pred[selected_features1]
print(X_brown.columns)

if mes_white[0] < mes_white[1] and mes_white[0] < mes_white[2]:
    brown_num = rf.predict(X_brown)
elif mes_white[1] < mes_white[0] and mes_white[1] < mes_white[2]:
    brown_num = knn_model.predict(X_brown)
elif mes_white[2] < mes_white[0] and mes_white[2] < mes_white[1]:
    brown_num = lr_model.predict(X_brown)

df_pred['褐飞虱'] = brown_num

df_pred.to_excel('pred_SSP126_Step7_固始-徽州(气象数据前移10天).xlsx', index=False)


# --------------------------------------------------------------------------------------------------------------
# 进行气象数据前移15天地方模型训练和预测
# 选择和目标变量"白背飞虱"和"褐飞虱"相关性较强的特征
target = "白背飞虱"
cor_target = correlation_matrix3[target].sort_values(ascending=False) #这里需要修改！！！！！！！！！！1
print(cor_target)

# 选择和目标变量相关性较高的特征
# 可以选择相关性高于某个阈值的特征，假设阈值为0.1
# 直接在筛选相关性绝对值大于0.1的特征索引时，排除指定的特征
selected_features = cor_target1[(cor_target > 0.1) & (~cor_target.index.isin(['白背飞虱', '褐飞虱']))].index.tolist()
print("Selected features:", selected_features)

# 目标和特征值
# X = df_5[selected_features].drop(['白背飞虱', '褐飞虱'], axis=1)
X = df_15[selected_features] #这里需要修改！！！！！！！！11
y = df_15[target] #这里需要修改！！！！！！！！！！！！

# 特征标准化
# ss = StandardScaler()
# X_std = ss.fit_transform(X)

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

mes_white = []

# 随机森林回归
rf_white = RandomForestRegressor(n_estimators=150, max_depth=10, random_state=0)
rf_white.fit(X_train, y_train)

y_pred_rf = rf_white.predict(X_test)
mes_white.append(mean_squared_error(y_test, y_pred_rf))
print("Random Forest R2:", r2_score(y_test, y_pred_rf))
print("Random Forest MSE:", mean_squared_error(y_test, y_pred_rf))
print('\n')


from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 初始化KNN模型
knn_model_white = KNeighborsRegressor(n_neighbors=5)
knn_model_white.fit(X_train, y_train)

# 预测并评估
y_pred_knn = knn_model_white.predict(X_test)
mes_white.append(mean_squared_error(y_test, y_pred_knn))
print("KNN R2:", r2_score(y_test, y_pred_knn))
print("KNN MSE:", mean_squared_error(y_test, y_pred_knn))
print('\n')

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 初始化线性回归模型
lr_model_white = LinearRegression()
lr_model_white.fit(X_train, y_train)

# 预测并评估
y_pred_lr = lr_model_white.predict(X_test)
mes_white.append(mean_squared_error(y_test, y_pred_lr))
print("Linear Regression R2:", r2_score(y_test, y_pred_lr))
print("Linear Regression MSE:", mean_squared_error(y_test, y_pred_lr))
print('\n')


# --------------------------------------------------------------------------------------------------------------



target1 = "褐飞虱"
cor_target1 = correlation_matrix3[target1].sort_values(ascending=False) #这里需要修改！！！！！！！！！！！！！！！！！！
print(cor_target1)

# 选择和目标变量相关性较高的特征
# 可以选择相关性高于某个阈值的特征，假设阈值为0.1
# 直接在筛选相关性绝对值大于0.1的特征索引时，排除指定的特征
selected_features1 = cor_target1[(cor_target1 > 0.1) & (~cor_target1.index.isin(['白背飞虱', '褐飞虱']))].index.tolist()
print("Selected features:", selected_features1)


# 目标和特征值
# X = df_5[selected_features1].drop(['白背飞虱', '褐飞虱'], axis=1)
X = df_15[selected_features1] #这里需要修改！！！！！！！！！！！！！1
y = df_15[target1] #这里需要修改！！！！！！！！！！！！！！！！！

# 特征标准化
# ss = StandardScaler()
# X_std = ss.fit_transform(X)

mes_brown = []

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# 随机森林回归
rf = RandomForestRegressor(n_estimators=150, max_depth=10, random_state=0)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
mes_brown.append(mean_squared_error(y_test, y_pred_rf))
print("Random Forest R2:", r2_score(y_test, y_pred_rf))
print("Random Forest MSE:", mean_squared_error(y_test, y_pred_rf))
print('\n')


from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 初始化KNN模型
knn_model = KNeighborsRegressor(n_neighbors=5)
knn_model.fit(X_train, y_train)

# 预测并评估
y_pred_knn = knn_model.predict(X_test)
mes_brown.append(mean_squared_error(y_test, y_pred_knn))
print("KNN R2:", r2_score(y_test, y_pred_knn))
print("KNN MSE:", mean_squared_error(y_test, y_pred_knn))
print('\n')

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 初始化线性回归模型
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# 预测并评估
y_pred_lr = lr_model.predict(X_test)
mes_brown.append(mean_squared_error(y_test, y_pred_lr))
print("Linear Regression R2:", r2_score(y_test, y_pred_lr))
print("Linear Regression MSE:", mean_squared_error(y_test, y_pred_lr))
print('\n')

print(mes_white)
print(mes_brown)


# ----------------------------------------------------------------------------------

# 进行数据预测



# 预测白背飞虱
X_white = df_pred[selected_features]
print(X_white.columns)

if mes_white[0] < mes_white[1] and mes_white[0] < mes_white[2]:
    white_num = rf_white.predict(X_white)
elif mes_white[1] < mes_white[2] and mes_white[1] < mes_white[0]:
    white_num = knn_model_white.predict(X_white)
elif mes_white[2] < mes_white[0] and mes_white[2] < mes_white[1]:
    white_num = lr_model_white.predict(X_white)

df_pred['白背飞虱'] = white_num

# 预测褐飞虱
X_brown = df_pred[selected_features1]
print(X_brown.columns)

if mes_white[0] < mes_white[1] and mes_white[0] < mes_white[2]:
    brown_num = rf.predict(X_brown)
elif mes_white[1] < mes_white[0] and mes_white[1] < mes_white[2]:
    brown_num = knn_model.predict(X_brown)
elif mes_white[2] < mes_white[0] and mes_white[2] < mes_white[1]:
    brown_num = lr_model.predict(X_brown)

df_pred['褐飞虱'] = brown_num

df_pred.to_excel('pred_SSP126_Step7_固始-徽州(气象数据前移15天).xlsx', index=False)
