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
df = pd.read_excel('dealed_data(xlsx)/combined_华南.xlsx')

# 时间转换
df['首迁期'] = pd.to_datetime(df['首迁期'])
df['首迁期'] = df['首迁期'].apply(lambda x: x.timestamp())
df['日期'] = df['日期'].apply(lambda x: x.timestamp())

# 异常值检测与清洗
# 使用Z-score方法来检测异常值
from scipy import stats
z_scores = np.abs(stats.zscore(df.select_dtypes(include=[np.number])))  # 只对数值型数据计算Z-score
df_cleaned = df[(z_scores < 3).all(axis=1)]  # 删除Z-score大于3的异常值

# 相关性矩阵
correlation_matrix = df_cleaned.corr()

# # 绘制热力图，查看各特征与目标变量的相关性
# plt.figure(figsize=(20, 12))
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
# plt.title("Correlation Matrix")
# plt.show()

    # 选择和目标变量"日期-首迁期"相关性较强的特征
target = "日期-首迁期"
target = '本候灯下白背飞虱虫量（头）'
cor_target = correlation_matrix[target].sort_values(ascending=False)
print(cor_target)

# 选择和目标变量相关性较高的特征
# 可以选择相关性高于某个阈值的特征，假设阈值为0.1
selected_features = cor_target[cor_target.abs() > 0.1].index.tolist()
print("Selected features:", selected_features)
# 特征和目标变量
#X = df_cleaned[selected_features].drop(['日期-首迁期', '首迁期', '925hPa_air', '925hPa_wind', '925hPa_number','925hPa_azimuth', '1000hPa_omega', 'air2m'], axis=1)
X = df_cleaned[selected_features].drop(['本候灯下白背飞虱虫量（头）', '本候灯下稻飞虱合计', '本侯灯下褐飞虱虫量（头）', '925hPa_air'] ,axis=1)
print("The choien feature：", X)
y = df_cleaned[target].copy()  # 使用copy方法避免后续修改影响原始的df_cleaned数据
y = np.log(y+1)  # 直接对整个Series进行对数变换，会自动对每个元素应用对数函数

# 特征标准化
ss = StandardScaler()
X_std = ss.fit_transform(X)

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.2, random_state=0)

# 随机森林回归
rf = RandomForestRegressor(n_estimators=150, max_depth=10, random_state=0)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
print("Random Forest R2:", r2_score(y_test, y_pred_rf))
print("Random Forest MSE:", mean_squared_error(y_test, y_pred_rf))
print('\n')
# 支持向量机回归
svm_model = SVR(kernel='poly', degree=3, gamma='auto')
svm_model.fit(X_train, y_train)

y_pred_svm = svm_model.predict(X_test)
print("SVM R2:", r2_score(y_test, y_pred_svm))
print("SVM MSE:", mean_squared_error(y_test, y_pred_svm))
print('\n')

from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 初始化XGBoost模型
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
xgb_model.fit(X_train, y_train)

# 预测并评估
y_pred_xgb = xgb_model.predict(X_test)
print("XGBoost R2:", r2_score(y_test, y_pred_xgb))
print("XGBoost MSE:", mean_squared_error(y_test, y_pred_xgb))
print('\n')

from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 初始化KNN模型
knn_model = KNeighborsRegressor(n_neighbors=5)
knn_model.fit(X_train, y_train)

# 预测并评估
y_pred_knn = knn_model.predict(X_test)
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
print("Linear Regression R2:", r2_score(y_test, y_pred_lr))
print("Linear Regression MSE:", mean_squared_error(y_test, y_pred_lr))
print('\n')

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 扩展特征为多项式duo
poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X_train)

# 使用线性回归拟合多项式回归
poly_model = LinearRegression()
poly_model.fit(X_poly, y_train)

# 对测试集进行预测
X_test_poly = poly.transform(X_test)
y_pred_poly = poly_model.predict(X_test_poly)

print("Polynomial Regression R2:", r2_score(y_test, y_pred_poly))
print("Polynomial Regression MSE:", mean_squared_error(y_test, y_pred_poly))
print('\n')

from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 初始化多层感知机回归模型
mlp_model = MLPRegressor(hidden_layer_sizes=(100, 100), max_iter=1000, random_state=0)
mlp_model.fit(X_train, y_train)

# 预测并评估
y_pred_mlp = mlp_model.predict(X_test)
print("MLP R2:", r2_score(y_test, y_pred_mlp))
print("MLP MSE:", mean_squared_error(y_test, y_pred_mlp))
