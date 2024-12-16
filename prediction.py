#此python文件用于通过cimp6数据来预测稻飞虱的首迁期和虫量，在之前的模型对比中，
#线性回归模型和多项式回归模型的拟合度是最高的，其中线性回归模型的R^2为1，MSE为2.44e^-29
#多项式回归模型的R^2为1，MSE为1.127e^-26，两者的模型拟合度极高。
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
df['日期'] = df['日期'].apply(lambda x: x.timestamp())

# 异常值检测与清洗
# 使用Z-score方法来检测异常值
from scipy import stats
z_scores = np.abs(stats.zscore(df.select_dtypes(include=[np.number])))  # 只对数值型数据计算Z-score
df_cleaned = df[(z_scores < 3).all(axis=1)]  # 删除Z-score大于3的异常值

#选择相关性高的特征，分别对首迁期和虫数做预测
selected_feature_forFirstComing = ['month', '1000hPa_air', '日期', '850hPa_air', 'year',  '850hPa_number', '850hPa_azimuth']
selected_feature_forInsectNum = ['1000hPa_air', '850hPa_air', '1000hPa_number', '1000hPa_azimuth', '1000hPa_wind']

X = df_cleaned[selected_feature]
y_FirstCome = df_cleaned['日期-首迁期']
y_InscectNumWhite = df_cleaned['本候灯下白背飞虱虫量（头）'].copy()
y_InscectNumBro = df_cleaned['本候灯下稻飞虱合计'].copy()
y_InscectNumWhite = np.log(y_InscectNumWhite+1)  # 直接对整个Series进行对数变换，会自动对每个元素应用对数函数
y_InscectNumBro = np.log(y_InscectNumBro+1)

ss = StandardScaler()
X_std = ss.fit_transform(X)

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.2, random_state=0)


#导入线性回归相关库
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 初始化线性回归模型
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

print(r2_score(y_test,y_pred_lr ))
print(mean_squared_error(y_test, y_pred_lr))


#导入预测数据
# df_pred = pd.read_excel('cipm_Data/combined_华南ssp126.xlsx')
# df_pred['日期'] = df_pred['日期'].apply(lambda x: x.timestamp())
# # 异常值检测与清洗
# # 使用Z-score方法来检测异常值
# from scipy import stats
# z_scores = np.abs(stats.zscore(df_pred.select_dtypes(include=[np.number])))  # 只对数值型数据计算Z-score
# df_pred_cleaned = df_pred[(z_scores < 3).all(axis=1)]  # 删除Z-score大于3的异常值
#
# selected_feature = ['month', '1000hPa_air', '日期', '850hPa_air', 'year',  '850hPa_number', '850hPa_azimuth']
# X_pred = df_pred_cleaned[selected_feature]
# # 此处修改为使用已训练好的ss进行标准化变换，而不是重新拟合标准化
# X_pred_std = ss.transform(X_pred)
#
# y_pred_lr = lr_model.predict(X_pred_std)
# # 选择需要的列
# # 假设需要的列来自另一个 DataFrame，例如 X_pred_std
# print(y_pred_lr.shape())
# output = pd.DataFrame({
#     'prediction': y_pred_lr,
#     '1000hPa_air': X_pred['1000hPa_air'],
#     '850hPa_air': X_pred['850hPa_air'],
#     '850hPa_number': X_pred['850hPa_number'],
#     '850hPa_azimuth': X_pred['850hPa_azimuth'],
#     'month': X_pred['month'],
#     'year': X_pred['year']
# })
#
# output.to_excel("output.xlsx", index=False)