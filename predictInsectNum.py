import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

#导入训练数据
df = pd.read_excel('dealed_data(xlsx)/combined_华南.xlsx')

# 时间转换
df['首迁期'] = pd.to_datetime(df['首迁期'])
df['首迁期'] = df['首迁期'].apply(lambda x: x.timestamp())
df['日期'] = df['日期'].apply(lambda x: x.timestamp())

#异常值的检测和清洗
#使用Z-score的方式检测异常值
from scipy import stats
z_scores = np.abs(stats.zscore(df.select_dtypes(include=[np.number])))  # 只对数值型数据计算Z-score
df_cleaned = df[(z_scores < 3).all(axis=1)]  # 删除Z-score大于3的异常值

# 相关性矩阵
correlation_matrix = df_cleaned.corr()


target_white = "本候灯下白背飞虱虫量（头）"
target_brown = "本侯灯下褐飞虱虫量（头）"
cor_target_white = correlation_matrix[target_white].sort_values(ascending=False)
cor_target_brown = correlation_matrix[target_brown].sort_values(ascending=False)


# 选择和目标变量相关性较高的特征
# 可以选择相关性高于某个阈值的特征，假设阈值为0.1
selected_features_white = cor_target_white[cor_target_white.abs() > 0.1].index.tolist()
print("Selected features for the white insect:", selected_features_white)

selected_features_brown = cor_target_brown[cor_target_brown.abs() > 0.1].index.tolist()
print("Selected features for the brown insect:", selected_features_brown)

#设置训练集的输入的特征
X_white = df_cleaned[selected_features_white].drop(['本候灯下白背飞虱虫量（头）', '本候灯下稻飞虱合计', '本侯灯下褐飞虱虫量（头）', '925hPa_air','air2m'] ,axis=1)
print("Final selected features for the white insect:", X_white.columns)
X_Brown = df_cleaned[selected_features_brown].drop(['本候灯下白背飞虱虫量（头）', '本候灯下稻飞虱合计', '本侯灯下褐飞虱虫量（头）', '925hPa_air','air2m'] ,axis=1)
print("Final selected features for the brown insect:", X_Brown.columns)

#设置训练集的目标特征
y_white = df_cleaned[target_white].copy()  # 使用copy方法避免后续修改影响原始的df_cleaned数据
y_white = np.log(y_white+1)  # 直接对整个Series进行对数变换，会自动对每个元素应用对数函数
y_brown = df_cleaned[target_brown].copy()  # 使用copy方法避免后续修改影响原始的df_cleaned数据
y_brown = np.log(y_brown+1)  # 直接对整个Series进行对数变换，会自动对每个元素应用对数函数

# 特征标准化
ss_white = StandardScaler()
X_std_white = ss_white.fit_transform(X_white)
ss_brown = StandardScaler()
X_std_brown = ss_brown.fit_transform(X_Brown)

# 分割数据集
X_train_white, X_test_white, y_train_white, y_test_white = train_test_split(X_std_white, y_white, test_size=0.2, random_state=0)
X_train_brown, X_test_brown, y_train_brown, y_test_brown = train_test_split(X_std_brown, y_brown, test_size=0.2, random_state=0)

#使用线性回归对虫数进行预测

# 初始化线性回归模型
rf_white = RandomForestRegressor(n_estimators=150, max_depth=10, random_state=0)
rf_white.fit(X_train_white, y_train_white)

# 预测并评估
y_pred_lr = rf_white.predict(X_test_white)
print("Random Forest R2:", r2_score(y_test_white, y_pred_lr))
print("Random Forest MSE:", mean_squared_error(y_test_white, y_pred_lr))
print('\n')

rf_brown = RandomForestRegressor(n_estimators=150, max_depth=10, random_state=0)
rf_brown.fit(X_train_brown, y_train_brown)


y_pred_rl_brown = rf_brown.predict(X_test_brown)
print("Random Forest R2:", r2_score(y_test_brown, y_pred_rl_brown))
print("Random Forest MSE:", mean_squared_error(y_test_brown, y_pred_rl_brown))
print('\n')





#导入正式的预测数据
file_name = 'combined_华南ssp126.xlsx'
df_pred = pd.read_excel(f'cipm_Data\{file_name}')

#格式特殊的日期转换成时间戳格式
df_pred['日期'] = df_pred['日期'].apply(lambda x: x.timestamp())

# 异常值检测与清洗
# 使用Z-score方法来检测异常值
from scipy import stats
z_scores = np.abs(stats.zscore(df_pred.select_dtypes(include=[np.number])))  # 只对数值型数据计算Z-score
df_cleaned_pred = df_pred[(z_scores < 3).all(axis=1)].copy()  # 删除Z-score大于3的异常值

#选择预测数据中的相关性高的特征，因为我们已经知道了哪些特征的相关性高，直接手动输入
X_pred_white = df_cleaned_pred[['1000hPa_air', '850hPa_air', '1000hPa_azimuth', '1000hPa_number', '1000hPa_wind']]
X_pred_brown = df_cleaned_pred[['1000hPa_air', '850hPa_air', '1000hPa_number', '1000hPa_azimuth']]

# 特征标准化
ss_white_pred = StandardScaler()
X_std_pred_white = ss_white.fit_transform(X_pred_white)
ss_brown_pred = StandardScaler()
X_std_pred_brown = ss_brown.fit_transform(X_pred_brown)

y_pred_lr_formal_white = rf_white.predict(X_std_pred_white)
y_pred_lr_formal_brown = rf_brown.predict(X_std_pred_brown)

df_cleaned_pred['y_pred_lr_formal_white'] = y_pred_lr_formal_white
df_cleaned_pred['y_pred_lr_formal_brown'] = y_pred_lr_formal_brown

# 把时间戳格式的日期转换回日期格式
df_cleaned_pred['日期'] = pd.to_datetime(df_cleaned_pred['日期'], unit='s')  # 单位是秒


selected_column = ['日期', 'y_pred_lr_formal_white', 'y_pred_lr_formal_brown', '1000hPa_air', '850hPa_air', '1000hPa_azimuth', '1000hPa_number', '1000hPa_wind']
df_selected = df_cleaned_pred[selected_column]

df_selected.to_excel(f'predNum{file_name}', index=False)




