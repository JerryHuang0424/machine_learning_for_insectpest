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
df = pd.read_excel('气象地点+虫情地点数据/Step1_丘北-锦屏.xlsx')
df_5 = pd.read_excel('气象地点+虫情地点数据/Step1_丘北-锦屏(气象数据前移5天).xlsx')
df_10 = pd.read_excel('气象地点+虫情地点数据/Step1_丘北-锦屏(气象数据前移10天).xlsx')
df_15 = pd.read_excel('气象地点+虫情地点数据/Step1_丘北-锦屏(气象数据前移15天).xlsx')

df['date'] = df['date'].apply(lambda x: x.timestamp())
df_5['date'] = df_5['date'].apply(lambda x: x.timestamp())
df_10['date'] = df_10['date'].apply(lambda x: x.timestamp())
df_15['date'] = df_15['date'].apply(lambda x: x.timestamp())

df['白背飞虱'] = np.log(df['白背飞虱'] + 1)
df_5['白背飞虱'] = np.log(df_5['白背飞虱'] + 1)
df_10['白背飞虱'] = np.log(df_10['白背飞虱'] + 1)
df_15['白背飞虱'] = np.log(df_15['白背飞虱'] + 1)

df['褐飞虱'] = np.log(df['褐飞虱'] + 1)
df_5['褐飞虱'] = np.log(df_5['褐飞虱'] + 1)
df_10['褐飞虱'] = np.log(df_10['褐飞虱'] + 1)
df_15['褐飞虱'] = np.log(df_15['褐飞虱'] + 1)


print(df.head())


# 相关性矩阵
correlation_matrix = df.corr()
# 绘制热力图，查看各特征与目标变量的相关性
selected_rows_cols = correlation_matrix.loc[['date', 'year', 'month', 'hou', '1000hPa_air', '925hPa_air', '850hPa_air', '1000hPa_rhum', '925hPa_rhum', '850hPa_rhum', '1000hPa_wind', '925hPa_wind', '850hPa_wind', '1000hPa_azimuth', '925hPa_azimuth', '850hPa_azimuth', '1000hPa_number', '925hPa_number', '850hPa_number', '1000hPa_omega', '925hPa_omega', '850hPa_omega'], ['白背飞虱', '褐飞虱']]
plt.figure(figsize=(20, 12))
sns.heatmap(selected_rows_cols, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Correlation Matrix")
plt.show()

correlation_matrix = df_5.corr()
# 绘制热力图，查看各特征与目标变量的相关性
selected_rows_cols = correlation_matrix.loc[['date', 'year', 'month', 'hou', '1000hPa_air', '925hPa_air', '850hPa_air', '1000hPa_rhum', '925hPa_rhum', '850hPa_rhum', '1000hPa_wind', '925hPa_wind', '850hPa_wind', '1000hPa_azimuth', '925hPa_azimuth', '850hPa_azimuth', '1000hPa_number', '925hPa_number', '850hPa_number', '1000hPa_omega', '925hPa_omega', '850hPa_omega'], ['白背飞虱', '褐飞虱']]
plt.figure(figsize=(20, 12))
sns.heatmap(selected_rows_cols, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Correlation Matrix")
plt.show()


correlation_matrix = df_10.corr()
# 绘制热力图，查看各特征与目标变量的相关性
selected_rows_cols = correlation_matrix.loc[['date', 'year', 'month', 'hou', '1000hPa_air', '925hPa_air', '850hPa_air', '1000hPa_rhum', '925hPa_rhum', '850hPa_rhum', '1000hPa_wind', '925hPa_wind', '850hPa_wind', '1000hPa_azimuth', '925hPa_azimuth', '850hPa_azimuth', '1000hPa_number', '925hPa_number', '850hPa_number', '1000hPa_omega', '925hPa_omega', '850hPa_omega'], ['白背飞虱', '褐飞虱']]
plt.figure(figsize=(20, 12))
sns.heatmap(selected_rows_cols, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Correlation Matrix")
plt.show()

correlation_matrix = df_15.corr()
# 绘制热力图，查看各特征与目标变量的相关性
selected_rows_cols = correlation_matrix.loc[['date', 'year', 'month', 'hou', '1000hPa_air', '925hPa_air', '850hPa_air', '1000hPa_rhum', '925hPa_rhum', '850hPa_rhum', '1000hPa_wind', '925hPa_wind', '850hPa_wind', '1000hPa_azimuth', '925hPa_azimuth', '850hPa_azimuth', '1000hPa_number', '925hPa_number', '850hPa_number', '1000hPa_omega', '925hPa_omega', '850hPa_omega'], ['白背飞虱', '褐飞虱']]
plt.figure(figsize=(20, 12))
sns.heatmap(selected_rows_cols, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Correlation Matrix")
plt.show()