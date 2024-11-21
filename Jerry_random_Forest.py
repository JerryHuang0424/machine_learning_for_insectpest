import pandas as pd
import numpy as np
import csv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('dealed_data(xlsx)/combined_华南.csv')
df['首迁期'] = pd.to_datetime(df['首迁期'])
#把csv文件里面的时间类型的列转换为浮点类型的秒数时间戳
df['首迁期'] = df['首迁期'].apply(lambda x: x.timestamp())
print(df.dtypes)
target1 = pd.read_csv("cmip_Nor/SSP1/combined_华南ssp126.csv")
target2 = pd.read_csv("cmip_Nor/SSP2/combined_华南ssp245.csv")
target3 = pd.read_csv("cmip_Nor/SSP3/combined_华南ssp370.csv")
target4 = pd.read_csv("cmip_Nor/SSP3/combined_华南ssp370.csv")


X = df.drop(["首迁期", "本候灯下白背飞虱虫量（头）", "本侯灯下褐飞虱虫量（头）", "本候灯下稻飞虱合计", "year", "month", "hou"],axis=1,inplace=False)
y = df["本侯灯下褐飞虱虫量（头）"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=5)

rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print(y_pred)
print(r2_score(y_test, y_pred))
print(mean_squared_error(y_test, y_pred))