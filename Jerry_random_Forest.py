import pandas as pd
import numpy as np
import xlsx
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, r2_score


#导入数据
df = pd.read_excel('dealed_data(xlsx)/combined_华南.xlsx')

df['首迁期'] = pd.to_datetime(df['首迁期'])
#把csv文件里面的时间类型的列转换为浮点类型的秒数时间戳
df['首迁期'] = df['首迁期'].apply(lambda x: x.timestamp())
df['日期'] = df['日期'].apply(lambda x: x.timestamp())
print(df.dtypes)
target1 = pd.read_csv("cmip_Nor/SSP1/combined_华南ssp126.xlsx")
target2 = pd.read_csv("cmip_Nor/SSP2/combined_华南ssp245.xlsx")
target3 = pd.read_csv("cmip_Nor/SSP3/combined_华南ssp370.xlsx")
target4 = pd.read_csv("cmip_Nor/SSP3/combined_华南ssp370.xlsx")


X = df.drop(["首迁期",  "日期-首迁期", "本候灯下白背飞虱虫量（头）", "本侯灯下褐飞虱虫量（头）", "本候灯下稻飞虱合计"],axis=1,inplace=False)
y = df["日期-首迁期"]


print(X.head)
print(y.head)

#将数据输入到机器学习中，使用了随机森林和长短期记忆两种方法

ss = StandardScaler()
X_std = ss.fit_transform(X)

df_X_std = pd.DataFrame(X_std)

X_train, X_test, y_train, y_test = train_test_split(df_X_std, y, test_size=0.2, random_state=0)

rf = RandomForestRegressor(n_estimators=150, max_depth=10, random_state=0)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
print(y_pred)
print(r2_score(y_test, y_pred))
print(mean_squared_error(y_test, y_pred))


print("Use svm_modle /n")
svm_model = SVR(kernel='poly', degree=3, gamma='auto')
svm_model.fit(X_train, y_train)

y_pred = svm_model.predict(X_test)
print(r2_score(y_test, y_pred))
print(mean_squared_error(y_test, y_pred))



