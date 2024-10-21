import pandas as pd

train = pd.read_excel('华南博白.xlsx')

print(train.head())
filtered_melbourne_data = train.dropna(axis=0)


# Choose target and features
output =['本候灯下白背飞虱虫量（头）', '本侯灯下褐飞虱虫量（头）', '本候灯下稻飞虱合计']
y = filtered_melbourne_data[output]
melbourne_features = ['year', 'month' ,'hou', '首迁期', '1000hPa_air', '925hPa_air', '850hPa_air', '1000hPa_omega',
                        '925hPa_omega', '850hPa_omega', '1000hPa_rhum', '925hPa_rhum', '850hPa_rhum',
                      '1000hPa_wind', '925hPa_wind', '850hPa_wind', '1000hPa_azimuth', '925hPa_azimuth', "850hPa_azimuth",
                      '1000hPa_number', '925hPa_number', '850hPa_number',]
X = filtered_melbourne_data[melbourne_features]

# 检查X和y的数据类型
print(X.dtypes)
print(y.dtypes)

# 确保 '首迁期' 列为 datetime 类型，再转换为 int64 类型的时间戳
if pd.api.types.is_datetime64_any_dtype(X['首迁期']):
    X.loc[:, '首迁期'] = X['首迁期'].astype('int64')

# 重新处理NaN值
X = X.dropna(axis=0)
y = y.loc[X.index]  # 保证X和y的行数一致

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error


# split data into training and validation data, for both features and target
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we
# run this script.
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
# Define model
melbourne_model = DecisionTreeRegressor()
# Fit model
melbourne_model.fit(train_X, train_y)

# get predicted prices on validation data
val_predictions = melbourne_model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))