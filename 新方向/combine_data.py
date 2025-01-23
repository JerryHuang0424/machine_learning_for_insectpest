import os
import pandas as pd

# 读取Excel文件
import pandas as pd

# 读取虫情数据
df_jingPin = pd.read_excel('虫情原始数据/西南稻区/贵州 锦屏县.xlsx')
print(df_jingPin.head())

# 提取列创建新的DataFrame
df_selected = df_jingPin[["date", "本候灯下白背飞虱虫量（头）", "本侯灯下褐飞虱虫量\n（头）"]].copy()
print(df_selected.head())

# 读取气象数据
df_climate = pd.read_excel('气象地点+虫情地点数据/丘北-climate.xlsx')

# 将df_selected中的date列转换为日期时间类型（这里假设原格式是常见的日期字符串格式，比如'YYYY-MM-DD'，如果不是需要调整解析格式）
df_selected['date'] = pd.to_datetime(df_selected['date'])

# 进行左连接合并
df_combined_left = pd.merge(df_climate, df_selected, on='date', how='left')

# 保存合并后的数据到Excel
df_combined_left.to_excel("Step1_丘北-锦屏.xlsx", index=False)