import pandas as pd

# 读取数据
df = pd.read_excel('江西 泰和县.xlsx')

# 尝试将日期列转换为 datetime 格式
# 使用 errors='coerce' 来处理无法解析的日期
df['Date'] = pd.to_datetime(df['日期'], errors='coerce')

# 检查转换后的结果
print("转换后的日期列的前几行数据:")
print(df['Date'].head())

# 检查是否有 NaT（缺失值）
print("转换后缺失值数量:")
print(df['Date'].isna().sum())
# 提取年、月、日
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day

# 根据日期分旬，添加一个新列 'Hou'
def assign_hou(day):
    if 1 <= day <= 5:
        return '1'
    elif 6 <= day <= 10:
        return '2'
    elif 11 <= day <= 15:
        return '3'
    elif 16 <= day <= 20:
        return '4'
    elif 21 <= day <= 25:
        return '5'
    else:
        return '6'

df['Hou'] = df['Day'].apply(assign_hou)

df.rename(columns={
    'Year': 'year',
    'Month': 'month',
    'Hou': 'hou'
}, inplace=True)

# 选择需要的特定列，假设你需要的特定列是 '特定列1', '特定列2'等
specific_columns = ['本候灯下白背飞虱虫量（头）', '本侯灯下褐飞虱虫量（头）', '本候灯下稻飞虱合计'] # 请替换为你需要的具体列名

# 创建最终的数据框，包含特定列和重命名后的年月候列
final_df = df[['year', 'month', 'hou'] + specific_columns]

# 保存结果为新的 Excel 文件
final_df.to_csv('江岭泰和.csv', index=False)