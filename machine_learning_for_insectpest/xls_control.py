import pandas as pd
import glob

# 修改为具体的文件路径
file_path = "D:\\JerryHuang\\活动\\创新创业\\利用机器学习解决稻飞虱问题\\project\\Input\\*.csv"

# 读取所有文件
files = glob.glob(file_path)

# 检查是否成功读取到文件
if not files:
    print("没有找到任何CSV文件，请检查文件路径。")
else:
    dataframes = []

    for file in files:
        try:
            df = pd.read_csv(file)
            dataframes.append(df)
        except PermissionError:
            print(f"权限被拒绝，无法读取文件: {file}")
        except Exception as e:
            print(f"读取文件时发生错误: {file}, 错误: {e}")


# 选择要作为连接键的列名
key_columns = ['year', 'month', 'hou']

# 使用pd.concat进行合并
merged_df = pd.concat(dataframes, axis=0, ignore_index=True)

# 根据键列名进行合并，保留重复的行
final_df = merged_df.groupby(key_columns, as_index=False).first()

output_path = 'D:\JerryHuang\活动\创新创业\利用机器学习解决稻飞虱问题\project\dealed_data\西南秀山.csv'
# 输出合并后的数据
final_df.to_csv( output_path, index=False)