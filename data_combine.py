
import pandas as pd

# 合并华南全部地区的数据
df1 = pd.read_excel('dealed_data(xlsx)/华南博白.xlsx')
df2 = pd.read_excel('dealed_data(xlsx)/华南琼海.xlsx')
df3 = pd.read_excel("dealed_data(xlsx)/华南曲江.xlsx")
df4 = pd.read_excel("dealed_data(xlsx)/华南阳春.xlsx")

# 合并两个 DataFrame（竖向拼接）
combined_df = pd.concat([df1, df2, df3, df4], ignore_index=True)

# 保存为新的 xlsx 文件
combined_df.to_excel('dealed_data(xlsx)/combined_华南.xlsx', index=False)

print(combined_df)

# 合并江岭所有地区的数据
df1 = pd.read_excel('dealed_data(xlsx)/江岭徽州.xlsx')
df2 = pd.read_excel('dealed_data(xlsx)/江岭桂阳.xlsx')
df3 = pd.read_excel("dealed_data(xlsx)/江岭泰和.xlsx")
df4 = pd.read_excel("dealed_data(xlsx)/江岭洪江.xlsx")
df5 = pd.read_excel("dealed_data(xlsx)/江岭福清.xlsx")


# 合并两个 DataFrame（竖向拼接）
combined_df = pd.concat([df1, df2, df3, df4, df5], ignore_index=True)

# 保存为新的 xlsx 文件
combined_df.to_excel('dealed_data(xlsx)/combined_江岭.xlsx', index=False)

print(combined_df)


# 合并江淮所有地区的数据
df1 = pd.read_excel('dealed_data(xlsx)/江淮固始.xlsx')
df2 = pd.read_excel('dealed_data(xlsx)/江淮奉贤.xlsx')
df3 = pd.read_excel("dealed_data(xlsx)/江淮盐都.xlsx")
df4 = pd.read_excel("dealed_data(xlsx)/江淮监利.xlsx")

# 合并两个 DataFrame（竖向拼接）
combined_df = pd.concat([df1, df2, df3, df4], ignore_index=True)

# 保存为新的 xlsx 文件
combined_df.to_excel('dealed_data(xlsx)/combined_江淮.xlsx', index=False)

print(combined_df)


# 合并西南所有地区的数据
df1 = pd.read_excel('dealed_data(xlsx)/西南丘北.xlsx')
df2 = pd.read_excel('dealed_data(xlsx)/西南大竹.xlsx')
df3 = pd.read_excel("dealed_data(xlsx)/西南秀山.xlsx")
df4 = pd.read_excel("dealed_data(xlsx)/西南锦屏.xlsx")

# 合并两个 DataFrame（竖向拼接）
combined_df = pd.concat([df1, df2, df3, df4], ignore_index=True)

# 保存为新的 xlsx 文件
combined_df.to_excel('dealed_data(xlsx)/combined_西南.xlsx', index=False)

print(combined_df)



# 合并华南全部地区的ssp126数据
df1 = pd.read_csv('cmip_Nor/SSP1/华南博白ssp126.xlsx')
df2 = pd.read_csv('cmip_Nor/SSP1/华南曲江ssp126.xlsx')
df3 = pd.read_csv("cmip_Nor/SSP1/华南琼海ssp126.xlsx")
df4 = pd.read_csv("cmip_Nor/SSP1/华南阳春ssp126.xlsx")

# 合并两个 DataFrame（竖向拼接）
combined_df = pd.concat([df1, df2, df3, df4], ignore_index=True)

# 保存为新的 xlsx 文件
combined_df.to_excel('cmip_Nor/SSP1/combined_华南ssp126.xlsx', index=False)

print(combined_df)

# 合并江岭所有地区的ssp126数据
df1 = pd.read_csv('cmip_Nor/SSP1/江岭徽州ssp126.xlsx')
df2 = pd.read_csv('cmip_Nor/SSP1/江岭桂阳ssp126.xlsx')
df3 = pd.read_csv("cmip_Nor/SSP1/江岭泰和ssp126.xlsx")
df4 = pd.read_csv("cmip_Nor/SSP1/江岭福清ssp126.xlsx")
df5 = pd.read_csv("cmip_Nor/SSP1/江岭洪江ssp126.xlsx")
# 合并两个 DataFrame（竖向拼接）
combined_df = pd.concat([df1, df2, df3, df4, df5], ignore_index=True)

# 保存为新的 xlsx 文件
combined_df.to_excel('cmip_Nor/SSP1/combined_江岭ssp126.xlsx', index=False)

print(combined_df)


# 合并江淮全部地区的ssp126数据
df1 = pd.read_csv('cmip_Nor/SSP1/江淮固始ssp126.xlsx')
df2 = pd.read_csv('cmip_Nor/SSP1/江淮奉贤ssp126.xlsx')
df3 = pd.read_csv("cmip_Nor/SSP1/江淮盐都ssp126.xlsx")
df4 = pd.read_csv("cmip_Nor/SSP1/江淮监利ssp126.xlsx")

# 合并两个 DataFrame（竖向拼接）
combined_df = pd.concat([df1, df2, df3, df4], ignore_index=True)

# 保存为新的 xlsx 文件
combined_df.to_excel('cmip_Nor/SSP1/combined_江淮ssp126.xlsx', index=False)

print(combined_df)

# 合并西南全部地区的ssp126数据
df1 = pd.read_csv('cmip_Nor/SSP1/西南丘北ssp126.xlsx')
df2 = pd.read_csv('cmip_Nor/SSP1/西南大竹ssp126.xlsx')
df3 = pd.read_csv("cmip_Nor/SSP1/西南秀山ssp126.xlsx")
df4 = pd.read_csv("cmip_Nor/SSP1/西南锦屏ssp126.xlsx")

# 合并两个 DataFrame（竖向拼接）
combined_df = pd.concat([df1, df2, df3, df4], ignore_index=True)

# 保存为新的 xlsx 文件
combined_df.to_excel('cmip_Nor/SSP1/combined_西南ssp126.xlsx', index=False)

print(combined_df)


#合并华南全部地区的ssp245数据
df1 = pd.read_csv('cmip_Nor/SSP2/华南博白ssp245.xlsx')
df2 = pd.read_csv('cmip_Nor/SSP2/华南曲江ssp245.xlsx')
df3 = pd.read_csv("cmip_Nor/SSP2/华南琼海ssp245.xlsx")
df4 = pd.read_csv("cmip_Nor/SSP2/华南阳春ssp245.xlsx")

# 合并两个 DataFrame（竖向拼接）
combined_df = pd.concat([df1, df2, df3, df4], ignore_index=True)

# 保存为新的 xlsx 文件
combined_df.to_excel('cmip_Nor/SSP2/combined_华南ssp245.xlsx', index=False)

print(combined_df)

# 合并江岭所有地区的ssp245数据
df1 = pd.read_csv('cmip_Nor/SSP2/江岭徽州ssp245.xlsx')
df2 = pd.read_csv('cmip_Nor/SSP2/江岭桂阳ssp245.xlsx')
df3 = pd.read_csv("cmip_Nor/SSP2/江岭泰和ssp245.xlsx")
df4 = pd.read_csv("cmip_Nor/SSP2/江岭福清ssp245.xlsx")
df5 = pd.read_csv("cmip_Nor/SSP2/江岭洪江ssp245.xlsx")
# 合并两个 DataFrame（竖向拼接）
combined_df = pd.concat([df1, df2, df3, df4, df5], ignore_index=True)

# 保存为新的 xlsx 文件
combined_df.to_excel('cmip_Nor/SSP2/combined_江岭ssp245.xlsx', index=False)

print(combined_df)


# 合并江淮全部地区的ssp245数据
df1 = pd.read_csv('cmip_Nor/SSP2/江淮固始ssp245.xlsx')
df2 = pd.read_csv('cmip_Nor/SSP2/江淮奉贤ssp245.xlsx')
df3 = pd.read_csv("cmip_Nor/SSP2/江淮盐都ssp245.xlsx")
df4 = pd.read_csv("cmip_Nor/SSP2/江淮监利ssp245.xlsx")

# 合并两个 DataFrame（竖向拼接）
combined_df = pd.concat([df1, df2, df3, df4], ignore_index=True)

# 保存为新的 xlsx 文件
combined_df.to_excel('cmip_Nor/SSP2/combined_江淮ssp245.xlsx', index=False)

print(combined_df)

# 合并西南全部地区的ssp245数据
df1 = pd.read_csv('cmip_Nor/SSP2/西南丘北ssp245.xlsx')
df2 = pd.read_csv('cmip_Nor/SSP2/西南大竹ssp245.xlsx')
df3 = pd.read_csv("cmip_Nor/SSP2/西南秀山ssp245.xlsx")
df4 = pd.read_csv("cmip_Nor/SSP2/西南锦屏ssp245.xlsx")

# 合并两个 DataFrame（竖向拼接）
combined_df = pd.concat([df1, df2, df3, df4], ignore_index=True)

# 保存为新的 xlsx 文件
combined_df.                                                                                                                                          to_excel('cmip_Nor/SSP2/combined_西南ssp245.xlsx', index=False)

print(combined_df)


#合并华南全部地区的ssp370数据
df1 = pd.read_csv('cmip_Nor/SSP3/华南博白ssp370.xlsx')
df2 = pd.read_csv('cmip_Nor/SSP3/华南曲江ssp370.xlsx')
df3 = pd.read_csv("cmip_Nor/SSP3/华南琼海ssp370.xlsx")
df4 = pd.read_csv("cmip_Nor/SSP3/华南阳春ssp370.xlsx")

# 合并两个 DataFrame（竖向拼接）
combined_df = pd.concat([df1, df2, df3, df4], ignore_index=True)

# 保存为新的 xlsx 文件
combined_df.to_excel('cmip_Nor/SSP3/combined_华南ssp370.xlsx', index=False)

print(combined_df)

# 合并江岭所有地区的ssp370数据
df1 = pd.read_csv('cmip_Nor/SSP3/江岭徽州ssp370.xlsx')
df2 = pd.read_csv('cmip_Nor/SSP3/江岭桂阳ssp370.xlsx')
df3 = pd.read_csv("cmip_Nor/SSP3/江岭泰和ssp370.xlsx")
df4 = pd.read_csv("cmip_Nor/SSP3/江岭福清ssp370.xlsx")
df5 = pd.read_csv("cmip_Nor/SSP3/江岭洪江ssp370.xlsx")
# 合并两个 DataFrame（竖向拼接）
combined_df = pd.concat([df1, df2, df3, df4, df5], ignore_index=True)

# 保存为新的 xlsx 文件
combined_df.to_excel('cmip_Nor/SSP3/combined_江岭ssp370.xlsx', index=False)

print(combined_df)


# 合并江淮全部地区的ssp370数据
df1 = pd.read_csv('cmip_Nor/SSP3/江淮固始ssp370.xlsx')
df2 = pd.read_csv('cmip_Nor/SSP3/江淮奉贤ssp370.xlsx')
df3 = pd.read_csv("cmip_Nor/SSP3/江淮盐都ssp370.xlsx")
df4 = pd.read_csv("cmip_Nor/SSP3/江淮监利ssp370.xlsx")

# 合并两个 DataFrame（竖向拼接）
combined_df = pd.concat([df1, df2, df3, df4], ignore_index=True)

# 保存为新的 xlsx 文件
combined_df.to_excel('cmip_Nor/SSP3/combined_江淮ssp370.xlsx', index=False)

print(combined_df)

# 合并西南全部地区的ssp370数据
df1 = pd.read_csv('cmip_Nor/SSP3/西南丘北ssp370.xlsx')
df2 = pd.read_csv('cmip_Nor/SSP3/西南大竹ssp370.xlsx')
df3 = pd.read_csv("cmip_Nor/SSP3/西南秀山ssp370.xlsx")
df4 = pd.read_csv("cmip_Nor/SSP3/西南锦屏ssp370.xlsx")

# 合并两个 DataFrame（竖向拼接）
combined_df = pd.concat([df1, df2, df3, df4], ignore_index=True)

# 保存为新的 xlsx 文件
combined_df.to_excel('cmip_Nor/SSP3/combined_西南ssp370.xlsx', index=False)

print(combined_df)


#合并华南全部地区的ssp585数据
df1 = pd.read_csv('cmip_Nor/SSP4/华南博白ssp585.xlsx')
df2 = pd.read_csv('cmip_Nor/SSP4/华南曲江ssp585.xlsx')
df3 = pd.read_csv("cmip_Nor/SSP4/华南琼海ssp585.xlsx")
df4 = pd.read_csv("cmip_Nor/SSP4/华南阳春ssp585.xlsx")

# 合并两个 DataFrame（竖向拼接）
combined_df = pd.concat([df1, df2, df3, df4], ignore_index=True)

# 保存为新的 xlsx 文件
combined_df.to_excel('cmip_Nor/SSP4/combined_华南ssp585.xlsx', index=False)

print(combined_df)

# 合并江岭所有地区的ssp585数据
df1 = pd.read_csv('cmip_Nor/SSP4/江岭徽州ssp585.xlsx')
df2 = pd.read_csv('cmip_Nor/SSP4/江岭桂阳ssp585.xlsx')
df3 = pd.read_csv("cmip_Nor/SSP4/江岭泰和ssp585.xlsx")
df4 = pd.read_csv("cmip_Nor/SSP4/江岭福清ssp585.xlsx")
df5 = pd.read_csv("cmip_Nor/SSP4/江岭洪江ssp585.xlsx")
# 合并两个 DataFrame（竖向拼接）
combined_df = pd.concat([df1, df2, df3, df4, df5], ignore_index=True)

# 保存为新的 xlsx 文件
combined_df.to_excel('cmip_Nor/SSP4/combined_江岭ssp585.xlsx', index=False)

print(combined_df)


# 合并江淮全部地区的ssp370数据
df1 = pd.read_csv('cmip_Nor/SSP4/江淮固始ssp585.xlsx')
df2 = pd.read_csv('cmip_Nor/SSP4/江淮奉贤ssp585.xlsx')
df3 = pd.read_csv("cmip_Nor/SSP4/江淮盐都ssp585.xlsx")
df4 = pd.read_csv("cmip_Nor/SSP4/江淮监利ssp585.xlsx")

# 合并两个 DataFrame（竖向拼接）
combined_df = pd.concat([df1, df2, df3, df4], ignore_index=True)

# 保存为新的 xlsx 文件
combined_df.to_excel('cmip_Nor/SSP4/combined_江淮ssp585.xlsx', index=False)

print(combined_df)

# 合并西南全部地区的ssp585数据
df1 = pd.read_csv('cmip_Nor/SSP4/西南丘北ssp585.xlsx')
df2 = pd.read_csv('cmip_Nor/SSP4/西南大竹ssp585.xlsx')
df3 = pd.read_csv("cmip_Nor/SSP4/西南秀山ssp585.xlsx")
df4 = pd.read_csv("cmip_Nor/SSP4/西南锦屏ssp585.xlsx")

# 合并两个 DataFrame（竖向拼接）
combined_df = pd.concat([df1, df2, df3, df4], ignore_index=True)

# 保存为新的 xlsx 文件
combined_df.to_excel('cmip_Nor/SSP4/combined_西南ssp585.xlsx', index=False)

print(combined_df)