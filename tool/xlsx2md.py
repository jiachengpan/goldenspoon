import pandas as pd
file_name = 'JY_Linear_Model_Analysis-2'
# 数据文件
excel_file = file_name+".xlsx"

excel = pd.read_excel(excel_file)              # 读取Excel表格

excel_table_head = list(excel.columns.values)  # 读取表头
table_head = '|'.join(excel_table_head) + "\n" # 拼接表头

# 获取表格主体
excel_table_body = list(excel.iloc[0:].values)

new_table_body = []

# 将每一个列表项转换为字符串
for i in excel_table_body:
    row = []
    for j in i:             # 对这一行的遍历
        row.append(str(j))  # 转换为字符串并加入row列表
    
    new_table_body.append(row)  # 再将row加入new_table_body

# 拼接列表主体
table_body = '\n'.join(['|'.join(i) for i in new_table_body])

# 制作列表分隔符
table_split = '-|' * len(excel_table_head) + "\n"

# 拼接成table变量
table = table_head + table_split + table_body

# 输出到文件
with open(file_name+".md", "w", encoding="UTF-8") as f:
    f.write(table)
