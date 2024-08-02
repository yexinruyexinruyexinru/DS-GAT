import matplotlib.pyplot as plt

# 定义数据
x = [1, 2, 3, 4, 5]  # 时间或横坐标
y = [98, 88, 90, 84, 94]  # 对应的纵坐标数据

# 绘制折线图
plt.plot(x, y, marker='o', linestyle='-')

# 添加标题和标签
plt.title('accuracy_rate line chart ')
plt.xlabel('improvement')

# 指定x轴的标签
# plt.xticks(x, ['初步调整参数', '指定输出形式', '添加角色', '添加特定提示词', '赋予较广知识面'])

plt.ylabel('accuracy_rate(%)')

# 显示图形
plt.grid(True)  # 添加网格线
plt.show()

# import matplotlib.pyplot as plt
#
# # 示例数据
# x = ['1', '2', '3', '4', '5']
# y = [1489, 300, 290, 280, 300]
#
# # 绘制柱状图
# plt.bar(x, y)
#
# # 添加标题和标签
# plt.title('time bar chart')
# plt.xlabel('improvement')
# plt.ylabel('time(ms)')
#
# # 显示图形
# plt.show()