# -*- coding: utf-8 -*-
# @Time    : 2024/08/29
# @Author  : ***
# @E-mail  : ***

import numpy as np
import matplotlib.pyplot as plt

m = [0.39377477, 0.49973613, 0.10648911]

plt.figure(figsize=(8, 5.5))

bars = plt.bar(['$MFS$', '$MCS$', '$MGS$'], m, color='#352A86')

# 为每个柱状图添加数值标签
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.8f}',
             ha='center', va='bottom', fontsize=15, color='black')

# 设置 y 轴刻度、标签和字体大小
plt.yticks(np.linspace(0.1, 1.0, 10), fontsize=18)
plt.xlabel('microbe Space', fontsize=18)
plt.ylabel('Weight', fontsize=18)
plt.xticks(fontsize=15)

# 保存图像
plt.savefig(r"microbe_weight.png", dpi=300, transparent=True)
plt.show()
