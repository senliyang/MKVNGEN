# -*- coding: utf-8 -*-
# @Time    : 2024/08/29
# @Author  : ***
# @E-mail  : ***

import numpy as np
import matplotlib.pyplot as plt

d = [1.06803090e-07 ,2.71472476e-07 ,9.99999622e-01]

plt.figure(figsize=(8, 5.5))

bars = plt.bar(['$DSS$', '$DCS$', '$DGS$'], d, color='#CD4F39')

# 为每个柱状图添加数值标签
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.8f}',
             ha='center', va='bottom', fontsize=15, color='black')

plt.yticks(np.linspace(0.1, 1.0, 10), fontsize=18)

plt.xlabel('disease Space', fontsize=18)
plt.ylabel('Weight', fontsize=18)
plt.xticks(fontsize=15)

plt.savefig(r"disease_weight.png", dpi=300,transparent=True)
plt.show()