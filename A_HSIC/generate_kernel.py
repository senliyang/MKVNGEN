import numpy as np
import pandas as pd
from hsic_kernel import hsic_kernel_m
from hsic_kernel import hsic_kernel_d
from kernel_normalized import kernel_normalized
def generate_kernel_m(cv):
    kernel_microbe_fun = pd.read_csv('fun_MS.txt', sep='\t', header=None)
    # print(kernel_microbe_fun.shape)
    kernel_microbe_cos = pd.read_csv('cos_MS.txt', sep='\t', header=None)
    # print(kernel_microbe_cos.shape)
    kernel_microbe_gas = pd.read_csv('Ga_MS.txt', sep='\t', header=None)
    # print(kernel_microbe_gas.shape)
    kernel_list = np.array([kernel_microbe_fun,
                            kernel_microbe_cos,
                            kernel_microbe_gas])
    return kernel_list
def generate_kernel_d(cv):
    kernel_disease_sem = pd.read_csv('sem_DS.txt', sep='\t', header=None)
    # print(kernel_disease_sem.shape)
    kernel_disease_cos = pd.read_csv('cos_DS.txt', sep='\t', header=None)
    # print(kernel_disease_cos.shape)
    kernel_disease_gas = pd.read_csv('Ga_DS.txt', sep='\t', header=None)
    # print(kernel_disease_gas.shape)

    kernel_list = np.array([kernel_disease_sem,
                            kernel_disease_cos,
                            kernel_disease_gas])
    return kernel_list
def use_weight_combine_kernel_d(kernel_list_d, weight_d):
    weight_d = weight_d.reshape(weight_d.shape[0], 1, 1)
    kernel_d = kernel_list_d * weight_d
    kernel_d = np.sum(kernel_d, axis=0)

    return kernel_d
def use_weight_combine_kernel_s(kernel_list_s, weight_s):
    weight_s = weight_s.reshape(weight_s.shape[0], 1, 1)
    kernel_s = kernel_list_s * weight_s
    kernel_s = np.sum(kernel_s, axis=0)

    return kernel_s
if __name__ == "__main__":
    for i in range(1):

        F_asterisk= pd.read_csv('MD_A.csv', sep=',', header=0, index_col=0)

        kernel_list_d = generate_kernel_m(i)
        weight_d = hsic_kernel_m(kernel_list_d, F_asterisk, 0.01, 0.001)
        print(i, weight_d)
        d = use_weight_combine_kernel_d(kernel_list_d, weight_d)
        d = pd.DataFrame(d)  # 确保 d 是 DataFrame 类型
        d.to_csv('m_kernel.csv', index=False,header=False)  # 指定文件名，并选择是否包含索引
        d = kernel_normalized(d)
        d = pd.DataFrame(d)  # 确保 d 是 DataFrame 类型
        d.to_csv('m_kernel_normalized.csv', index=False,header=False)  # 指定文件名，并选择是否包含索引

        kernel_list_t = generate_kernel_d(i)
        weight_t = hsic_kernel_d(kernel_list_t, F_asterisk, 0.01, 0.001)
        print(i, weight_t)
        t = use_weight_combine_kernel_s(kernel_list_t, weight_t)
        t = pd.DataFrame(t)  # 确保 d 是 DataFrame 类型
        t.to_csv('d_kernel.csv', index=False,header=False)  # 指定文件名，并选择是否包含索引
        t = kernel_normalized(t)
        t = pd.DataFrame(t)  # 确保 d 是 DataFrame 类型
        t.to_csv('d_kernel_normalized.csv', index=False,header=False)  # 指定文件名，并选择是否包含索引