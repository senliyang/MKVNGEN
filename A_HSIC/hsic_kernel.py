import numpy as np
import matlab
import matlab.engine
def hsic_kernel_m(kernel_list_d, F_asterisk, nu1, nu2):
    kernel_list_d_format = np.ones((kernel_list_d.shape[1], kernel_list_d.shape[2], kernel_list_d.shape[0]))
    for i in range(kernel_list_d.shape[0]):
        kernel_list_d_format[:, :, i] = kernel_list_d[i]
    engine = matlab.engine.start_matlab()
    weight_d = engine.hsic_kernel_weights_norm(matlab.double(kernel_list_d_format.tolist()),
                                               matlab.double(F_asterisk.to_numpy().tolist()),
                                               matlab.int16([1]),
                                               matlab.double([nu1]),
                                               matlab.double([nu2]))
    engine.exit()
    weight_d = np.array(weight_d)
    weight_d = weight_d.reshape(kernel_list_d.shape[0])
    # np.save('intermediate_result/weight_d.npy', weight_d)
    return weight_d


def hsic_kernel_d(kernel_list_t, F_asterisk, nu1, nu2):
    kernel_list_s_format = np.ones((kernel_list_t.shape[1], kernel_list_t.shape[2], kernel_list_t.shape[0]))
    for i in range(kernel_list_t.shape[0]):
        kernel_list_s_format[:, :, i] = kernel_list_t[i]
    engine = matlab.engine.start_matlab()
    weight_s = engine.hsic_kernel_weights_norm(matlab.double(kernel_list_s_format.tolist()),
                                               matlab.double(F_asterisk.to_numpy().tolist()),
                                               matlab.int16([2]),
                                               matlab.double([nu1]),
                                               matlab.double([nu2]))
    engine.exit()
    weight_s = np.array(weight_s)
    weight_s = weight_s.reshape(kernel_list_t.shape[0])
    # np.save('intermediate_result/weight_s.npy', weight_s)
    return weight_s