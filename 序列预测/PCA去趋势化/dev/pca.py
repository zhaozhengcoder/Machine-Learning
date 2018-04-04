import numpy as np

class PCA:
    def __init__(self, x, retain):
        # X=m*n m为样本个数，n为样本的维度
        # X必须均值为0,只有在均值为0时，协方差sigma=xT*x=[n,m]*[m,n]=[n,n]
        self.x = x
        m, n = x.shape
        sigma = np.matmul(self.x.T, self.x) / m  # sigma是协方差矩阵，T是转置,sigma=xT*x=[n,m]*[m,n]=[n,n]
        self.u, self.s, _ = np.linalg.svd(sigma)  # u的每一列都是特征向量，s是特征值与u一一对应
        # here iteration is over rows but the columns are the eigenvectors of sigma
        # u_sum = np.cumsum(self.s)
        self.retain_num = retain
        # for i in range(n):
        #     if u_sum[i] / u_sum[-1] >= retain:
        #         self.retain_num = i
        #         break
        self.main_vector = self.u[:, 0:self.retain_num + 1]
        self.rest_vector = self.u[:, self.retain_num + 1:]
        main_x_rot = np.matmul(self.x, self.main_vector)
        rest_x_rot = np.matmul(self.x, self.rest_vector)
        self.main_x = np.matmul(main_x_rot, self.main_vector.T)
        self.rest_x = np.matmul(rest_x_rot, self.rest_vector.T)

    def reduce(self, data):
        main_data_rot = np.matmul(data, self.main_vector)
        rest_data_rot = np.matmul(data, self.rest_vector)
        data_main = np.matmul(main_data_rot, self.main_vector.T)
        data_rest = np.matmul(rest_data_rot, self.rest_vector.T)
        return data_main, data_rest

    def reconstruct(self, rest_x):
        return self.main_x + rest_x