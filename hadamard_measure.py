import torch
import numpy as np
from scipy.io import savemat
class CCHadamardSensor(object):
    def __init__(self, shapey, shapex,
                 low_freq_ratio=1
                 ):
        self.shapeOI = (shapey, shapex)
        self.n_pixels = np.prod(shapex)
        a = int(np.sqrt(self.n_pixels))

        num_list = []
        for row_idx in range(self.n_pixels):
            h_row = self._generate_cchadamard_row(row_idx, a)
            first_row = h_row[:a]
            first_col = h_row[::a]
            h_changes = np.sum(np.diff(first_row) != 0)
            v_changes = np.sum(np.diff(first_col) != 0)
            num_list.append((h_changes + 1) * (v_changes + 1))

        sorted_indices = np.argsort(num_list)
        self.k = np.prod(shapey)

        m = int(low_freq_ratio * self.k)
        m = max(0, min(m, self.k))
        remaining_k = self.k - m

        low_freq_indices = sorted_indices[:m]

        if remaining_k > 0:
            high_freq_pool = sorted_indices[m:]
            prob_weights = 1.0 / (np.array([num_list[i] for i in high_freq_pool]) + 1)
            prob_weights /= prob_weights.sum()

            try:
                selected_high = np.random.choice(high_freq_pool, size=remaining_k,
                                                 replace=False, p=prob_weights)
            except ValueError:
                selected_high = np.random.choice(high_freq_pool, size=remaining_k,
                                                 replace=True, p=prob_weights)

            selected_indices = np.concatenate([low_freq_indices, selected_high])
        else:
            selected_indices = low_freq_indices

        self.H = np.zeros((self.k, self.n_pixels), dtype=np.float32)
        for i, idx in enumerate(selected_indices):
            self.H[i, :] = self._generate_cchadamard_row(idx, a).astype(np.float32)

        self.H_tf = torch.tensor(self.H, dtype=torch.float32).cuda()
        self.shape = self.H.shape
        print('Measurement matrix shape:', self.H.shape)

    def _generate_cchadamard_row(self, row_idx, a):
        if not hasattr(self, '_hadamard_cache'):
            self._hadamard_cache = {}

        cache_key = (row_idx, a)
        if cache_key in self._hadamard_cache:
            return self._hadamard_cache[cache_key]

        if a == 1:
            result = np.array([1], dtype=np.int8)
        else:
            half = a // 2
            sub_row = self._generate_cchadamard_row(row_idx % (half ** 2), half)
            quadrant = row_idx // (half ** 2)

            if quadrant == 0:
                result = np.tile(sub_row, 4)
            elif quadrant == 1:
                result = np.tile(np.hstack([sub_row, -sub_row]), 2)
            elif quadrant == 2:
                result = np.hstack([sub_row, sub_row, -sub_row, -sub_row])
            else:
                result = np.hstack([sub_row, -sub_row, -sub_row, sub_row])

        self._hadamard_cache[cache_key] = result
        return result

    def __call__(self, x):
        assert x.shape[1] == 3
        x = torch.squeeze(x)
        x = (x + 1) / 2

        y_rgb = torch.zeros((self.H_tf.shape[0], 3),
                   dtype=torch.float32, device=x.device)

        for i in range(3):
            x_flat = x[i].reshape(-1, 1)
            y_channel = torch.matmul(self.H_tf, x_flat)
            y_rgb[:, i] = y_channel.squeeze()

        return y_rgb

    def trans_torch(self, y):
        x_hat = torch.matmul(self.H_tf.T, y)
        return x_hat.reshape(self.shapeOI[1])

    def trans_torch_3(self, y):
        image_ATy = torch.zeros((3, *self.shapeOI[1]),
                        dtype=torch.float32, device=y.device)
        for i in range(3):
            image_ATy[i] = self.trans_torch(y[:, i])
        return image_ATy

    def trans_np(self, y):
        x_hat = np.matmul(self.H.T, y)
        return x_hat.reshape(self.shapeOI[1])

    def trans_np_3(self, y):
        image_ATy = np.zeros((3, *self.shapeOI[1]), dtype=np.float32)
        for i in range(3):
            image_ATy[i] = self.trans_np(y[:, i])
        return image_ATy

    # 保持其他辅助方法不变
    def _np(self, x):
        assert x.shape[0] == 3
        x = (x + 1) / 2

        y_rgb = np.zeros((self.H.shape[0], 3), dtype=np.float32)  # 修改尺寸
        for i in range(3):
            x_flat = np.reshape(x[i, :, :], [-1, 1])
            y_rgb[:, i] = np.matmul(self.H, x_flat).squeeze()
        return y_rgb

    def adj_np(self, y):
        y_flat = y.flatten()
        x_hat = self.H.T @ y_flat
        return x_hat.reshape(self.shapeOI[1])