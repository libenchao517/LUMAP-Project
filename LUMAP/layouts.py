################################################################################
# 本文件用于实现LUMAP和2DUMAP算法的优化
################################################################################
# 导入模块
import numpy as np
import numba
from LUMAP.utils import tau_rand_int
from tqdm.auto import tqdm
################################################################################
@numba.njit()
def clip(val):
    """
    梯度截断
    Standard clamping of Data_size value into Data_size fixed range (in this case -4.0 to 4.0)
    Parameters
    ----------
    val: float
        The value to be clamped.

    Returns
    -------
    The clamped value, now fixed to be in the range -4.0 to 4.0.
    """
    if val > 4.0:
        return 4.0
    elif val < -4.0:
        return -4.0
    else:
        return val
################################################################################
@numba.njit()
def matrix_multiply(a, b):
    """
    加速矩阵乘法
    :param a:
    :param b:
    :return:
    """
    return a @ b
################################################################################
@numba.njit(
    "f4(f4[::1],f4[::1])",
    fastmath=True,
    cache=True,
    locals={
        "result": numba.types.float32,
        "diff": numba.types.float32,
        "dim": numba.types.intp,
        "i": numba.types.intp,
    },
)
def rdist(x, y):
    """
    欧氏距离的平方
    Reduced Euclidean distance.
    Parameters
    ----------
    x: array of shape (embedding_dim,)
    y: array of shape (embedding_dim,)

    Returns
    -------
    The squared euclidean distance between x and y
    """
    result = 0.0
    dim = x.shape[0]
    for i in range(dim):
        diff = x[i] - y[i]
        result += diff * diff
    return result
################################################################################
def _optimize_layout_linear_euclidean_single_epoch(
        map_matrix,
        data,
        head,
        tail,
        n_vertices,
        epochs_per_sample,
        a,
        b,
        rng_state,
        gamma,
        dim,
        move_other,
        alpha,
        epochs_per_negative_sample,
        epoch_of_next_negative_sample,
        epoch_of_next_sample,
        n
):
    """
    Linear UMAP算法的单次训练
    :param map_matrix: 映射矩阵 [D, d]
    :param data:       高维数据 [N, D]
    :param head:       图的行索引
    :param tail:       图的列索引
    :param n_vertices:
    :param epochs_per_sample:
    :param a:
    :param b:
    :param rng_state:
    :param gamma:
    :param dim:
    :param move_other:
    :param alpha:
    :param epochs_per_negative_sample:
    :param epoch_of_next_negative_sample:
    :param epoch_of_next_sample:
    :param n:
    :return:
    """
    data_dim = data.shape[1]
    for i in numba.prange(epochs_per_sample.shape[0]):
        if epoch_of_next_sample[i] <= n:
            j = head[i]
            k = tail[i]
            current = matrix_multiply(map_matrix.T, data[j].reshape((-1, 1)))
            other = matrix_multiply(map_matrix.T, data[k].reshape((-1, 1)))
            csubo = (current.T - other.T).flatten()
            dist_squared = rdist(current.flatten(), other.flatten())
            if dist_squared > 0.0:
                grad_coeff = -2.0 * a * b * pow(dist_squared, b - 1.0)
                grad_coeff /= a * pow(dist_squared, b) + 1.0
            else:
                grad_coeff = 0.0
            for d in range(dim):
                for h in range(data_dim):
                    grad_d = clip(grad_coeff * (data[j][h] - data[k][h]) * csubo[d])
                    map_matrix[h][d] += grad_d * alpha
            epoch_of_next_sample[i] += epochs_per_sample[i]
            n_neg_samples = int((n - epoch_of_next_negative_sample[i]) / epochs_per_negative_sample[i])
            for p in range(n_neg_samples):
                k = tau_rand_int(rng_state) % n_vertices
                other = matrix_multiply(map_matrix.T, data[k].reshape((-1, 1)))
                csubo = (current.T - other.T).flatten()
                dist_squared = rdist(current.flatten(), other.flatten())
                if dist_squared > 0.0:
                    grad_coeff = 2.0 * gamma * b
                    grad_coeff /= (0.001 + dist_squared) * (a * pow(dist_squared, b) + 1)
                elif j == k:
                    continue
                else:
                    grad_coeff = 0.0
                for d in range(dim):
                    for h in range(data_dim):
                        if grad_coeff > 0.0:
                            grad_d = clip(grad_coeff * (data[j][h] - data[k][h]) * csubo[d])
                        else:
                            grad_d = 0
                        map_matrix[h][d] += grad_d * alpha
            epoch_of_next_negative_sample[i] += (n_neg_samples * epochs_per_negative_sample[i])
################################################################################
def optimize_layout_linear_euclidean(
        map_matrix,
        data,
        head,
        tail,
        n_epochs,
        n_vertices,
        epochs_per_sample,
        a,
        b,
        rng_state,
        gamma=1.0,
        initial_alpha=1.0,
        negative_sample_rate=5.0,
        parallel=False,
        move_other=False
):
    """
    线性UMAP的优化函数
    :param map_matrix: 映射矩阵 [D, d]
    :param data:       高维数据 [N, D]
    :param head:       近邻图的行索引
    :param tail:       近邻图的列索引
    :param n_epochs:   迭代次数
    :param n_vertices:
    :param epochs_per_sample:
    :param a:
    :param b:
    :param rng_state:
    :param gamma:
    :param initial_alpha:
    :param negative_sample_rate:
    :param parallel:
    :param move_other:
    :return:
    """
    dim = map_matrix.shape[1]
    alpha = initial_alpha
    epochs_per_negative_sample = epochs_per_sample / negative_sample_rate
    epoch_of_next_negative_sample = epochs_per_negative_sample.copy()
    epoch_of_next_sample = epochs_per_sample.copy()
    optimize_fn = numba.njit(_optimize_layout_linear_euclidean_single_epoch, fastmath=True, parallel=parallel)
    for n in tqdm(range(n_epochs)):
        optimize_fn(
            map_matrix,
            data,
            head,
            tail,
            n_vertices,
            epochs_per_sample,
            a,
            b,
            rng_state,
            gamma,
            dim,
            move_other,
            alpha,
            epochs_per_negative_sample,
            epoch_of_next_negative_sample,
            epoch_of_next_sample,
            n
        )
        alpha = initial_alpha * (1.0 - (float(n) / float(n_epochs)))
    return map_matrix
################################################################################
def _optimize_layout_linear_2D_euclidean_single_epoch(
        map_matrix,
        data,
        head,
        tail,
        sample_height,
        sample_weight,
        n_vertices,
        epochs_per_sample,
        a,
        b,
        rng_state,
        gamma,
        dim,
        move_other,
        alpha,
        epochs_per_negative_sample,
        epoch_of_next_negative_sample,
        epoch_of_next_sample,
        n
):
    """
    2DUMAP算法的单次训练
    :param map_matrix: 映射矩阵
    :param data: 样本数据
    :param head:
    :param tail:
    :param sample_height:
    :param sample_weight:
    :param n_vertices:
    :param epochs_per_sample:
    :param a:
    :param b:
    :param rng_state:
    :param gamma:
    :param dim:
    :param move_other:
    :param alpha:
    :param epochs_per_negative_sample:
    :param epoch_of_next_negative_sample:
    :param epoch_of_next_sample:
    :param n:
    :return:
    """
    for i in numba.prange(epochs_per_sample.shape[0]):
        if epoch_of_next_sample[i] <= n:
            j = head[i]
            k = tail[i]
            current = matrix_multiply(data[j], map_matrix)
            other = matrix_multiply(data[k], map_matrix)
            dist_squared = rdist(current.flatten(), other.flatten())
            if dist_squared > 0.0:
                grad_coeff = -2.0 * a * b * pow(dist_squared, b - 1.0)
                grad_coeff /= a * pow(dist_squared, b) + 1.0
            else:
                grad_coeff = 0.0
            xsub = data[j].T-data[k].T
            csubo = current-other
            grad = matrix_multiply(xsub, csubo)
            for d in range(dim):
                for w in range(sample_weight):
                    grad_d = clip(grad_coeff * grad[w][d])
                    map_matrix[w][d] += grad_d * alpha
            epoch_of_next_sample[i] += epochs_per_sample[i]
            n_neg_samples = int((n - epoch_of_next_negative_sample[i]) / epochs_per_negative_sample[i])
            for p in range(n_neg_samples):
                k = tau_rand_int(rng_state) % n_vertices
                other = matrix_multiply(data[k], map_matrix)
                dist_squared = rdist(current.flatten(), other.flatten())
                if dist_squared > 0.0:
                    grad_coeff = 2.0 * gamma * b
                    grad_coeff /= (0.001 + dist_squared) * (a * pow(dist_squared, b) + 1)
                elif j == k:
                    continue
                else:
                    grad_coeff = 0.0
                xsub = data[j].T - data[k].T
                csubo = current - other
                grad =  matrix_multiply(xsub, csubo)
                for d in range(dim):
                    for w in range(sample_weight):
                        if grad_coeff > 0.0:
                            grad_d = clip(grad_coeff * grad[w][d])
                        else:
                            grad_d = 0
                        map_matrix[w][d] += grad_d * alpha
            epoch_of_next_negative_sample[i] += (n_neg_samples * epochs_per_negative_sample[i])
################################################################################
def optimize_layout_linear_2D_euclidean(
        map_matrix,
        data, head,
        tail,
        sample_height,
        sample_weight,
        n_epochs,
        n_vertices,
        epochs_per_sample,
        a,
        b,
        rng_state,
        gamma=1.0,
        initial_alpha=1.0,
        negative_sample_rate=5.0,
        parallel=False,
        move_other=False
):
    """
    2DUMAP算法的循环训练
    :param map_matrix:
    :param data:
    :param head:
    :param tail:
    :param sample_height:
    :param sample_weight:
    :param n_epochs:
    :param n_vertices:
    :param epochs_per_sample:
    :param a:
    :param b:
    :param rng_state:
    :param gamma:
    :param initial_alpha:
    :param negative_sample_rate:
    :param parallel:
    :param move_other:
    :return:
    """
    dim = map_matrix.shape[1]
    data = data.reshape(data.shape[0], sample_height, sample_weight)
    alpha = initial_alpha
    epochs_per_negative_sample = epochs_per_sample / negative_sample_rate
    epoch_of_next_negative_sample = epochs_per_negative_sample.copy()
    epoch_of_next_sample = epochs_per_sample.copy()
    optimize_fn = numba.njit(_optimize_layout_linear_2D_euclidean_single_epoch, fastmath=True, parallel=parallel)
    for n in tqdm(range(n_epochs)):
        optimize_fn(
            map_matrix,
            data,
            head,
            tail,
            sample_height,
            sample_weight,
            n_vertices,
            epochs_per_sample,
            a,
            b,
            rng_state,
            gamma,
            dim,
            move_other,
            alpha,
            epochs_per_negative_sample,
            epoch_of_next_negative_sample,
            epoch_of_next_sample,
            n
        )
        alpha = initial_alpha * (1.0 - (float(n) / float(n_epochs)))
    return map_matrix
