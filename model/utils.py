from __future__ import print_function

import numpy as np


def zero_pad(X, seq_len):
    return np.array([x[:seq_len - 1] + [0] * max(seq_len - len(x), 1) for x in X])


def get_vocabulary_size(X):
    return max([max(x) for x in X]) + 1  # plus the 0th word


def fit_in_vocabulary(X, voc_size):
    return [[w for w in x if w < voc_size] for x in X]


def batch_generator(X, y, batch_size):
    """Primitive batch generator
    """
    size = X.shape[0]  # 获取整个数据集的长度
    X_copy = X.copy()
    y_copy = y.copy()
    # indices = np.arange(size)  # 产生0到size-1的数字
    # np.random.shuffle(indices)  # 打乱顺序
    # X_copy = X_copy[indices]
    # y_copy = y_copy[indices]
    i = 0
    while True:   # 总共循环多少次在调用该函数时指定
        if i + batch_size <= size:  # 还在数据集的范围中时，从中选取batch_size条数据
            yield X_copy[i:i + batch_size], y_copy[i:i + batch_size]
            i += batch_size
        else:  # 超出数据集的范围时，重新打乱数据集，再继续选取
            i = 0
            # indices = np.arange(size)
            # np.random.shuffle(indices)
            # X_copy = X_copy[indices]
            # y_copy = y_copy[indices]
            continue


if __name__ == "__main__":
    # Test batch generator
    gen = batch_generator(np.array(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']),
                          np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]), 2)
    for _ in range(10):
        xx, yy = next(gen)
        print(xx, yy)
