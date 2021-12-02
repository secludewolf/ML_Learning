"""
逻辑回归

开发流程
    收集数据: 可以使用任何方法
    准备数据: 由于需要进行距离计算，因此要求数据类型为数值型。另外，结构化数据格式则最佳
    分析数据: 画出决策边界
    训练算法: 使用梯度上升找到最佳参数
    测试算法: 使用 Logistic 回归进行分类
    使用算法: 对简单数据集中数据进行分类
"""
import matplotlib.pyplot as plt
from numpy import *


# 1.加载数据集
def load_data_set_demo1(file_path):
    """
    加载并解析数据集
    Args:
        file_path: 文件路径
    Returns:
        data_matrix: 样本特征集
        label_matrix: 样本分类
    """
    data_matrix = []
    label_matrix = []
    fr = open(file_path)
    for line in fr.readlines():
        line_split = line.strip().split()
        # 为了方便计算，我们将 X0 的值设为 1.0 ，也就是在每一行的开头添加一个 1.0 作为 X0
        data_matrix.append([1.0, float(line_split[0]), float(line_split[1])])
        label_matrix.append(int(line_split[2]))
    return data_matrix, label_matrix


def sigmoid(x):
    # y = Sigmoid(x) = 1 / [1 + e^(-x)]
    # x ∈ R | y ∈ (0, 1)
    return 1.0 / (1 + exp(-2 * x))


# 2.优化回归系数
def gradient_ascent(data_set, class_labels):
    """
    使用梯度上升的方式优化回归系数
    通过调整不同特征的回归系数w,使得sigmoid函数的结果接近label_matrix
    Args:
        data_set: 数据集
        class_labels: 样本分类
    Returns:
        weights: 回归系数
    """
    # 将原始数据集转换为矩阵
    data_matrix = mat(data_set)
    # 将样本分类转换为矩阵后转置
    label_matrix = mat(class_labels).transpose()
    # m 矩阵行数 样本数量
    # n 矩阵列数 样本特征数量
    m, n = shape(data_matrix)
    # 训练步长
    alpha = 0.001
    # 迭代次数
    max_cycles = 500
    # 创建一个与样本特征数量相同的参数数组
    weights = ones((n, 1))
    # 开始梯度优化回归系数
    for k in range(max_cycles):
        # 计算预测结果
        # sigmoid([...,[x_1 * w_1 + x_2 * w_2 + ... + x_n * w_n],...])
        y = sigmoid(data_matrix * weights)
        # 计算误差
        error = (label_matrix - y)
        # 根据误差修正回归系数
        # 决策函数: f(w) = w_T * x
        # 迭代公式: w = w + alpha * ▽w_f(w)   alpha=训练步长   ▽w_f(w)=梯度算子
        # 矩阵求导法则: d_(A_T * B)/d_A = B
        # ▽w_f(w) = d_f(w)/d_w = d_(w_T * x)/d_w = x = 误差 * 样本
        weights = weights + alpha * data_matrix.transpose() * error
    return array(weights)


# 3.分类
def plot_best_fit(data_set, label_matrix, weights):
    element_number = shape(data_set)[0]
    x_cord1 = []
    y_cord1 = []
    x_cord2 = []
    y_cord2 = []
    for i in range(element_number):
        if (int(label_matrix[i])) == 1:
            x_cord1.append(data_set[i, 1])
            y_cord1.append(data_set[i, 2])
        else:
            x_cord2.append(data_set[i, 1])
            y_cord2.append(data_set[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.scatter(x_cord1, y_cord1, s=30, c="red", marker="s")
    ax.scatter(x_cord2, y_cord2, s=30, c="green")
    x = arange(-3, 3, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()


def demo1():
    data_matrix, label_matrix = load_data_set_demo1("../../../data/5.Logistic/TestSet.txt")
    data_array = array(data_matrix)
    weights = gradient_ascent(data_array, label_matrix)
    plot_best_fit(data_array, label_matrix, weights)


if __name__ == '__main__':
    demo1()
