"""
KNN 工作原理

1.假设有一个带有标签的样本数据集（训练样本集），其中包含每条数据与所属分类的对应关系。
2.输入没有标签的新数据后，将新数据的每个特征与样本集中数据对应的特征进行比较。
    1.计算新数据与样本数据集中每条数据的距离。
    2.对求得的所有距离进行排序（从小到大，越小表示越相似）。
    3.取前 k （k 一般小于等于 20 ）个样本数据对应的分类标签。
3.求 k 个数据中出现次数最多的分类标签作为新数据的分类。

KNN 通俗理解:给定一个训练数据集，对新的输入实例，在训练数据集中找到与该实例最邻近的 k 个实例，这 k 个实例的多数属于某个类，就把该输入实例分为这个类。

训练集:
    40920    8.326976    0.953952    3
    14488    7.153469    1.673904    2
    26052    1.441871    0.805124    1
    75136    13.147394    0.428964    1
    38344    1.669788    0.134296    1
格式:
    每年获得的飞行常客里程数
    玩视频游戏所耗时间百分比
    每周消费的冰淇淋公升数
    所属类别
分类:
    类别1-不喜欢的人
    类别2-魅力一般的人
    类别3-极具魅力的人
"""
from os import listdir

from numpy import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def file2matrix(file_path):
    """
    加载数据集
    Args:
        file_path: 数据集文件路径

    Returns:
        object:
        数据矩阵 returnMat 和对应的类别 class_label_vector
    """
    file = open(file_path)
    lines = file.readlines()
    # 获取文件中数据的行数
    number_of_lines = len(lines)
    # zeros()生成对应形状的内容为0的空数组(array)
    # 训练集样本属性矩阵
    # 例如: zeros(2，3)就是生成一个 2*3的矩阵，各个位置上全是 0
    data_matrix = zeros((number_of_lines, 3))
    # 训练集样本类别矩阵
    data_label = []
    index = 0
    for line in lines:
        # 去除头尾空字符串
        line = line.strip()
        # 以 '\t' 切割字符串
        list_from_line = line.split('\t')
        # 每个样本的属性数据
        data_matrix[index, :] = list_from_line[0:3]
        # 每个样本的分类
        data_label.append(int(list_from_line[-1]))
        index += 1
    return data_matrix, data_label


def img2vector(file_path):
    """
    将图片转为向量
    Args:
        file_path:文件地址
    Returns:
        img_vector:图片内容的向量形式
    """
    # 图片规格32*32,既向量维度为1024
    img_vector = zeros((1, 1024))
    img_file = open(file_path)
    # 逐像素读取照片内容
    for i in range(32):
        line = img_file.readline()
        for j in range(32):
            img_vector[0, 32 * i + j] = int(line[j])
    return img_vector


def auto_norm(data_matrix):
    """
    样本属性归一化,消除特征之间数量级不同导致的影响
    归一化公式:
        Y = (X-Xmin)/(Xmax-Xmin)
        其中的 min 和 max 分别是数据集中某种特征的最小特征值和最大特征值。
        该函数可以自动将数字特征值转化为0到1的区间。
    Args:
        data_matrix: 数据集

    Returns:
        norm_data_matrix: 归一化数据集

    """
    # array.min(axis=维度)
    # 无参数 数组中的最小值
    # axis = 0 每列最小
    # axis = 1 每行最小
    # 每列最小值
    mins = data_matrix.min(0)
    # 每列最大值
    maxs = data_matrix.max(0)
    # 最大差值
    ranges = maxs - mins
    # shape(array) 获取数组形状
    norm_data_matrix = zeros(shape(data_matrix))
    # shape[i] 获取数组某一维度的宽度
    # shape[0] = len(array)
    row_numbers = data_matrix.shape[0]  # 获取矩阵行数
    # tile(A, reps) 生成以A为内容,形状为reps的数组
    # tile(1,5) = array(1, 1, 1, 1, 1)
    # tile(mines, (m, 1)) 生成一个与原矩阵相同大小内容是各列最小值的矩阵
    # 生成与最小值之差组成的矩阵
    norm_data_matrix = data_matrix - tile(mins, (row_numbers, 1))
    # 将最小值之差除以范围组成矩阵
    norm_data_matrix = norm_data_matrix / tile(ranges, (row_numbers, 1))
    return norm_data_matrix


def classify(target, data_set, labels, k):
    """
    KNN分类器
    根据与目标样本最近的前K个已知样本的类别判断目标样本所属的类别
    Args:
        target:需要分类的目标信息
        data_set:已知样本的数据集
        labels:已知样本的分类
        k:取值K的大小
    Returns:
        label:分类结果
    """
    # 获取数据集数据个数
    data_set_size = data_set.shape[0]
    # 目标样本与已知样本距离矩阵
    # 度量公式为欧氏距离
    # 各个特征的差值
    diff_matrix = tile(target, (data_set_size, 1)) - data_set
    square_diff_matrix = diff_matrix ** 2
    square_distances = square_diff_matrix.sum(axis=1)
    distances = square_distances ** 0.5
    # 将已知样本按距离排序
    sorted_list_indices = distances.argsort()
    class_count = {}
    # 统计前K个样本的类别
    for i in range(k):
        vote_label = labels[sorted_list_indices[i]]
        class_count[vote_label] = class_count.get(vote_label, 0) + 1
    # sorted(迭代器,key,reverse) 迭代器=可迭代对象, key=一个可以返回课用于比较的元素的函数, reverse=是否反转排序结果(默认升序)
    sorted_class_count = sorted(class_count.items(), key=lambda x: x[1], reverse=True)
    label = sorted_class_count[0][0]
    return label


def dating_class_test(data_matrix, data_label):
    """
    约会项目测试分类器结果
    Args:
        data_matrix:已知样本的数据集
        data_label:已知样本的分类
    Returns:
        error_percent:错误率
    """
    # 设置测试数据集的比例(训练数据集比例=1-test_data_ratio)
    test_data_ratio = 0.1
    # 归一化数据
    norm_data_matrix = auto_norm(data_matrix)
    # 数据集总行数,既矩阵第一维
    row_numbers = norm_data_matrix.shape[0]
    print("数据集大小:" + str(row_numbers))
    # 测试集大小
    test_data_number = int(row_numbers * test_data_ratio)
    print("样本集大小:" + str(row_numbers - test_data_number))
    print("测试集大小:" + str(test_data_number))
    # 错误样本个数
    error_count = 0
    # 对样本进行测试
    for i in range(test_data_number):
        classify_label = classify(norm_data_matrix[i], norm_data_matrix[test_data_number:row_numbers],
                                  data_label[test_data_number:row_numbers], 3)
        if classify_label != data_label[i]:
            error_count += 1
    error_percent = error_count / test_data_number
    return error_percent


def handwriting_class_test(train_folder_path, test_folder_path):
    labels = []
    # listdir(path) path=文件夹路径 return=[string] 返回包含文件夹下所有文件的文件名的列表
    # 获取所有训练文件名
    training_file_list = listdir(train_folder_path)
    # 获取训练文件数量
    training_file_numbers = len(training_file_list)
    # 获取所有测试文件名
    test_file_list = listdir(test_folder_path)
    # 获取测试文件数量
    test_file_numbers = len(test_file_list)
    # 生成测试训练矩阵
    training_matrix = zeros((training_file_numbers, 1024))
    print("样本集大小:" + str(training_file_numbers))
    print("测试集大小:" + str(test_file_numbers))
    for i in range(training_file_numbers):
        # 文件全名(包括扩展名)
        file_full_name = training_file_list[i]
        # 去除扩展名
        file_name = file_full_name.split(".")[0]
        # 获取样本分类
        label_number = int(file_name.split("_")[0])
        # 记录样本分类
        labels.append(label_number)
        # 将图片转为向量
        training_matrix[i, :] = img2vector(train_folder_path + file_full_name)
    # 错误个数
    error_count = 0.0
    # 逐个测试
    for i in range(test_file_numbers):
        file_full_name = test_file_list[i]
        file_name = file_full_name.split(".")[0]
        label_number = int(file_name.split("_")[0])
        test_vector = img2vector(test_folder_path + file_full_name)
        # 进行分类
        classify_label = classify(test_vector, training_matrix, labels, 3)
        if classify_label != label_number:
            error_count += 1.0
    error_percent = error_count / test_file_numbers
    return error_percent


def demo1():
    filepath = "../../data/2.KNN/datingTestSet2.txt"
    # 加载数据集
    (data_matrix, data_label) = file2matrix(filepath)
    error_percent = dating_class_test(data_matrix, data_label)
    print("错误率:" + str(error_percent) + "%")
    figure = plt.figure()
    # 绘制散点图scatter(x=x坐标,y=y坐标,s=点大小,c=点颜色,marker=点形状,cmap=颜色渐变,norm=亮度,(vmin，vmax)=(最小亮度,最大亮度),alpha=透明度,linewidths=线宽度)
    # (train_matrix[:, 0], train_matrix[:, 1])=(x,y) 点大小为5 array(class_label_vector)点分类颜色
    x = data_matrix[:, 0]
    y = data_matrix[:, 1]
    # 散点图
    ax = figure.add_subplot()
    ax.scatter(x, y, 5, array(data_label))
    plt.show()


def demo2():
    train_folder_path = "../../data/2.KNN/trainingDigits/"
    test_folder_path = "../../data/2.KNN/testDigits/"
    error_percent = handwriting_class_test(train_folder_path, test_folder_path)
    print("错误率:" + str(error_percent) + "%")


if __name__ == '__main__':
    demo1()
    demo2()
