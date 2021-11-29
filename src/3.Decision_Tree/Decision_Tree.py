"""
决策树（Decision Tree）算法是一种基本的分类与回归方法，是最经常使用的数据挖掘算法之一。我们这章节只讨论用于分类的决策树。
决策树模型呈树形结构，在分类问题中，表示基于特征对实例进行分类的过程。它可以认为是 if-then 规则的集合，也可以认为是定义在特征空间与类空间上的条件概率分布。
决策树学习通常包括 3 个步骤: 特征选择、决策树的生成和决策树的修剪。

熵（entropy）: 熵指的是体系的混乱的程度，在不同的学科中也有引申出的更为具体的定义，是各领域十分重要的参量。
信息论（information theory）中的熵（香农熵）: 是一种信息的度量方式，表示信息的混乱程度，也就是说: 信息越有序，信息熵越低。例如: 火柴有序放在火柴盒里，熵值很低，相反，熵值很高。
信息增益（information gain）: 在划分数据集前后信息发生的变化称为信息增益。

开发流程:
    收集数据: 可以使用任何方法。
    准备数据: 树构造算法 (这里使用的是ID3算法，只适用于标称型数据，这就是为什么数值型数据必须离散化。 还有其他的树构造算法，比如CART)
    分析数据: 可以使用任何方法，构造树完成之后，我们应该检查图形是否符合预期。
    训练算法: 构造树的数据结构。
    测试算法: 使用训练好的树计算错误率。
    使用算法: 此步骤可以适用于任何监督学习任务，而使用决策树可以更好地理解数据的内在含义。

算法特点:
    优点: 计算复杂度不高，输出结果易于理解，数据有缺失也能跑，可以处理不相关特征。
    缺点: 容易过拟合。
    适用数据类型: 数值型和标称型。

demo1:
    判断鱼类和非鱼类
    数据集:
    不浮出水面是否可以生存#是否有脚蹼#是否是鱼
demo2:
    预测隐形眼镜类型
"""

# 1.构造数据及
from math import log


def create_date_set():
    """
    创建数据集
    Returns:
        date_set:数据集
        labels:特征名称
    """
    data_set = [[1, 1, 'yes'],
                [1, 1, 'yes'],
                [1, 0, 'no'],
                [0, 1, 'no'],
                [0, 1, 'no']]
    labels = ['不浮出水面是否可以生存', '是否有脚蹼']
    return data_set, labels


# 2.树构造方法
def calculate_shannon_entropy(data_set):
    """
    求当前数据集的香农熵
    Args:
        data_set: 数据集
    Returns:
        shannon_entropy: 数据集的香农熵
    """
    # 统计数据集大小
    row_numbers = len(data_set)
    # 记录各个类别出现的次数
    label_counts = {}
    for elem in data_set:
        # 数据集最后一行是当前样本所属类别
        current_label = elem[-1]
        # 统计各类别出现的次数
        if current_label not in label_counts.keys():
            label_counts[current_label] = 0
        label_counts[current_label] += 1
    # 数据集整体香农熵
    shannon_entropy = 0.0
    # 求出数据集的总体香农熵
    for key in label_counts:
        # 求出每个类别的香农熵
        prob = float(label_counts[key]) / row_numbers
        shannon_entropy -= prob * log(prob, 2)
    return shannon_entropy


# 3.划分数据集
def split_data_set(data_set, index, value):
    """
        筛选数据集中index特征为value指定特征值子数据集
    Args:
        data_set: 待划分数据集
        index: 目标特征所在列
        value: 要保留的特征值
    Returns:
        return_data_set: 划分后的数据集
    """
    # 返回数据及
    return_data_set = []
    # 判断样本index特征是否为value
    for elem in data_set:
        if elem[index] == value:
            # 保留样本中除index特征信息以外的所有信息
            # 筛选后的子数据集中index特征均相同,所以无保留必要
            reduced_feat_elem = elem[:index] + elem[index + 1:]
            return_data_set.append(reduced_feat_elem)
    return return_data_set


# 4.选择最优特征
def choose_best_feature_to_split(data_set):
    """
    选出最优划分所使用的特征
    Args:
        data_set: 数据集
    Returns:
        best_feature_index: 最优划分特征
    """
    # 特征数量(最后一列是样本分类)
    feature_numbers = len(data_set[0]) - 1
    # 原始数据集的信息熵
    base_entropy = calculate_shannon_entropy(data_set)
    # 最优划分的信息增益值
    bast_info_gain = 0
    # 最优时选择特征的位置
    best_feature_index = -1
    # 逐个计算采用各个特征时的信息增益值
    for i in range(feature_numbers):
        # 当前特征的所有特征值
        feature_list = [features[i] for features in data_set]
        # 当前特征有哪几种特征值
        unique_feature = set(feature_list)
        # 当前特征的信息熵
        feature_entropy = 0.0
        # 计算当前特征的信息熵
        for value in unique_feature:
            # 求出当前特征值的子数据集
            sub_data_set = split_data_set(data_set, i, value)
            # 求出当前特征值的分布概率
            prob = len(sub_data_set) / len(data_set)
            # 求当前特征的信息熵
            feature_entropy += prob * calculate_shannon_entropy(sub_data_set)
        # 求出当前特征的信息增益
        feature_info_gain = base_entropy - feature_entropy
        if feature_info_gain > bast_info_gain:
            bast_info_gain = feature_info_gain
            best_feature_index = i
    return best_feature_index


# 5.构造决策树
def create_tree(data_set, labels):
    """
    构造决策树
    Args:
        data_set: 数据集
        labels: 特征名
    Returns:
        tree: 决策树
    """
    # 数据集分类列表
    class_list = [elem[-1] for elem in data_set]
    # 如果数据集中只有一个类别,则已经分类完成
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    # 如果所有特征均已使用过,但是仍不能实现完全分类
    # 则按照当前比例最大的类别分类
    if len(data_set[0]) == 1:
        max_class_count = -1
        max_class = -1
        for t in set(class_list):
            if class_list.count(t) > max_class_count:
                max_class_count = class_list.count(t)
                max_class = t
        return max_class
    # 最优特征位置
    bast_feature = choose_best_feature_to_split(data_set)
    # 最优特征名称
    bast_feature_label = labels[bast_feature]
    # 决策树
    tree = {bast_feature_label: {}}
    # 删除已经加入到生成树的特征
    del labels[bast_feature]
    # 样本特征的所有特征值
    feature_values = [elem[bast_feature] for elem in data_set]
    # 特征的各种特征值
    unique_values = set(feature_values)
    # 为每个特征值创建子树
    for value in unique_values:
        # 子树中不包含已经分类过的特征
        sub_labels = labels[:]
        # 创建子树
        tree[bast_feature_label][value] = create_tree(split_data_set(data_set, bast_feature, value), sub_labels)
    return tree


# 6.使用决策树进行分类
def classify(tree, feature_labels, element):
    # 获取根节点划分所使用的特征
    root_feature = list(tree.keys())[0]
    # 获取子树
    second_tree = tree[root_feature]
    # 获取根节点特征所在位置
    root_feature_index = feature_labels.index(root_feature)
    # 获取样本根节点特征的特征值
    element_feature_value = element[root_feature_index]
    # 判断样本所进入的子树
    element_feature_tree = second_tree[element_feature_value]
    # 判断是否是叶子节点
    # 如果是叶子节点则分类完成,否则进入子树继续分类
    if isinstance(element_feature_tree, dict):
        class_label = classify(element_feature_tree, feature_labels, element)
    else:
        class_label = element_feature_tree
    return class_label


def store_tree(tree, file_name):
    """
    储存决策树
    Args:
        tree: 决策树
        file_name: 文件名
    """
    import pickle
    fw = open(file_name, 'wb')
    pickle.dump(tree, fw)
    fw.close()


def grab_tree(file_name):
    """
    读取决策树
    Args:
        file_name: 文件名
    Returns:
        file: 文件
    """
    import pickle
    fr = open(file_name, 'rb')
    return pickle.load(fr)


def demo1():
    (data_set, labels) = create_date_set()
    labels_clone = labels.copy()
    tree = create_tree(data_set, labels_clone)
    class_label = classify(tree, labels, (1, 1))
    print("决策树:" + str(tree))
    print("判断类型:" + class_label)


if __name__ == '__main__':
    demo1()
