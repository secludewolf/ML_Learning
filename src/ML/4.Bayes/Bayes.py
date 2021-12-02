"""
X_i(结果,样本)
W_i(原因,类别)
先验概率: P(W_i)  (各类别在所有样本中的分布概率)
条件概率: P(X_i | W_i)  (已知类别为W_i的情况下,求样本是X_i的概率)
后验概率: P(W_i | X_i)  (已知样本是X_i的情况下,求类别是W_i的概率)

条件概率与后验概率看起来形式相同,但条件概率指的是一直某原因的情况某结果发生的概率,后验概率指的是已知某结果已经发生,求是在某原因下发生的概率
贝叶斯概率既是后验概率

重要假设:
1.特征之间相互独立,且每个特征同等重要.
2.先验概率已知.

实现方式:
1.伯努利(本次使用的方式)
2.多项式
3.搞死

算法特点:
    优点: 在数据较少的情况下仍然有效，可以处理多类别问题。
    缺点: 对于输入数据的准备方式较为敏感。
    适用数据类型: 标称型数据。

Demo:文本分类(识别侮辱性言论)
    提取所有文档中的词条并进行去重
    获取文档的所有类别
    计算每个类别中的文档数目
    对每篇训练文档:
        对每个类别:
            如果词条出现在文档中-->增加该词条的计数值（for循环或者矩阵相加）
            增加所有词条的计数值（此类别下词条总数）
    对每个类别:
        对每个词条:
            将该词条的数目除以总词条数目得到的条件概率（P(词条|类别)）
    返回该文档属于每个类别的条件概率（P(类别|文档的所有词条)）

开发流程:
    收集数据: 可以使用任何方法。
    准备数据: 需要数值型或者布尔型数据。
    分析数据: 有大量特征时，绘制特征作用不大，此时使用直方图效果更好。
    训练算法: 计算不同的独立特征的条件概率。
    测试算法: 计算错误率。
    使用算法: 一个常见的朴素贝叶斯应用是文档分类。可以在任意的分类场景中使用朴素贝叶斯分类器，不一定非要是文本。
"""


# 1.创建数据集
def load_data_set():
    """
    创建数据集
    Returns:
        posting_list: 文本列表
        class_vector: 文本分类
    """
    posting_list = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                    ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                    ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                    ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                    ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                    ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    class_vector = [0, 1, 0, 1, 0, 1]
    return posting_list, class_vector


# 2.1 构建词汇列表
def create_word_list(data_set):
    """
    获取出现过的单词的集合
    Args:
        data_set: 数据集
    Returns:
        word_list: 词汇列表
    """
    word_list = set()
    for posting in data_set:
        # set | set 合集
        # set & set 交集
        # set - set 差集
        word_list = word_list | set(posting)
    return list(word_list)


# 2,2 构建特征向量
def set_of_words_vector(word_list, input_set):
    """
    构建输入样本对应的特征向量
    Args:
        word_list: 词汇列表
        input_set: 输入样本
    Returns:
        word_sector: 特征向量
    """
    # list * number 列表扩增相应倍数
    # [1,2,3] * 3 = [1,2,3,1,2,3,1,2,3]
    word_vector = [0] * len(word_list)
    for word in input_set:
        if word in word_list:
            word_vector[word_list.index(word)] = 1
        else:
            print("词汇:'{}'在训练集中不存在".format(word))
    return word_vector


from numpy import *


# 3.训练模型(计算先验概率与条件概率)
def train(train_matrix, train_category):
    """
    根据样本集计算先验概率与条件概率
    由于每一类中P(W)相同,因此在比较是只是在比较分子的大小,故不必计算
    Args:
        train_matrix: 样本集
        train_category: 样本分类
    Returns:
        abusive_vector: 每个词汇在侮辱性文本中出现的概率  条件概率  [P(X_i | W_j)]
        normal_vector: 每个词汇在正常文本中出现的概率  条件概率  [P(X_i | W_j)]
        prob_abusive:  侮辱性文本在样本集中出现大概率  先验概率  P(W_i)
    """
    # 文本数量
    posting_number = len(train_matrix)
    # 词汇数量
    word_number = len(train_matrix[0])
    # 侮辱性文本出现的概率
    # 既先验概率P(W)
    prob_abusive = sum(train_category) / posting_number
    # 防止在后面求后验概率中出现多个概率相乘时,因为某个概率为零导致整体概率为零
    # 单词出现次数列表
    abusive_number = ones(word_number)
    normal_number = ones(word_number)
    # 与上面相同职位为了防止分母为零,其他非零值也可以
    # 文本长度
    abusive_list = 2.0
    normal_list = 2.0
    for i in range(posting_number):
        if train_category[i] == 1:
            abusive_number += train_matrix[i]
            abusive_list += sum(train_matrix[i])
        else:
            normal_number += train_matrix[i]
            normal_list += sum(train_matrix[i])
    # log化处理,防止因为小数连乘出现正下溢问题
    # 各个词汇在侮辱性文本中出现的概率
    abusive_vector = log(abusive_number / abusive_list)
    # 各个词汇在正常文本中出现的概率
    normal_vector = log(normal_number / normal_list)
    return abusive_vector, normal_vector, prob_abusive


# 4.分类
def classify(input_set, abusive_vector, normal_vector, prob_abusive):
    # 使用加法与sum是因为上面训练时已经进行了log化处理
    abusive = sum(input_set * abusive_vector) + log(prob_abusive)
    normal = sum(input_set * normal_vector) + log(1.0 - prob_abusive)
    if abusive > normal:
        return 1
    else:
        return 0


def demo1():
    posting_list, class_vector = load_data_set()
    word_list = create_word_list(posting_list)
    train_matrix = []
    for posting in posting_list:
        train_matrix.append(set_of_words_vector(word_list, posting))
    abusive_vector, normal_vector, prob_abusive = train(train_matrix, array(class_vector))
    test_posting1 = ['love', 'my', 'dalmation']
    test_posting2 = ['stupid', 'garbage']
    test_vector1 = set_of_words_vector(word_list, test_posting1)
    test_vector2 = set_of_words_vector(word_list, test_posting2)
    test_class1 = classify(test_vector1, abusive_vector, normal_vector, prob_abusive)
    test_class2 = classify(test_vector2, abusive_vector, normal_vector, prob_abusive)
    print(test_class1)
    print(test_class2)


if __name__ == '__main__':
    demo1()
