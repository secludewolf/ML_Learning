"""
支持向量机

超平面函数: y(x) = w_T * x + b
分类结果: f(x) = sign(w_T * x + b)    sign(x) = 1  x>0 | sign(x) = -1  x<0
点到超平面的距离: d(x) = (w_T + b) / ||w||(向量的模)

目标: 找到合适的w和b使得超平面到点的距离最大

w和b只受距离平面最近的样本点影响,既支持向量

目标函数: min ||w|| <==> min (0.5 * (||w||)^2)
约束条件: f(x) * (w_T * x + b) = 1

开发流程:
    收集数据: 可以使用任意方法。
    准备数据: 需要数值型数据。
    分析数据: 有助于可视化分隔超平面。
    训练算法: SVM的大部分时间都源自训练，该过程主要实现两个参数的调优。
    测试算法: 十分简单的计算过程就可以实现。
    使用算法: 几乎所有分类问题都可以使用SVM，值得一提的是，SVM本身是一个二类分类器，对多类问题应用SVM需要对代码做一些修改。

算法特点:
    优点: 泛化（由具体的、个别的扩大为一般的，就是说: 模型训练完后的新样本）错误率低，计算开销不大，结果易理解。
    缺点: 对参数调节和核函数的选择敏感，原始分类器不加修改仅适合于处理二分类问题。
    使用数据类型: 数值型和标称型数据。

"""
from matplotlib import pyplot as plt
from numpy import *


def loadDataSet(file_path):
    """
    对文件进行逐行解析，从而得到第行的类标签和整个特征矩阵
    Args:
        file_path: 文件名
    Returns:
        data_matrix: 特征矩阵
        label_matrix: 类标签
    """
    data_matrix = []
    label_matrix = []
    fr = open(file_path)
    for line in fr.readlines():
        line_array = line.strip().split('\t')
        data_matrix.append([float(line_array[0]), float(line_array[1])])
        label_matrix.append(float(line_array[2]))
    return data_matrix, label_matrix


def select_rand(i, m):
    """
    随机选择一个整数
    Args:
        i  第一个alpha的下标
        m  所有alpha的数目
    Returns:
        j  返回一个不为i的随机数，在0~m之间的整数值
    """
    j = i
    while j == i:
        j = int(random.uniform(0, m))
    return j


def clip_alpha(aj, H, L):
    """clipAlpha(调整aj的值，使aj处于 L<=aj<=H)
    Args:
        aj  目标值
        H   最大值
        L   最小值
    Returns:
        aj  目标值
    """
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


def smo_simple(data_set, class_labels, c, toler, max_cycles):
    """
    Args:
        data_set: 特征集合
        class_labels: 类别标签
        c: 松弛变量(常量值)，允许有些数据点可以处于分隔面的错误一侧。
            控制最大化间隔和保证大部分的函数间隔小于1.0这两个目标的权重。
            可以通过调节该参数达到不同的结果。
        toler: 容错率（是指在某个体系中能减小一些因素或选择对某个系统产生不稳定的概率。）
        max_cycles: 退出前最大的循环次数
    Returns:
        b: 模型的常量值
        alphas: 拉格朗日乘子
    """
    data_matrix = mat(data_set)
    # 矩阵转置 和 .T 一样的功能
    label_matrix = mat(class_labels).transpose()
    m, n = shape(data_matrix)

    # 初始化 b和alphas(alpha有点类似权重值。)
    b = 0
    alphas = mat(zeros((m, 1)))

    # 没有任何alpha改变的情况下遍历数据的次数
    iter = 0
    while iter < max_cycles:
        # 记录alpha是否已经进行优化，每次循环时设为0，然后再对整个集合顺序遍历
        alpha_pairs_changed = 0
        for i in range(m):
            # 我们预测的类别 y[i] = w^Tx[i]+b; 其中因为 w = Σ(1~n) a[n]*label[n]*x[n]
            fXi = float(multiply(alphas, label_matrix).T * (data_matrix * data_matrix[i, :].T)) + b
            # 预测结果与真实结果比对，计算误差Ei
            Ei = fXi - float(label_matrix[i])

            # 约束条件 (KKT条件是解决最优化问题的时用到的一种方法。我们这里提到的最优化问题通常是指对于给定的某一函数，求其在指定作用域上的全局最小值)
            # 0<=alphas[i]<=C，但由于0和C是边界值，我们无法进行优化，因为需要增加一个alphas和降低一个alphas。
            # 表示发生错误的概率: labelMat[i]*Ei 如果超出了 toler， 才需要优化。至于正负号，我们考虑绝对值就对了。
            '''
            # 检验训练样本(xi, yi)是否满足KKT条件
            yi*f(i) >= 1 and alpha = 0 (outside the boundary)
            yi*f(i) == 1 and 0<alpha< C (on the boundary)
            yi*f(i) <= 1 and alpha = C (between the boundary)
            '''
            if ((label_matrix[i] * Ei < -toler) and (alphas[i] < c)) or (
                    (label_matrix[i] * Ei > toler) and (alphas[i] > 0)):

                # 如果满足优化的条件，我们就随机选取非i的一个点，进行优化比较
                j = select_rand(i, m)
                # 预测j的结果
                fXj = float(multiply(alphas, label_matrix).T * (data_matrix * data_matrix[j, :].T)) + b
                Ej = fXj - float(label_matrix[j])
                alpha_i_old = alphas[i].copy()
                alpha_j_old = alphas[j].copy()

                # L和H用于将alphas[j]调整到0-C之间。如果L==H，就不做任何改变，直接执行continue语句
                # labelMat[i] != labelMat[j] 表示异侧，就相减，否则是同侧，就相加。
                if label_matrix[i] != label_matrix[j]:
                    L = max(0, alphas[j] - alphas[i])
                    H = min(c, c + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - c)
                    H = min(c, alphas[j] + alphas[i])
                # 如果相同，就没法优化了
                if L == H:
                    print("L==H")
                    continue

                # eta是alphas[j]的最优修改量，如果eta==0，需要退出for循环的当前迭代过程
                # 参考《统计学习方法》李航-P125~P128<序列最小最优化算法>
                eta = 2.0 * data_matrix[i, :] * data_matrix[j, :].T - data_matrix[i, :] * data_matrix[i,
                                                                                          :].T - data_matrix[
                                                                                                 j,
                                                                                                 :] * data_matrix[
                                                                                                      j, :].T
                if eta >= 0:
                    print("eta>=0")
                    continue

                # 计算出一个新的alphas[j]值
                alphas[j] -= label_matrix[j] * (Ei - Ej) / eta
                # 并使用辅助函数，以及L和H对其进行调整
                alphas[j] = clip_alpha(alphas[j], H, L)
                # 检查alpha[j]是否只是轻微的改变，如果是的话，就退出for循环。
                if abs(alphas[j] - alpha_j_old) < 0.00001:
                    print("j not moving enough")
                    continue
                # 然后alphas[i]和alphas[j]同样进行改变，虽然改变的大小一样，但是改变的方向正好相反
                alphas[i] += label_matrix[j] * label_matrix[i] * (alpha_j_old - alphas[j])
                # 在对alpha[i], alpha[j] 进行优化之后，给这两个alpha值设置一个常数b。
                # w= Σ[1~n] ai*yi*xi => b = yj- Σ[1~n] ai*yi(xi*xj)
                # 所以:   b1 - b = (y1-y) - Σ[1~n] yi*(a1-a)*(xi*x1)
                # 为什么减2遍？ 因为是 减去Σ[1~n]，正好2个变量i和j，所以减2遍
                b1 = b - Ei - label_matrix[i] * (alphas[i] - alpha_i_old) * data_matrix[i, :] * data_matrix[i, :].T - \
                     label_matrix[
                         j] * (alphas[j] - alpha_j_old) * data_matrix[i, :] * data_matrix[j, :].T
                b2 = b - Ej - label_matrix[i] * (alphas[i] - alpha_i_old) * data_matrix[i, :] * data_matrix[j, :].T - \
                     label_matrix[
                         j] * (alphas[j] - alpha_j_old) * data_matrix[j, :] * data_matrix[j, :].T
                if (0 < alphas[i]) and (c > alphas[i]):
                    b = b1
                elif (0 < alphas[j]) and (c > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                alpha_pairs_changed += 1
                print("iter: %d i:%d, pairs changed %d" % (iter, i, alpha_pairs_changed))
        # 在for循环外，检查alpha值是否做了更新，如果更新则将iter设为0后继续运行程序
        # 直到更新完毕后，iter次循环无变化，才退出循环。
        if alpha_pairs_changed == 0:
            iter += 1
        else:
            iter = 0
        print("iteration number: %d" % iter)
    return b, alphas


def calcWs(alphas, data_arr, class_labels):
    """
    基于alpha计算w值
    Args:
        alphas:        拉格朗日乘子
        data_arr:       feature数据集
        class_labels:   目标变量数据集
    Returns:
        wc:  回归系数
    """
    X = mat(data_arr)
    labelMat = mat(class_labels).transpose()
    m, n = shape(X)
    w = zeros((n, 1))
    for i in range(m):
        w += multiply(alphas[i] * labelMat[i], X[i, :].T)
    return w


def plot_fig_SVM(x_mat, y_mat, ws, b, alphas):
    """
    参考地址:
       http://blog.csdn.net/maoersong/article/details/24315633
       http://www.cnblogs.com/JustForCS/p/5283489.html
       http://blog.csdn.net/kkxgx/article/details/6951959
    """

    x_mat = mat(x_mat)
    y_mat = mat(y_mat)

    # b原来是矩阵，先转为数组类型后其数组大小为（1,1），所以后面加[0]，变为(1,)
    b = array(b)[0]
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # 注意flatten的用法
    ax.scatter(x_mat[:, 0].flatten().A[0], x_mat[:, 1].flatten().A[0])

    # x最大值，最小值根据原数据集dataArr[:, 0]的大小而定
    x = arange(-1.0, 10.0, 0.1)

    # 根据x.w + b = 0 得到，其式子展开为w0.x1 + w1.x2 + b = 0, x2就是y值
    y = (-b - ws[0, 0] * x) / ws[1, 0]
    ax.plot(x, y)

    for i in range(shape(y_mat[0, :])[1]):
        if y_mat[0, i] > 0:
            ax.plot(x_mat[i, 0], x_mat[i, 1], 'cx')
        else:
            ax.plot(x_mat[i, 0], x_mat[i, 1], 'kp')

    # 找到支持向量，并在图中标红
    for i in range(100):
        if alphas[i] > 0.0:
            ax.plot(x_mat[i, 0], x_mat[i, 1], 'ro')
    plt.show()


def demo():
    # 获取特征和目标变量
    dataArr, labelArr = loadDataSet('../../data/6.SVM/testSet.txt')
    # print labelArr

    # b是常量值， alphas是拉格朗日乘子
    b, alphas = smo_simple(dataArr, labelArr, 0.6, 0.001, 40)
    print('/n/n/n')
    print('b=', b)
    print('alphas[alphas>0]=', alphas[alphas > 0])
    print('shape(alphas[alphas > 0])=', shape(alphas[alphas > 0]))
    for i in range(100):
        if alphas[i] > 0:
            print(dataArr[i], labelArr[i])
    # 画图
    ws = calcWs(alphas, dataArr, labelArr)
    plot_fig_SVM(dataArr, labelArr, ws, b, alphas)


if __name__ == "__main__":
    demo()
