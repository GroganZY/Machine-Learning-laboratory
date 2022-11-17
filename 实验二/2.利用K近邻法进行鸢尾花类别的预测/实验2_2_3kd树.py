import pandas as pd
import numpy as np
from collections import Counter

x = np.array(pd.read_csv('iris.csv', usecols=(0, 1, 2, 3), delimiter=',', header=0))  # 读取特征集合
y = np.array(pd.read_csv('iris.csv')['species'])  # 读取标签集
 
simple = np.array([3.5,2.4,0.3,2.5])

class KDtreeNode:
    def __init__(self, val, label, dim, left=None, right=None):
        self.val = val  # 特征集
        self.dim = dim  # 维度
        self.label = label  # 标签
        self.left = left  # 左子树
        self.right = right  # 右子树
 
    def __str__(self):
        return f'特征是：{self.val}, 标签是：{self.label},划分维度:{self.dim}'

def CreateKDtree(x, y, dim):
    if x.size == 0:
        return None
    else:
        nidx = np.argsort(x, axis=0)[:, dim]  # 按照dim这个维度排序
        center_num = x.shape[0] // 2  # 中位数的序号
 
        cut_idx = nidx[center_num]  # 根节点的索引号
        left_idx = nidx[:center_num]  # 左子树的索引号
        right_idx = nidx[center_num + 1:]  # 右子树的索引号
 
        node_tree = KDtreeNode(x[cut_idx], y[cut_idx], dim)  # KD树的根节点
        dim = (dim + 1) % x.shape[1]  # 更新维度dim值
        node_tree.left = CreateKDtree(x[left_idx], y[left_idx], dim)  # 递归左子树
        node_tree.right = CreateKDtree(x[right_idx], y[right_idx], dim)  # 递归右子树
        return node_tree  # 得到KD树

def search_KDtree(simple, k):
    # 初始化距离,最近点为None,最近距离为无穷大
    nearest_knn = np.array([[None, float('inf')] for _ in range(k)])
    # 创建一个列表,用于存放从根节点到一个叶子结点的所有节点,找距离最近的点
    node_list = []
    # 得到KD树,node_tree是一颗KD树
    node_tree = CreateKDtree(x, y, 0)
    while node_tree:
        # 将所有可能的节点加入到列表中,加入的位置为列表的第一个元素
        node_list.insert(0, node_tree)
        dim = node_tree.dim
        if simple[dim] < node_tree.val[dim]:
            node_tree = node_tree.left
        else:
            node_tree = node_tree.right
    #从叶子结点开始,回溯
    for node in node_list:
        #计算欧几里得距离
        distance = np.linalg.norm(node.val - simple, ord=2)
        #np.where返回一个二维数组,及满足要求的位置坐标.less_index为距离小于inf的行的索引
        less_index = np.where(distance < nearest_knn[:,1])[0]
        #print(nearest_knn)
        if less_index.size > 0:
            #对nearest_knn进行更新
            nearest_knn = np.insert(nearest_knn, less_index[0], [node, distance], axis=0)[:k]  #只取前k个距离最短的
        radius = nearest_knn[:,1][k-1]                #radius为k个距离中最远的那个,欧几里得距离
        dis = simple[node.dim] - node.val[node.dim]   #所求点到超平面的距离
        if radius > abs(dis):                              #如果欧几里得距离大于到超平面的距离
            if dis > 0:                               #如果simple[node.dim] > node.val[node.dim],加入左子树
                append_node = node.left
            else:
                append_node = node.right              #否则,加入左右树
            if append_node is not None:
                node_list.append(append_node)
    return([lab[0].label for lab in nearest_knn if lab[0] is not None])


lb = search_KDtree(simple, 3)
print('预测结果为:'+Counter(lb).most_common(1)[0][0])