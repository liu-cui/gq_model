from gurobipy import *
import gurobipy as gp
from gurobipy import GRB
import numpy as np

# ======================
#   数据准备（示例数据）
# ======================
n = 5  # 客户数量(节点0为车场)
Q = 100  # 车辆容量
M = 1e6  # 大常数

# 节点坐标 (示例数据)
coordinates = [
    (0, 0),   # 车场
    (2, 3), (5, 8), (6, 1), (8, 4), (3, 7)]


def print_hi(name):
    print(f"Hi, {name}")


def calc_distance_matrix(coords):
    """ 计算距离矩阵
    :param coords:
    :return:
    """
    n_nodes = len(coords)
    dist = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(n_nodes):
            dist[i][j] = np.hypot(coords[i][0]-coords[j][0], coords[i][1]-coords[j][1])
    return dist


model = gp.Model('gq_model')
V = range(len(coordinates))  # 节点索引(0是维护站点，1～n是病害点)
K = range(4)   # 车辆集合数
x = model.addVars(V, V, K, vtype=GRB.BINARY, name='x')  # 路径变量

if __name__ == "__main__":
    print_hi("PyCharm")
    dist_matrix = calc_distance_matrix(coordinates)
    print(f"距离矩阵={dist_matrix}")

