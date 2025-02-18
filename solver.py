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
    (0, 0),  # 车场
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
            dist[i][j] = np.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1])
    return dist


class GModel:
    def __init__(self):
        self.model = gp.Model('gq_model')
        # 集合
        self.V = range(len(coordinates))  # 节点索引(0是维护站点，1～n是病害点)
        self.K = range(4)  # 车辆集合数
        # 参数
        self.bigM = 1e6
        self.d = calc_distance_matrix(coordinates)
        self.time_window = [(0, 100) for _ in range(len(self.V))]  # 所有点都一样的服务时间，23:00-4:00
        # 决策变量
        self.x = self.model.addVars(self.V, self.V, self.K, vtype=GRB.BINARY, name='x')  # 车辆k是否从i行驶到j
        self.u = self.model.addVars(self.V, lb=0, vtype=GRB.CONTINUOUS, name='u')  # 到达时间

    def set_objective(self):
        self.model.setObjective(
            gp.quicksum(self.d[i][j] * self.x[i, j, k] for i in self.V for j in self.V for k in self.K),
            GRB.MAXIMIZE
        )

    def constraint_visit(self):
        """病害点至多被访问一次
        :return:
        """
        self.model.addConstrs(
            gp.quicksum(self.x[j, i, k] for j in self.V for k in self.K if j != i) <= 1 for i in self.V[1:])

    def constraint_flow_balance(self):
        """流平衡约束
        :return:
        """
        for k in self.K:
            for i in self.V:
                self.model.addConstr(
                    gp.quicksum(self.x[i, j, k] for j in self.V if j != i) ==
                    gp.quicksum(self.x[j, i, k] for j in self.V if j != i),
                    name=f"flow_balance_{i}_{k}")

    def constraint_deport_depart(self):
        """车进车出
        :return:
        """
        for k in self.K:
            self.model.addConstr(
                gp.quicksum(self.x[0, j, k] for j in self.V[1:]) <= 1,
                name=f'deport_depart_{k}'
            )
            self.model.addConstr(
                gp.quicksum(self.x[i, 0, k] for i in self.V[1:]) <= 1,
                name=f'deport_depart_{k}'
            )

    def set_constraints(self):
        self.constraint_visit()
        self.constraint_flow_balance()
        self.constraint_deport_depart()

    def solve(self):
        self.set_constraints()
        self.set_objective()
        self.model.Params.TimeLimit = 300  # 设置5分钟求解限制
        self.model.optimize()
        if self.model.status == GRB.OPTIMAL:
            print(f"最优总距离：{self.model.objVal:.2f}")


if __name__ == "__main__":
    print_hi("PyCharm")
    d = calc_distance_matrix(coordinates)
    print(f"距离矩阵={d}")

    gq_model = GModel()
    gq_model.solve()
