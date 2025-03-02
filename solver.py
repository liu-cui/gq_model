"""
模型约束已经文档序号一一对应
"""

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import matplotlib.pyplot as plt
import numpy as np


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
            dist[i][j] = round(np.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1]), 4)
    return dist


class Config:
    """
    模型参数设置
    """

    def __init__(self):
        self.vehicle_number = 4  # 车辆数
        self.people_number_per_vehicle = 4  # 每车人数
        self.time_window_a = 0  # 所有点都一样的服务时间，23:00-4:00
        self.time_window_b = 100
        self.bigM = 1e6
        self.coordinates = [(0, 0), (2, 3), (5, 8), (6, 1), (8, 4), (3, 7)]  # 损伤点, (0,0)车场
        self.CAP = 50  # 养护车辆的最大载客量


class GModel:
    def __init__(self, cfg):
        self.model = gp.Model('gq_model')
        # 集合
        self.V = range(len(cfg.coordinates))  # 节点索引(0是维护站点，1～n是病害点)
        self.K = range(cfg.vehicle_number)  # 车辆集合数
        self.P = range(cfg.people_number_per_vehicle)  # 每车人人数4人
        # 参数
        self.M = cfg.bigM
        self.CAP = cfg.CAP
        self.d_matrix = calc_distance_matrix(cfg.coordinates)  # 距离矩阵
        self.t_matrix = calc_distance_matrix(cfg.coordinates)  # 时间矩阵
        self.time_window = [(cfg.time_window_a, cfg.time_window_b) for _ in range(len(self.V))]
        # 决策变量
        self.x = self.model.addVars(self.V, self.V, self.K, vtype=GRB.BINARY, name='x')  # x(i,j,k)
        self.y = self.model.addVars(self.V, self.K, vtype=GRB.BINARY, name='y')  # y(i,k)
        self.u = self.model.addVars(self.V, lb=0, vtype=GRB.CONTINUOUS, name='u')  # u(i)
        self.t = self.model.addVars(self.V, self.K, vtype=GRB.CONTINUOUS, name='t')  # t(i,k)
        # self.p = self.model.addVars(self.K, lb=0, vtype=GRB.INTEGER, name="p")  # p(k)
        self.p = [cfg.people_number_per_vehicle] * cfg.vehicle_number  # 约束固定

        # 结果收集
        self.routes = list()

    def set_objective(self):
        self.model.setObjective(
            gp.quicksum(self.d_matrix[i][j] * self.x[i, j, k] for i in self.V for j in self.V for k in self.K),
            GRB.MAXIMIZE
        )

    def constraint_visit(self):
        """病害点至多被访问一次
        :return:
        """
        self.model.addConstrs(
            (gp.quicksum(self.x[j, i, k] for j in self.V for k in self.K if j != i) <= 1 for i in self.V[1:]),
            name="at_least_one_visit")

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

    def constraint_time_continuity(self):
        """时间连续性
        :return:
        """
        # 时间连续性
        self.model.addConstrs(
            (self.u[j] >= self.u[i] + self.t[i, k] + self.t_matrix[i, j] - self.M * (1 - self.x[i, j, k])
             for k in self.K for i in self.V for j in self.V[1:] if i != j), name="time_continuity")

    def constraint_x_leq_y(self):
        """ x_ijk <= y_ik
        :return:
        """
        self.model.addConstrs(
            self.x[i, j, k] <= self.y[i, k]
            for i in self.V for j in self.V if i != j for k in self.K
        )

    def constraint_time_window(self):
        self.model.addConstrs(self.u[i] >= self.time_window[i][0] for i in self.V)
        self.model.addConstrs(self.u[i] <= self.time_window[i][1] for i in self.V)

    def set_constraints(self):
        self.constraint_time_window()  # (5.17)
        self.constraint_visit()  # (5.18)
        self.constraint_flow_balance()  # (5.19)
        self.constraint_deport_depart()  # (5.20-5.21)
        self.constraint_x_leq_y()  # (5.22)
        self.constraint_time_continuity()  # (5.23)

    def set_model_parameters(self):
        self.model.Params.TimeLimit = 300  # 设置5分钟求解限制
        self.model.Params.OutputFlag = 0
        self.model.Params.IntegralityFocus = 1  # 防止0-1变量有小数位
        self.model.Params.LicenseID = 2613921

    def model_solver_print(self):
        if self.model.status == GRB.OPTIMAL:
            print(f"最优总距离：{self.model.objVal:.2f}")
            for key in self.x.keys():
                if self.x[key].x > 0:
                    print(f"{self.x[key].VarName}= {self.x[key].x}")

    def model_result_analysis(self):
        if self.model.status == GRB.OPTIMAL:
            print(f"最优总距离：{self.model.objVal:.2f}")
            # 提取车辆路径
            for k in self.K:
                active_routes = [(i, j) for i in self.V for j in self.V if self.x[i, j, k].X > 0.5]
                if not active_routes:
                    print(f"车辆{k + 1}: 未使用")
                    continue
                route = [0]
                current = [j for (i, j) in active_routes if i == 0][0]
                while current != 0:
                    route.append(current)
                    current = [j for (i, j) in active_routes if i == current][0]
                self.routes.append(route)
                print(f"\n车辆:{k + 1}路径:{'->'.join(map(str, route))}")
                print(f"到达时间: {[self.u[i].X for i in route]}")
                # print(f"k={k}, {self.routes[k]}")
        else:
            print("未找到可行解")

    def solve(self):
        self.set_constraints()
        self.set_objective()
        self.model.write('gq_model.lp')
        self.set_model_parameters()
        self.model.optimize()
        self.model_solver_print()
        self.model_result_analysis()


if __name__ == "__main__":
    print_hi("PyCharm")
    gq_model = GModel(cfg=Config())
