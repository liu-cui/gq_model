"""
模型约束已经文档序号一一对应
"""

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd
import random


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


class GModel():
    def __init__(self):
        self.model = gp.Model('gq_model')
        self.vehicle_number = 2  # 车辆数
        self.people_number_per_vehicle = 2  # 每车人数
        self.time_window_a = 0  # 所有点都一样的服务时间，23:00-4:00
        self.time_window_b = 5
        self.bigM = 1e3
        # self.coordinates = [(0, 0), (2, 3), (5, 8), (6, 1), (8, 4), (3, 7)]  # 损伤点, (0,0)车场
        self.CAP = 4  # 养护车辆的最大载客量
        self.t_matrix = pd.read_excel('./t_matrix.xlsx', header=None).to_numpy()  # 时间矩阵/h
        self.d_matrix = pd.read_excel('./d_matrix.xlsx', header=None).to_numpy()  # 距离矩阵/km
        self.MPI = list(pd.read_excel('./area+MPI.xlsx')['MPI'])  # 各点的MPI值
        self.area = list(pd.read_excel('./area+MPI.xlsx')['area'])  # 各点的面积/长度
        self.dis_type = list(pd.read_excel('./area+MPI.xlsx')['type'])  # 各点的损伤类型
        self.renxiao = {0: 100000, 1: 11.141, 2: 1.937, 3: 1.937}  # 1为裂缝维修人效，2为网裂和坑槽的维修人效，单位为米/h/人和平方米/h/人
        # 每个损伤的维修人效用字典索引，即renxiao[distype],下面费用同理
        self.c_worker = 112  # 单日人工费，元
        self.c_machine = {0: 0, 1: 17.06, 2: 53.24, 3: 53.24}  # 单位机械费，元/h
        self.c_material = {0: 0, 1: 9.83, 2: 43.52, 3: 43.52}  # 单位材料费，元/米、元/平方米
        self.c_fuel = 1.08  # 单位油费，元/km
        # 集合
        self.V = range(len(self.MPI))  # 节点索引(0是维护站点，1～n是病害点)
        self.K = range(self.vehicle_number)  # 车辆集合数
        self.P = range(self.people_number_per_vehicle)  # 每车人人数4人
        # 参数
        self.M = self.bigM

        self.time_window = [(self.time_window_a, self.time_window_b) for _ in range(len(self.V))]
        # 决策变量
        self.x = self.model.addVars(self.V, self.V, self.K, vtype=GRB.BINARY, name='x')  # x(i,j,k)
        self.y = self.model.addVars(self.V, self.K, vtype=GRB.BINARY, name='y')  # y(i,k)
        self.u = self.model.addVars(self.V, lb=0, vtype=GRB.CONTINUOUS, name='u')  # u(i)
        self.t = self.model.addVars(self.V, self.K, vtype=GRB.CONTINUOUS, name='t')  # t(i,k)
        # self.p = self.model.addVars(self.K, lb=0, vtype=GRB.INTEGER, name="p")  # p(k)
        self.p = [self.people_number_per_vehicle] * self.vehicle_number  # 约束固定

        # 结果收集
        self.routes = list()

    def set_objective(self):
        '''总MPI最大化'''
        # todo: 目标函数1解到150s 保存中间状态解，输出的目标函数2
        self.model.setObjectiveN(
            gp.quicksum(-self.MPI[i] * self.y[i, k] for i in self.V for k in self.K),
            index=0, priority=2
        )

        # '''总费用最小化'''
        # self.model.setObjectiveN(
        #     gp.quicksum(self.p[k] * self.c_worker for k in self.K)
        #     + gp.quicksum(
        #         self.t[i, k] * self.c_machine[self.dis_type[i]] + self.area[i] * self.c_material[self.dis_type[i]] *
        #         self.y[i, k] for i in self.V for k in self.K)
        #     + gp.quicksum(
        #         self.x[i, j, k] * self.d_matrix[i, j] * self.c_fuel for i in self.V for j in self.V for k in self.K),
        #     index=1, priority=1
        # )

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
            self.y[i, k] <= gp.quicksum(self.x[i, j, k] for j in self.V if j != i)
            for i in self.V for k in self.K
        )

    def constraint_time_window(self):
        self.model.addConstrs(self.u[i] >= self.time_window[i][0] for i in self.V)
        self.model.addConstrs(self.u[i] <= self.time_window[i][1] for i in self.V)

    '''维修时间计算'''

    def constraint_time_calculation(self):
        self.model.addConstrs(
            self.t[i, k] == self.area[i] * self.y[i, k] / self.people_number_per_vehicle
            / self.renxiao[self.dis_type[i]]
            # + random.randint(5,10)/60   #随机加上5-10分钟
            for i in self.V for k in self.K
        )

    def set_constraints(self):
        self.constraint_time_window()  # (5.17)
        self.constraint_visit()  # (5.18)
        self.constraint_flow_balance()  # (5.19)
        self.constraint_deport_depart()  # (5.20-5.21)
        self.constraint_x_leq_y()  # (5.22)
        self.constraint_time_continuity()  # (5.23)
        '''最后两个约束不确定'''
        self.constraint_time_calculation()  # (5.25)

    def set_model_parameters(self):
        '''求解时间限制'''
        self.model.Params.TimeLimit = 300  # 设置300s
        self.model.Params.OutputFlag = 1
        self.model.Params.IntegralityFocus = 1  # 防止0-1变量有小数位
        self.model.Params.PoolGap = 0.1  # 距离最优解的gap

        # self.model.Params.LicenseID = 2613921

    def model_solver_print(self):
        if self.model.status == GRB.OPTIMAL:
            print(f"最优总距离(settled_obj_value) = {self.model.objVal:.2f}")
            for key in self.x.keys():
                if self.x[key].x > 0:
                    print(f"{self.x[key].VarName}= {self.x[key].x}")

    def model_result_analysis(self):
        # if self.model.status == GRB.OPTIMAL:
        obj_val = None
        if 1:
            # print(f"MPI总值：{self.model.objVal:.2f}")
            obj_val = self.model.getObjective(0).getValue()
            print(f"MPI总值：{self.model.getObjective(0).getValue()}")
            print(f"总费用：{self.model.getObjective(1).getValue()}")

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

                print(f"\n车辆:{k + 1}路径:{','.join(map(str, route))}")
                print(f"到达时间: {[self.u[i].X for i in route]}")
                print(f"维修的损伤点数：{len(route)}")
                # print(f"k={k}, {self.routes[k]}")
            return obj_val

    def solve(self):
        self.set_constraints()
        self.set_objective()
        self.model.write('gq_model.lp')
        self.set_model_parameters()
        self.model.optimize()
        self.model_solver_print()
        obj_val = self.model_result_analysis()
        return obj_val


if __name__ == "__main__":
    print_hi("PyCharm")
    gq_model = GModel()
    obj_vali = gq_model.solve()
