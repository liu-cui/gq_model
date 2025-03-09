from solver_obj_1 import GModel
from solver_obj_2 import G2Model
if __name__ == "__main__":
    # 执行 solver1，获取模型和结果
    gq_model = GModel()
    f1_val = gq_model.solve()    # 将模型传递给 solver2
    gq2_model = G2Model(f1_val)
    gq2_model.solve()