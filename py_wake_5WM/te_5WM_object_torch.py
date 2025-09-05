import torch
import numpy as np
from py_wake.wind_turbines import WindTurbine
from py_wake.wind_turbines.power_ct_functions import PowerCtFunction
from torch import nn
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
import pandas as pd


# 1. 定义PyTorch神经网络模型（仅接受ws输入）
class PowerCtNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 25),  # 输入仅ws，所以输入维度为1
            nn.Tanh(),
            nn.Linear(25, 25),
            nn.Tanh(),
            nn.Linear(25, 2)  # 输出: power, ct
        )

    def forward(self, x):
        return self.net(x)

# 2. 定义功率和CT计算函数（仅使用ws）
def nn_power_ct(ws, run_only):
    """
    使用神经网络模型计算功率和CT
    :param ws: 风速数组
    :param run_only: 0=只计算功率, 1=只计算CT
    :return: 功率或CT值
    """
    # 确保模型已加载
    if not hasattr(nn_power_ct, 'model'):
        # 初始化模型
        nn_power_ct.model = PowerCtNet()
        state_dict = torch.load("wind_power_ct_best.pt", map_location='cpu')
        nn_power_ct.model.load_state_dict(state_dict)
        nn_power_ct.model.eval()

    # 准备输入数据（仅ws）
    ws_array = np.asarray(ws)
    input_data = ws_array.reshape(-1, 1)  # 形状变为(n_samples, 1)

    # 转换为tensor并进行预测
    with torch.no_grad():
        inputs = torch.tensor(input_data, dtype=torch.float32)
        outputs = nn_power_ct.model(inputs).numpy()

    # 根据run_only返回相应结果
    if run_only == 0:  # 功率
        return outputs[:, 0]
    elif run_only == 1:  # CT
        return outputs[:, 1]
    else:
        return outputs  # 返回两者

'''
# 3. 创建自定义PowerCtFunction类
class NNTurbinePowerCt(PowerCtFunction):
    def __init__(self):
        super().__init__(
            input_keys='ws',
            power_ct_func=nn_power_ct,
            power_unit='kW',
            optional_inputs=['Air_density', 'TI_eff']
        )
'''
class NNTurbinePowerCt(PowerCtFunction):
    def __init__(self):
        super().__init__(input_keys ='ws', power_ct_func=nn_power_ct,optional_inputs=['Air_density', 'TI_eff'],power_unit = 'kw')  # 声明输入参数

    def power(self, ws, Air_density=1.225, TI_eff=0.1):
        return nn_power_ct(ws,0)

    def ct(self, ws, Air_density=1.225, TI_eff=0.1):
        # 自定义CT曲线
        return nn_power_ct(ws,1)

# 4. 创建风力机实例
custom_nn_turbine = WindTurbine(
    name="NN_Powered_Turbine",
    diameter=126,
    hub_height=90,
    powerCtFunction=NNTurbinePowerCt()
)
if __name__ == "__main__":
    # === 1. 读取数据 ===
    df = pd.read_csv("wind_turbine_results.csv")

    # 确保按风速升序排列（PowerCtTabular 要求风速单调）
    df = df.sort_values(by="WindSpeed")
    df = df.drop_duplicates(subset="WindSpeed", keep='first')  # 如果有重复，保留第一个

    # === 2. 提取列 ===
    wind_speeds = df["WindSpeed"].values
    power_values = df["Power"].values*1000
    ct_values = df["Ct"].values

    ws_plot = np.linspace(0, 30, 300)
    p = custom_nn_turbine.power(ws_plot)
    ct = custom_nn_turbine.ct(ws_plot)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(ws_plot, p, label="Interpolated Power")
    plt.scatter(wind_speeds, power_values, color='red', label="Original Data")
    plt.xlabel("Wind Speed (m/s)")
    plt.ylabel("Power (kW)")
    plt.title("Power Curve")
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(ws_plot, ct, label="Interpolated Ct", color='green')
    plt.scatter(wind_speeds, ct_values, color='red', label="Original Ct")
    plt.xlabel("Wind Speed (m/s)")
    plt.ylabel("Ct")
    plt.title("Ct Curve")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()
