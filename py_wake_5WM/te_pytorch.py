import torch
from torch import nn
from py_wake.wind_turbines.power_ct_functions import PowerCtFunction
import numpy as np

# 你的网络定义
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 25),
            nn.Tanh(),
            nn.Linear(25, 25),
            nn.Tanh(),
            nn.Linear(25, 2)
        )

    def forward(self, x):
        return self.net(x)

# 包装PowerCtFunction的预测模型
class NNPowerCtModel(PowerCtFunction):
    def __init__(self, model_path):
        super().__init__(['ws'], ['Air_density', 'TI_eff'], power_unit='kW')
        self.device = torch.device('cpu')

        self.model = Net()
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def predict_nn(self, ws):
        with torch.no_grad():
            ws_tensor = torch.tensor(ws, dtype=torch.float32).unsqueeze(-1).to(self.device)  # ws shape (N,1)
            pred = self.model(ws_tensor).cpu().numpy()  # shape (N,2)
        return pred

    def power(self, ws, **kwargs):
        pred = self.predict_nn(ws)
        return pred[:, 0]

    def ct(self, ws, **kwargs):
        pred = self.predict_nn(ws)
        return pred[:, 1]

# 使用示例
model_path = "wind_power_ct_best.pt"  # 你的模型参数文件
nn_power_ct_model = NNPowerCtModel(model_path)

from py_wake.wind_turbines import WindTurbine
custom_nn_turbine = WindTurbine(
    name="NN_Predicted_Turbine",
    diameter=130,
    hub_height=110,
    powerCtFunction=nn_power_ct_model
)

# 测试预测
test_ws = np.array([3.0, 5.0, 7.0, 10.0])
print("风速:", test_ws)
print("预测功率:", custom_nn_turbine.power(test_ws))
print("预测推力系数Ct:", custom_nn_turbine.ct(test_ws))
