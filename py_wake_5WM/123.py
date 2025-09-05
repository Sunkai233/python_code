from py_wake.wind_turbines.power_ct_functions import PowerCtFunction
import numpy as np
from py_wake.wind_turbines import WindTurbine


def density_scaled_power_ct(ws, run_only, Air_density=1.225, TI_eff=0.1):
    rho = Air_density
    if run_only==0:  # 计算功率
        rated_power = 3e6
        density_scale = rho/.95
        return 0
    elif run_only==1:  # 计算CT
        return 0  # 示例中未使用CT

class MyPowerCtModel(PowerCtFunction):
    def __init__(self):
        super().__init__(input_keys ='ws', power_ct_func=density_scaled_power_ct,optional_inputs=['Air_density', 'TI_eff'],power_unit = 'kw')  # 声明输入参数

    def power(self, ws, Air_density=1.225, TI_eff=0.1):
        # 自定义功率计算逻辑
        rated_power = 2000  # kW
        cut_in = 4  # m/s
        rated_ws = 12  # m/s

        power = np.zeros_like(ws)
        mask = (ws >= cut_in) & (ws <= rated_ws)
        power[mask] = rated_power * ((ws[mask] - cut_in) / (rated_ws - cut_in)) ** 2

        # 考虑空气密度修正
        power *= (Air_density / 1.225)
        return density_scaled_power_ct(ws,0)

    def ct(self, ws, Air_density=1.225, TI_eff=0.1):
        # 自定义CT曲线
        return density_scaled_power_ct(ws,1)


custom_turbine = WindTurbine(
    name='DynamicTurbine',
    diameter=130,
    hub_height=110,
    powerCtFunction=MyPowerCtModel()
)


# 3. 测试预测示例
test_ws = np.array([3.0, 5.0, 7.0, 10.0])
print("风速:", test_ws)
print("预测功率:", custom_turbine.power(test_ws))
print("预测推力系数Ct:", custom_turbine.ct(test_ws))