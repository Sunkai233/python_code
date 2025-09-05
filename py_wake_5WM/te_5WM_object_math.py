import pandas as pd
import numpy as np
from py_wake.wind_turbines import WindTurbine
from py_wake.wind_turbines.power_ct_functions import PowerCtTabular
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt

# === 1. 读取数据 ===
df = pd.read_csv("wind_turbine_results.csv")

# === 2. 数据预处理 ===

# 删除缺失值
df = df.dropna()

# 按风速升序排序，并去除重复值
df = df.sort_values(by="WindSpeed")
df = df.drop_duplicates(subset="WindSpeed", keep='first')

# 转为一维数组 + 类型转换
wind_speeds = np.asarray(df["WindSpeed"]).flatten().astype(float)
power_values = np.asarray(df["Power"]).flatten().astype(float)
ct_values = np.asarray(df["Ct"]).flatten().astype(float)

# === 3. 调试用断言 ===
assert wind_speeds.ndim == 1, "wind_speeds must be 1D"
assert power_values.ndim == 1, "power_values must be 1D"
assert ct_values.ndim == 1, "ct_values must be 1D"
assert len(wind_speeds) == len(power_values) == len(ct_values), "所有数组长度必须一致"

# === 4. 创建 PowerCtTabular ===
power_ct_func = PowerCtTabular(
    ws=wind_speeds,
    power=power_values,
    power_unit="kW",
    ct=ct_values,
    method="pchip"
)

# === 5. 构建风机对象 ===
custom_turbine = WindTurbine(
    name="MyTurbineFromData",
    diameter=126,       # 例如 NREL 5MW 的转子直径
    hub_height=90,      # 轮毂高度
    powerCtFunction=power_ct_func
)

# === 6. 绘图展示插值结果 ===
if __name__ == "__main__":
    ws_plot = np.linspace(0, 30, 300)
    p = custom_turbine.power(ws_plot)
    ct = custom_turbine.ct(ws_plot)

    plt.figure(figsize=(12, 5))

    # Power 曲线
    plt.subplot(1, 2, 1)
    plt.plot(ws_plot, p, label="Interpolated Power")
    plt.scatter(wind_speeds, power_values*1000, color='red', label="Original Data")
    plt.xlabel("Wind Speed (m/s)")
    plt.ylabel("Power (kW)")
    plt.title("Power Curve")
    plt.legend()
    plt.grid()

    # Ct 曲线
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
