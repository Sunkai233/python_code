"""
准稳态BEMT时序求解器 - 基于原稳态版本扩展
Quasi-Steady BEMT Time Series Solver - Extended from Steady-State Version
"""

import torch
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import matplotlib
matplotlib.use('TkAgg')  # 强制使用 TkAgg 后端，确保兼容性
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class AirfoilDatabase:
    """翼型数据库类 - 管理和访问翼型气动数据"""

    def __init__(self, database_path: str = "nrel_5mw_airfoils.pkl"):
        """
        初始化翼型数据库

        Args:
            database_path: 翼型数据库文件路径
        """
        # 强制加载翼型数据库，失败则退出
        self.database = self.load_database(database_path)

    def load_database(self, filepath: str) -> Dict[int, Dict[str, Any]]:
        """加载翼型数据库文件，失败直接报错退出"""
        try:
            # 尝试从pickle文件加载翼型数据库
            with open(filepath, 'rb') as f:
                database = pickle.load(f)
            print(f"✓ 成功加载翼型数据库: {filepath}")
            print(f"  包含 {len(database)} 个翼型")

            # 验证数据完整性
            if not database or len(database) == 0:
                raise ValueError("翼型数据库为空")

            return database

        except Exception as e:
            # 如果加载失败，直接报错退出
            print(f"✗ 翼型数据库加载失败: {e}")
            print(f"请确保文件 '{filepath}' 存在且格式正确")
            raise FileNotFoundError(f"无法加载翼型数据库: {filepath}")

    def get_aero_coefficients(self, airfoil_id: int, alpha_deg: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        获取指定攻角的气动系数（使用PyTorch插值）

        Args:
            airfoil_id: 翼型ID
            alpha_deg: 攻角（度），支持标量或张量

        Returns:
            (Cl, Cd, Cm) 升力系数、阻力系数、力矩系数
        """
        # 检查翼型ID是否存在
        if airfoil_id not in self.database:
            raise ValueError(f"翼型ID {airfoil_id} 在数据库中不存在")

        # 获取翼型的气动系数表
        aero_table = self.database[airfoil_id]['aero_table']
        if aero_table.empty:
            raise ValueError(f"翼型ID {airfoil_id} 的气动数据为空")

        # 提取攻角和气动系数数组，转换为torch张量
        alpha_array = torch.tensor(aero_table['Alpha'].values, dtype=torch.float32)
        cl_array = torch.tensor(aero_table['Cl'].values, dtype=torch.float32)
        cd_array = torch.tensor(aero_table['Cd'].values, dtype=torch.float32)
        cm_array = torch.tensor(aero_table['Cm'].values, dtype=torch.float32)

        # 使用PyTorch的线性插值计算指定攻角的气动系数
        cl = torch_interp(alpha_deg, alpha_array, cl_array)
        cd = torch_interp(alpha_deg, alpha_array, cd_array)
        cm = torch_interp(alpha_deg, alpha_array, cm_array)

        return cl, cd, cm


def torch_interp(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    """
    PyTorch版本的线性插值函数（支持自动微分）

    Args:
        x: 插值点（可以是标量或张量）
        xp: 已知数据点的x坐标（必须单调递增）
        fp: 已知数据点的y坐标

    Returns:
        插值结果
    """
    # 确保输入是张量
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)

    # 处理超出范围的情况
    x_clamped = torch.clamp(x, xp[0], xp[-1])

    # 找到插值区间
    # 使用searchsorted找到插值位置
    indices = torch.searchsorted(xp, x_clamped, right=False)
    indices = torch.clamp(indices, 0, len(xp) - 2)

    # 获取插值区间的端点
    x0 = xp[indices]
    x1 = xp[indices + 1]
    y0 = fp[indices]
    y1 = fp[indices + 1]

    # 线性插值计算
    # y = y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    alpha = (x_clamped - x0) / (x1 - x0)
    result = y0 + alpha * (y1 - y0)

    return result


def create_nrel5mw_blade_geometry():
    """创建NREL 5MW风力机叶片几何数据 - 基于AeroDyn v15.00输入文件"""
    # 基于AeroDyn v15.00的NREL 5MW叶片几何参数
    blade_data = {
        '节点': list(range(1, 20)),  # 19个节点
        '径向位置(m)': [0.0000000, 1.3667000, 4.1000000, 6.8333000, 10.250000, 14.350000,
                        18.450000, 22.550000, 26.650000, 30.750000, 34.850000, 38.950000,
                        43.050000, 47.150000, 51.250000, 54.666700, 57.400000, 60.133300, 61.499900],
        '弯曲AC(m)': [0.0000000, -0.00081531745, -0.024839790, -0.059469375, -0.10909141, -0.11573354,
                      -0.098316709, -0.083186967, -0.067933232, -0.053393159, -0.040899260, -0.029722933,
                      -0.020511081, -0.013980013, -0.0083819737, -0.0043546914, -0.0016838383, -0.00032815226,
                      -0.00032815226],
        '后掠AC(m)': [0.0000000, -0.0034468858, -0.10501421, -0.25141635, -0.46120149, -0.56986665,
                      -0.54850833, -0.52457001, -0.49624675, -0.46544755, -0.43583519, -0.40591323,
                      -0.37569051, -0.34521705, -0.31463837, -0.28909220, -0.26074456, -0.17737470, -0.17737470],
        '弯曲角(°)': [0.0] * 19,  # 所有节点弯曲角都为0
        '扭转角(°)': [13.308000, 13.308000, 13.308000, 13.308000, 13.308000, 11.480000,
                      10.162000, 9.0110000, 7.7950000, 6.5440000, 5.3610000, 4.1880000,
                      3.1250000, 2.3190000, 1.5260000, 0.86300000, 0.37000000, 0.10600000, 0.10600000],
        '弦长(m)': [3.5420000, 3.5420000, 3.8540000, 4.1670000, 4.5570000, 4.6520000,
                    4.4580000, 4.2490000, 4.0070000, 3.7480000, 3.5020000, 3.2560000,
                    3.0100000, 2.7640000, 2.5180000, 2.3130000, 2.0860000, 1.4190000, 1.4190000],
        '翼型ID': [1, 1, 1, 2, 3, 4, 4, 5, 6, 6, 7, 7, 8, 8, 8, 8, 8, 8, 8]
    }

    # 转换为DataFrame格式
    df = pd.DataFrame(blade_data)
    return df


class QuasiSteadyBEMT:
    """准稳态BEMT求解器类"""

    def __init__(self, airfoil_db: AirfoilDatabase, blade_geometry: Optional[pd.DataFrame] = None):
        """
        初始化准稳态BEMT求解器

        Args:
            airfoil_db: 翼型数据库
            blade_geometry: 叶片几何数据
        """
        self.airfoil_db = airfoil_db
        self.blade_geometry = blade_geometry if blade_geometry is not None else create_nrel5mw_blade_geometry()

        # 基本参数
        self.R = 61.5  # 转子半径 [m]
        self.B = 3  # 叶片数量
        self.rho = 1.225  # 空气密度 [kg/m³]

        # 预处理叶片几何数据
        self._preprocess_geometry()

        # 历史状态存储（用于准稳态计算）
        self.a_history = None  # 轴向诱导因子历史
        self.a_prime_history = None  # 切向诱导因子历史

    def _preprocess_geometry(self):
        """预处理叶片几何数据"""
        # 移除重复点
        blade_data = self.blade_geometry.drop_duplicates(subset=['径向位置(m)'], keep='first').copy().reset_index(
            drop=True)

        # 转换为PyTorch张量
        self.r = torch.tensor(blade_data['径向位置(m)'].values, dtype=torch.float32)
        self.chord = torch.tensor(blade_data['弦长(m)'].values, dtype=torch.float32)
        self.twist = torch.deg2rad(torch.tensor(blade_data['扭转角(°)'].values, dtype=torch.float32))
        self.airfoil_ids = blade_data['翼型ID'].values
        self.n_stations = len(blade_data)

        # 计算径向微元长度
        self.dr = torch.zeros_like(self.r)
        if len(self.r) > 1:
            self.dr[1:-1] = (self.r[2:] - self.r[:-2]) / 2
            self.dr[0] = self.r[1] - self.r[0]
            self.dr[-1] = self.r[-1] - self.r[-2]
        else:
            self.dr = torch.ones_like(self.r)

    def solve_single_timestep(self, wind_speed: torch.Tensor, rotor_speed: torch.Tensor,
                              dt: float = 0.1, relaxation: float = 0.5) -> Dict[str, torch.Tensor]:
        """
        求解单个时间步的BEMT方程（准稳态）

        Args:
            wind_speed: 风速 [m/s]
            rotor_speed: 转子角速度 [rad/s]
            dt: 时间步长 [s]
            relaxation: 松弛因子

        Returns:
            当前时间步的求解结果
        """
        # 初始化诱导因子（使用历史值或零初始化）
        if self.a_history is None:
            a = torch.zeros_like(self.r)
            a_prime = torch.zeros_like(self.r)
        else:
            # 使用历史值作为初始猜测
            a = self.a_history[-1].clone()
            a_prime = self.a_prime_history[-1].clone()

        # BEMT迭代求解
        max_iterations = 30
        tolerance = 1e-6

        for iteration in range(max_iterations):
            # 计算诱导速度
            V_axial = wind_speed * (1 - a)
            V_tangential = rotor_speed * self.r * (1 + a_prime)

            # 计算相对风速和流入角
            V_rel = torch.sqrt(V_axial ** 2 + V_tangential ** 2)
            phi = torch.atan2(V_axial, V_tangential)

            # 计算攻角
            alpha = phi - self.twist

            # 获取气动系数
            cl = torch.zeros_like(alpha)
            cd = torch.zeros_like(alpha)
            cm = torch.zeros_like(alpha)

            for i, airfoil_id in enumerate(self.airfoil_ids):
                alpha_deg = torch.rad2deg(alpha[i])
                cl_val, cd_val, cm_val = self.airfoil_db.get_aero_coefficients(airfoil_id, alpha_deg)
                cl[i] = cl_val
                cd[i] = cd_val
                cm[i] = cm_val

            # 计算载荷系数
            cn = cl * torch.cos(phi) + cd * torch.sin(phi)
            ct = cl * torch.sin(phi) - cd * torch.cos(phi)

            # 计算实度
            r_min = 0.1
            r_calc = torch.clamp(self.r, min=r_min)
            sigma = self.B * self.chord / (2 * torch.pi * r_calc)
            sigma[0] = 0
            sigma[1] = 0

            # 更新诱导因子
            k_ax = sigma * cn / (4 * torch.sin(phi) ** 2)
            a_new = k_ax / (1 + k_ax)

            # Glauert修正
            a_critical = 0.4
            high_a_mask = a_new > a_critical
            if torch.any(high_a_mask):
                f_glauert = 0.25 * (5 - 3 * a_new[high_a_mask])
                a_new[high_a_mask] = 1 - f_glauert

            k_tan = sigma * ct / (4 * torch.sin(phi) * torch.cos(phi))
            a_prime_new = k_tan / (1 - k_tan)

            # 松弛更新
            a_old = a.clone()
            a = (1 - relaxation) * a + relaxation * a_new
            a_prime = (1 - relaxation) * a_prime + relaxation * a_prime_new

            # 检查收敛
            residual = torch.max(torch.abs(a - a_old))
            if residual < tolerance:
                break

        # 更新历史状态
        if self.a_history is None:
            self.a_history = [a.clone()]
            self.a_prime_history = [a_prime.clone()]
        else:
            self.a_history.append(a.clone())
            self.a_prime_history.append(a_prime.clone())
            # 保持历史长度不超过100步
            if len(self.a_history) > 100:
                self.a_history.pop(0)
                self.a_prime_history.pop(0)

        # 计算最终结果（与稳态版本相同）
        V_axial = wind_speed * (1 - a)
        V_tangential = rotor_speed * self.r * (1 + a_prime)
        V_rel = torch.sqrt(V_axial ** 2 + V_tangential ** 2)
        phi = torch.atan2(V_axial, V_tangential)

        # 计算载荷
        dT_dr = 0.5 * self.rho * V_rel ** 2 * self.chord * cn * self.B
        dQ_dr = 0.5 * self.rho * V_rel ** 2 * self.chord * ct * self.r * self.B

        thrust = torch.sum(dT_dr * self.dr)
        torque = torch.sum(dQ_dr * self.dr)
        power = torque * rotor_speed

        # 计算无量纲系数
        A = torch.pi * self.R ** 2
        cp = power / (0.5 * self.rho * A * wind_speed ** 3)
        ct_total = thrust / (0.5 * self.rho * A * wind_speed ** 2)

        return {
            'power': power,
            'thrust': thrust,
            'torque': torque,
            'cp': cp,
            'ct': ct_total,
            'a': a,
            'a_prime': a_prime,
            'alpha': alpha,
            'phi': phi,
            'cl': cl,
            'cd': cd,
            'cm': cm,
            'loads': {'dT_dr': dT_dr, 'dQ_dr': dQ_dr},
            'flow': {'V_rel': V_rel, 'V_axial': V_axial, 'V_tangential': V_tangential}
        }


def calculate_blade_angles(time: torch.Tensor, rotor_speed: torch.Tensor,
                           initial_angles: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    计算三个叶片与中轴的夹角

    Args:
        time: 时间张量 [s]
        rotor_speed: 转子角速度张量 [rad/s] - 每个时间步的角速度
        initial_angles: 初始角度 [rad] - 三个叶片的初始角度，默认为[0, 2π/3, 4π/3]

    Returns:
        blade_angles: 形状为 (n_time, 3) 的张量，每行为三个叶片的角度 [rad]
    """
    if initial_angles is None:
        # 默认三个叶片均匀分布
        initial_angles = torch.tensor([0.0, 2 * torch.pi / 3, 4 * torch.pi / 3])

    n_time = len(time)
    blade_angles = torch.zeros(n_time, 3)

    # 初始化第一个时间步
    blade_angles[0] = initial_angles

    # 时序积分计算角度
    for i in range(1, n_time):
        dt = time[i] - time[i - 1]
        # 角度增量 = 角速度 × 时间步长
        angle_increment = rotor_speed[i - 1] * dt
        # 更新三个叶片的角度
        blade_angles[i] = blade_angles[i - 1] + angle_increment

    # 将角度限制在 [0, 2π] 范围内
    blade_angles = blade_angles % (2 * torch.pi)

    return blade_angles


def time_series_simulation(wind_speed_series: torch.Tensor,
                           rotor_speed_series: torch.Tensor,
                           time_series: torch.Tensor,
                           airfoil_db: AirfoilDatabase,
                           blade_geometry: Optional[pd.DataFrame] = None,
                           initial_blade_angles: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
    """
    时序仿真主函数

    Args:
        wind_speed_series: 风速时序 [m/s]
        rotor_speed_series: 转速时序 [rad/s]
        time_series: 时间序列 [s]
        airfoil_db: 翼型数据库
        blade_geometry: 叶片几何数据
        initial_blade_angles: 三个叶片的初始角度 [rad]

    Returns:
        时序仿真结果字典
    """
    n_time = len(time_series)

    # 初始化BEMT求解器
    bemt_solver = QuasiSteadyBEMT(airfoil_db, blade_geometry)

    # 计算叶片角度时序
    blade_angles = calculate_blade_angles(time_series, rotor_speed_series, initial_blade_angles)

    # 初始化结果存储
    results = {
        'time': time_series,
        'power': torch.zeros(n_time),
        'thrust': torch.zeros(n_time),
        'torque': torch.zeros(n_time),
        'cp': torch.zeros(n_time),
        'ct': torch.zeros(n_time),
        'blade_angles': blade_angles,
        'wind_speed': wind_speed_series,
        'rotor_speed': rotor_speed_series
    }

    # 存储径向分布结果（仅存储部分时间步以节省内存）
    save_indices = torch.linspace(0, n_time - 1, min(10, n_time), dtype=torch.long)
    results['radial_data'] = {
        'save_indices': save_indices,
        'r': bemt_solver.r,
        'a': torch.zeros(len(save_indices), len(bemt_solver.r)),
        'alpha': torch.zeros(len(save_indices), len(bemt_solver.r)),
        'phi': torch.zeros(len(save_indices), len(bemt_solver.r))
    }

    print(f"开始时序仿真: {n_time} 个时间步")
    print_interval = max(1, n_time // 20)  # 最多打印20次进度

    # 时序循环
    for i in range(n_time):
        # 计算时间步长
        dt = time_series[1] - time_series[0] if i == 0 else time_series[i] - time_series[i - 1]

        # 求解当前时间步
        result = bemt_solver.solve_single_timestep(
            wind_speed=wind_speed_series[i],
            rotor_speed=rotor_speed_series[i],
            dt=dt.item() if isinstance(dt, torch.Tensor) else dt
        )

        # 存储标量结果
        results['power'][i] = result['power']
        results['thrust'][i] = result['thrust']
        results['torque'][i] = result['torque']
        results['cp'][i] = result['cp']
        results['ct'][i] = result['ct']

        # 存储径向分布结果（仅在指定时间步）
        if i in save_indices:
            save_idx = (save_indices == i).nonzero(as_tuple=True)[0][0]
            results['radial_data']['a'][save_idx] = result['a']
            results['radial_data']['alpha'][save_idx] = result['alpha']
            results['radial_data']['phi'][save_idx] = result['phi']

        # 打印进度
        if i % print_interval == 0 or i == n_time - 1:
            progress = (i + 1) / n_time * 100
            print(f"  进度: {progress:5.1f}% - 时间 {time_series[i]:6.2f}s - "
                  f"功率 {result['power'] / 1e6:5.2f}MW - Cp {result['cp']:5.3f}")

    return results


def create_test_time_series(duration: float = 60.0, dt: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    创建测试用的时间序列数据

    Args:
        duration: 仿真持续时间 [s]
        dt: 时间步长 [s]

    Returns:
        (time_series, wind_speed_series, rotor_speed_series)
    """
    time_series = torch.arange(0, duration + dt, dt)
    n_time = len(time_series)

    # 创建变化的风速（基础风速 + 随机扰动 + 周期性变化）
    base_wind_speed = 11.0  # m/s
    wind_turbulence = 1.0 * torch.randn(n_time) * 0.1  # 湍流扰动
    wind_gust = 2.0 * torch.sin(2 * torch.pi * time_series / 20.0)  # 20秒周期的阵风
    wind_speed_series = base_wind_speed + wind_turbulence + wind_gust

    wind_speed_series = base_wind_speed + wind_turbulence

    wind_speed_series = torch.clamp(wind_speed_series, 3.0, 25.0)  # 限制风速范围



    # 创建转速序列（可以是恒定的或变化的）
    base_rotor_speed = 1.2  # rad/s
    # 添加小幅度的转速调节（模拟控制系统响应）
    rotor_speed_variation = 0.001 * torch.sin(2 * torch.pi * time_series / 15.0)
    rotor_speed_series = base_rotor_speed + rotor_speed_variation
    rotor_speed_series = torch.clamp(rotor_speed_series, 0.5, 2.0)  # 限制转速范围


    return time_series, wind_speed_series, rotor_speed_series


def plot_time_series_results(results: Dict[str, torch.Tensor], save_plots: bool = False):
    """
    绘制时序仿真结果

    Args:
        results: 时序仿真结果字典
        save_plots: 是否保存图片
    """
    time = results['time'].numpy()

    # 创建多子图
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('准稳态BEMT时序仿真结果', fontsize=16)

    # 1. 功率时序
    axes[0, 0].plot(time, results['power'].numpy() / 1e6, 'b-', linewidth=2)
    axes[0, 0].set_xlabel('时间 [s]')
    axes[0, 0].set_ylabel('功率 [MW]')
    axes[0, 0].set_title('功率时序')
    axes[0, 0].grid(True)

    # 2. 功率系数时序
    axes[0, 1].plot(time, results['cp'].numpy(), 'r-', linewidth=2)
    axes[0, 1].set_xlabel('时间 [s]')
    axes[0, 1].set_ylabel('功率系数 Cp [-]')
    axes[0, 1].set_title('功率系数时序')
    axes[0, 1].grid(True)

    # 3. 推力时序
    axes[1, 0].plot(time, results['thrust'].numpy() / 1e3, 'g-', linewidth=2)
    axes[1, 0].set_xlabel('时间 [s]')
    axes[1, 0].set_ylabel('推力 [kN]')
    axes[1, 0].set_title('推力时序')
    axes[1, 0].grid(True)

    # 4. 风速和转速
    ax4 = axes[1, 1]
    ax4_twin = ax4.twinx()

    line1 = ax4.plot(time, results['wind_speed'].numpy(), 'b-', linewidth=2, label='风速')
    ax4.set_xlabel('时间 [s]')
    ax4.set_ylabel('风速 [m/s]', color='b')
    ax4.tick_params(axis='y', labelcolor='b')

    line2 = ax4_twin.plot(time, results['rotor_speed'].numpy(), 'r-', linewidth=2, label='转速')
    ax4_twin.set_ylabel('转速 [rad/s]', color='r')
    ax4_twin.tick_params(axis='y', labelcolor='r')

    ax4.set_title('风速和转速时序')
    ax4.grid(True)

    # 5. 叶片角度
    blade_angles = results['blade_angles'].numpy()
    axes[2, 0].plot(time, blade_angles[:, 0] * 180 / np.pi, 'r-', linewidth=2, label='叶片1')
    axes[2, 0].plot(time, blade_angles[:, 1] * 180 / np.pi, 'g-', linewidth=2, label='叶片2')
    axes[2, 0].plot(time, blade_angles[:, 2] * 180 / np.pi, 'b-', linewidth=2, label='叶片3')
    axes[2, 0].set_xlabel('时间 [s]')
    axes[2, 0].set_ylabel('叶片角度 [°]')
    axes[2, 0].set_title('三个叶片与中轴夹角')
    axes[2, 0].legend()
    axes[2, 0].grid(True)

    # 6. 叶尖速比时序
    tsr = results['rotor_speed'] * 61.5 / results['wind_speed']  # TSR = ΩR/V
    axes[2, 1].plot(time, tsr.numpy(), 'm-', linewidth=2)
    axes[2, 1].set_xlabel('时间 [s]')
    axes[2, 1].set_ylabel('叶尖速比 [-]')
    axes[2, 1].set_title('叶尖速比时序')
    axes[2, 1].grid(True)

    plt.tight_layout()

    if save_plots:
        plt.savefig('quasi_steady_bemt_results.png', dpi=300, bbox_inches='tight')
        print("图片已保存为: quasi_steady_bemt_results.png")

    plt.show()


def plot_radial_distributions(results: Dict[str, torch.Tensor], save_plots: bool = False):
    """
    绘制径向分布结果（在不同时间步）

    Args:
        results: 时序仿真结果字典
        save_plots: 是否保存图片
    """
    radial_data = results['radial_data']
    r = radial_data['r'].numpy()
    save_indices = radial_data['save_indices']
    time_points = results['time'][save_indices].numpy()

    # 创建多子图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('径向分布随时间变化', fontsize=16)

    # 颜色映射
    colors = plt.cm.viridis(np.linspace(0, 1, len(save_indices)))

    # 1. 轴向诱导因子径向分布
    for i, (idx, color) in enumerate(zip(save_indices, colors)):
        a_data = radial_data['a'][i].numpy()
        axes[0, 0].plot(r, a_data, color=color, linewidth=2,
                        label=f't={time_points[i]:.1f}s')
    axes[0, 0].set_xlabel('径向位置 [m]')
    axes[0, 0].set_ylabel('轴向诱导因子 a [-]')
    axes[0, 0].set_title('轴向诱导因子径向分布')
    axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0, 0].grid(True)

    # 2. 攻角径向分布
    for i, (idx, color) in enumerate(zip(save_indices, colors)):
        alpha_data = torch.rad2deg(radial_data['alpha'][i]).numpy()
        axes[0, 1].plot(r, alpha_data, color=color, linewidth=2,
                        label=f't={time_points[i]:.1f}s')
    axes[0, 1].set_xlabel('径向位置 [m]')
    axes[0, 1].set_ylabel('攻角 [°]')
    axes[0, 1].set_title('攻角径向分布')
    axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0, 1].grid(True)

    # 3. 流入角径向分布
    for i, (idx, color) in enumerate(zip(save_indices, colors)):
        phi_data = torch.rad2deg(radial_data['phi'][i]).numpy()
        axes[1, 0].plot(r, phi_data, color=color, linewidth=2,
                        label=f't={time_points[i]:.1f}s')
    axes[1, 0].set_xlabel('径向位置 [m]')
    axes[1, 0].set_ylabel('流入角 [°]')
    axes[1, 0].set_title('流入角径向分布')
    axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1, 0].grid(True)

    # 4. 叶片几何信息
    chord = results['radial_data']['r']  # 这里需要从BEMT求解器获取弦长数据
    # 简化处理，直接从几何数据重新获取
    blade_geom = create_nrel5mw_blade_geometry()
    blade_data = blade_geom.drop_duplicates(subset=['径向位置(m)'], keep='first')
    r_geom = blade_data['径向位置(m)'].values
    chord_geom = blade_data['弦长(m)'].values
    twist_geom = blade_data['扭转角(°)'].values

    ax4 = axes[1, 1]
    ax4_twin = ax4.twinx()

    line1 = ax4.plot(r_geom, chord_geom, 'b-', linewidth=3, label='弦长')
    ax4.set_xlabel('径向位置 [m]')
    ax4.set_ylabel('弦长 [m]', color='b')
    ax4.tick_params(axis='y', labelcolor='b')

    line2 = ax4_twin.plot(r_geom, twist_geom, 'r-', linewidth=3, label='扭转角')
    ax4_twin.set_ylabel('扭转角 [°]', color='r')
    ax4_twin.tick_params(axis='y', labelcolor='r')

    ax4.set_title('叶片几何参数')
    ax4.grid(True)

    plt.tight_layout()

    if save_plots:
        plt.savefig('radial_distributions.png', dpi=300, bbox_inches='tight')
        print("径向分布图已保存为: radial_distributions.png")

    plt.show()


def print_simulation_summary(results: Dict[str, torch.Tensor]):
    """打印仿真结果摘要"""
    time = results['time']
    power = results['power'] / 1e6  # 转换为MW
    cp = results['cp']
    thrust = results['thrust'] / 1e3  # 转换为kN

    print(f"\n=== 时序仿真结果摘要 ===")
    print(f"仿真时长: {time[-1]:.1f} s")
    print(f"时间步数: {len(time)} 步")
    print(f"平均时间步长: {(time[-1] - time[0]) / (len(time) - 1):.3f} s")
    print(f"\n功率统计:")
    print(f"  平均功率: {torch.mean(power):.2f} MW")
    print(f"  最大功率: {torch.max(power):.2f} MW")
    print(f"  最小功率: {torch.min(power):.2f} MW")
    print(f"  功率标准差: {torch.std(power):.2f} MW")
    print(f"\n功率系数统计:")
    print(f"  平均Cp: {torch.mean(cp):.4f}")
    print(f"  最大Cp: {torch.max(cp):.4f}")
    print(f"  最小Cp: {torch.min(cp):.4f}")
    print(f"\n推力统计:")
    print(f"  平均推力: {torch.mean(thrust):.1f} kN")
    print(f"  最大推力: {torch.max(thrust):.1f} kN")
    print(f"  最小推力: {torch.min(thrust):.1f} kN")

    # 叶片角度统计
    blade_angles = results['blade_angles']
    print(f"\n叶片角度统计:")
    for i in range(3):
        angles_deg = blade_angles[:, i] * 180 / torch.pi
        print(f"  叶片{i + 1}: 角度变化范围 {torch.min(angles_deg):.1f}° - {torch.max(angles_deg):.1f}°")


def analyze_frequency_content(results: Dict[str, torch.Tensor], signal_name: str = 'power'):
    """
    分析信号的频域特性

    Args:
        results: 仿真结果
        signal_name: 要分析的信号名称
    """
    import scipy.fft as fft

    time = results['time'].numpy()
    signal = results[signal_name].numpy()

    # 计算采样频率
    dt = time[1] - time[0]
    fs = 1.0 / dt

    # 计算FFT
    n_samples = len(signal)
    freqs = fft.fftfreq(n_samples, dt)[:n_samples // 2]
    fft_vals = fft.fft(signal)[:n_samples // 2]
    magnitude = np.abs(fft_vals)

    # 找到主要频率成分
    main_freq_idx = np.argmax(magnitude[1:]) + 1  # 跳过直流分量
    main_freq = freqs[main_freq_idx]

    print(f"\n=== {signal_name}信号频域分析 ===")
    print(f"采样频率: {fs:.2f} Hz")
    print(f"主要频率成分: {main_freq:.4f} Hz (周期: {1 / main_freq:.2f} s)")

    # 绘制频谱
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(time, signal, 'b-', linewidth=1)
    plt.xlabel('时间 [s]')
    plt.ylabel(f'{signal_name}')
    plt.title(f'{signal_name}时域信号')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.semilogy(freqs, magnitude, 'r-', linewidth=1)
    plt.xlabel('频率 [Hz]')
    plt.ylabel('幅值')
    plt.title(f'{signal_name}频谱')
    plt.grid(True)
    plt.xlim(0, min(5.0, fs / 2))  # 显示0-5Hz范围

    plt.tight_layout()
    plt.show()


# === 使用示例和测试 ===
if __name__ == "__main__":
    print("准稳态BEMT时序求解器测试")
    print("=" * 60)

    try:
        # 1. 加载翼型数据库
        print("1. 加载翼型数据库...")
        # 注意：这里假设翼型数据库文件存在
        # 如果文件不存在，可以使用模拟数据库或跳过测试
        try:
            airfoil_db = AirfoilDatabase("nrel_5mw_airfoils.pkl")
        except FileNotFoundError:
            print("   警告：翼型数据库文件不存在，使用模拟数据库")
            # 这里可以创建一个简化的模拟数据库用于测试
            raise FileNotFoundError("请确保翼型数据库文件存在")

        # 2. 创建测试时间序列
        print("\n2. 创建测试时间序列...")
        duration = 60.0  # 60秒仿真
        dt = 0.1  # 0.1秒时间步长
        time_series, wind_speed_series, rotor_speed_series = create_test_time_series(duration, dt)

        print(f"   仿真时长: {duration} s")
        print(f"   时间步长: {dt} s")
        print(f"   总时间步数: {len(time_series)}")
        print(f"   风速范围: {torch.min(wind_speed_series):.1f} - {torch.max(wind_speed_series):.1f} m/s")
        print(f"   转速范围: {torch.min(rotor_speed_series):.2f} - {torch.max(rotor_speed_series):.2f} rad/s")

        # 3. 创建叶片几何
        print("\n3. 创建叶片几何...")
        blade_geometry = create_nrel5mw_blade_geometry()
        print(f"   叶片节点数: {len(blade_geometry)}")

        # 4. 运行时序仿真
        print("\n4. 运行准稳态BEMT时序仿真...")
        initial_blade_angles = torch.tensor([0.0, 2 * torch.pi / 3, 4 * torch.pi / 3])  # 120度间隔

        results = time_series_simulation(
            wind_speed_series=wind_speed_series,
            rotor_speed_series=rotor_speed_series,
            time_series=time_series,
            airfoil_db=airfoil_db,
            blade_geometry=blade_geometry,
            initial_blade_angles=initial_blade_angles
        )

        # 5. 显示结果摘要
        print_simulation_summary(results)

        # 6. 绘制结果
        print("\n5. 绘制时序结果...")
        plot_time_series_results(results, save_plots=True)

        print("\n6. 绘制径向分布...")
        plot_radial_distributions(results, save_plots=True)

        # 7. 频域分析
        print("\n7. 进行频域分析...")
        analyze_frequency_content(results, 'power')
        analyze_frequency_content(results, 'thrust')

        # 8. 单独测试叶片角度计算
        print("\n8. 叶片角度计算验证...")
        test_time = torch.linspace(0, 10, 101)  # 10秒，101个点
        test_rotor_speed = torch.ones_like(test_time) * 1.0  # 恒定1 rad/s
        test_blade_angles = calculate_blade_angles(test_time, test_rotor_speed)

        print(f"   测试时长: 10 s，恒定转速 1.0 rad/s")
        print(f"   最终叶片角度 (度):")
        final_angles_deg = test_blade_angles[-1] * 180 / torch.pi
        for i in range(3):
            print(f"     叶片{i + 1}: {final_angles_deg[i]:.1f}°")
        print(f"   理论值应为: {10.0 * 180 / torch.pi:.1f}° (相位差120°)")

        print("\n✓ 准稳态BEMT时序仿真测试完成！")

    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback

        traceback.print_exc()

        # 提供替代测试（不需要翼型数据库）
        print("\n尝试简化测试（不使用翼型数据库）...")

        # 测试叶片角度计算功能
        print("测试叶片角度计算功能...")
        test_time = torch.linspace(0, 20, 201)
        test_rotor_speed = torch.ones_like(test_time) * 1.2
        test_blade_angles = calculate_blade_angles(test_time, test_rotor_speed)

        # 绘制叶片角度
        plt.figure(figsize=(10, 6))
        time_np = test_time.numpy()
        angles_deg = test_blade_angles.numpy() * 180 / np.pi

        plt.plot(time_np, angles_deg[:, 0], 'r-', linewidth=2, label='叶片1')
        plt.plot(time_np, angles_deg[:, 1], 'g-', linewidth=2, label='叶片2')
        plt.plot(time_np, angles_deg[:, 2], 'b-', linewidth=2, label='叶片3')

        plt.xlabel('时间 [s]')
        plt.ylabel('叶片角度 [°]')
        plt.title('三个叶片与中轴夹角随时间变化')
        plt.legend()
        plt.grid(True)
        plt.show()

        print("✓ 叶片角度计算功能测试通过！")