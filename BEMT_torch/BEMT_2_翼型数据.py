"""
集成翼型数据的BEMT实现 - 全PyTorch版本
BEMT Implementation with Real Airfoil Data - Full PyTorch Version
"""

import torch
import pickle
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional


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


def bemt_with_real_airfoils(wind_speed: torch.Tensor,
                            rotor_speed: torch.Tensor,
                            airfoil_db: AirfoilDatabase,
                            blade_geometry: Optional[pd.DataFrame] = None) -> Dict[str, torch.Tensor]:
    """
    使用真实翼型数据的叶素动量理论(BEMT)求解器 - 全PyTorch版本

    Args:
        wind_speed: 风速 [m/s] - torch.Tensor
        rotor_speed: 转子角速度 [rad/s] - torch.Tensor
        airfoil_db: 翼型数据库对象
        blade_geometry: 叶片几何数据DataFrame

    Returns:
        BEMT求解结果字典
    """
    # === 1. 基本参数定义 ===
    R = 61.5  # 转子半径 [m] - NREL 5MW标准值
    B = 3  # 叶片数量
    rho = 1.225  # 空气密度 [kg/m³] - 海平面标准大气条件

    # === 2. 叶片几何数据处理 ===
    if blade_geometry is None:
        # 如果没有提供几何数据，使用默认的NREL 5MW几何
        blade_geometry = create_nrel5mw_blade_geometry()

    # 移除重复的最后一个点（径向位置61.500重复）
    blade_data = blade_geometry.drop_duplicates(subset=['径向位置(m)'], keep='first').copy().reset_index(drop=True)

    n_stations = len(blade_data)  # 计算站点数量
    print(f"计算站点数: {n_stations}")

    # 转换为PyTorch张量
    r = torch.tensor(blade_data['径向位置(m)'].values, dtype=torch.float32)  # 径向位置
    chord = torch.tensor(blade_data['弦长(m)'].values, dtype=torch.float32)  # 弦长分布
    twist = torch.deg2rad(torch.tensor(blade_data['扭转角(°)'].values, dtype=torch.float32))  # 扭转角（转换为弧度）
    airfoil_ids = blade_data['翼型ID'].values  # 翼型ID数组

    # === 3. 诱导因子初始化 ===
    a = torch.zeros_like(r)  # 轴向诱导因子初始化为0
    a_prime = torch.zeros_like(r)  # 切向诱导因子初始化为0

    # === 4. BEMT迭代求解过程 ===
    max_iterations = 50  # 最大迭代次数
    tolerance = 1e-6  # 收敛容差

    print("开始BEMT迭代求解...")

    for iteration in range(max_iterations):
        # 步骤4.1: 计算诱导速度
        V_axial = wind_speed * (1 - a)  # 轴向诱导速度
        V_tangential = rotor_speed * r * (1 + a_prime)  # 切向诱导速度

        # 步骤4.2: 计算相对风速和流入角
        V_rel = torch.sqrt(V_axial ** 2 + V_tangential ** 2)  # 相对风速大小
        phi = torch.atan2(V_axial, V_tangential)  # 流入角

        # 步骤4.3: 计算攻角
        alpha = phi - twist  # 攻角 = 流入角 - 扭转角

        # 步骤4.4: 从翼型数据库获取气动系数
        cl = torch.zeros_like(alpha)  # 升力系数数组
        cd = torch.zeros_like(alpha)  # 阻力系数数组
        cm = torch.zeros_like(alpha)  # 力矩系数数组

        # 对每个径向站点查询对应翼型的气动系数
        for i, airfoil_id in enumerate(airfoil_ids):
            alpha_deg = torch.rad2deg(alpha[i])  # 转换为度

            # 从翼型数据库获取气动系数
            cl_val, cd_val, cm_val = airfoil_db.get_aero_coefficients(airfoil_id, alpha_deg)

            # 存储气动系数
            cl[i] = cl_val
            cd[i] = cd_val
            cm[i] = cm_val

        # 步骤4.5: 计算法向和切向载荷系数
        cn = cl * torch.cos(phi) + cd * torch.sin(phi)  # 法向载荷系数
        ct = cl * torch.sin(phi) - cd * torch.cos(phi)  # 切向载荷系数

        # 步骤4.6: 计算实度（局部实度）
        r_min = 0.1  # 设置最小径向位置阈值
        r_calc = torch.clamp(r, min=r_min)  # 限制最小值
        sigma = B * chord / (2 * torch.pi * r_calc)
        sigma[0] = 0
        sigma[1] = 0

        # 步骤4.7: 更新轴向诱导因子
        # 根据动量理论计算轴向诱导因子
        k_ax = sigma * cn / (4 * torch.sin(phi) ** 2)
        a_new = k_ax / (1 + k_ax)

        # Glauert修正：处理大诱导因子情况（避免动量理论失效）
        a_critical = 0.4  # 临界诱导因子
        high_a_mask = a_new > a_critical
        if torch.any(high_a_mask):
            # 使用Glauert经验修正公式
            f_glauert = 0.25 * (5 - 3 * a_new[high_a_mask])
            a_new[high_a_mask] = 1 - f_glauert

        # 步骤4.8: 更新切向诱导因子
        k_tan = sigma * ct / (4 * torch.sin(phi) * torch.cos(phi))
        a_prime_new = k_tan / (1 - k_tan)

        # 步骤4.9: 松弛更新诱导因子（提高数值稳定性）
        relaxation = 0.3  # 松弛因子
        a_old = a.clone()  # 保存旧值用于收敛判断
        a = (1 - relaxation) * a + relaxation * a_new  # 松弛更新轴向诱导因子
        a_prime = (1 - relaxation) * a_prime + relaxation * a_prime_new  # 松弛更新切向诱导因子

        # 步骤4.10: 检查收敛性
        residual = torch.max(torch.abs(a - a_old))  # 计算最大残差

        # 每10次迭代打印一次进度
        if iteration % 10 == 0:
            print(f"  迭代 {iteration:2d}: 残差 = {residual:.2e}")

        # 检查是否收敛
        if residual < tolerance:
            print(f"  收敛于第 {iteration} 次迭代，残差 = {residual:.2e}")
            break

    # === 5. 最终结果计算 ===
    # 重新计算最终的流动参数（使用收敛后的诱导因子）
    V_axial = wind_speed * (1 - a)
    V_tangential = rotor_speed * r * (1 + a_prime)
    V_rel = torch.sqrt(V_axial ** 2 + V_tangential ** 2)
    phi = torch.atan2(V_axial, V_tangential)

    # 计算径向微元长度（用于积分）
    if len(r) > 1:
        dr = torch.zeros_like(r)
        dr[1:-1] = (r[2:] - r[:-2]) / 2  # 中间点使用中心差分
        dr[0] = r[1] - r[0]  # 第一个点
        dr[-1] = r[-1] - r[-2]  # 最后一个点
    else:
        dr = torch.ones_like(r)  # 单点情况

    # 计算单位长度载荷
    # 推力单位长度载荷 [N/m]
    dT_dr = 0.5 * rho * V_rel ** 2 * chord * cn * B
    # 扭矩单位长度载荷 [N⋅m/m]
    dQ_dr = 0.5 * rho * V_rel ** 2 * chord * ct * r * B

    # 沿叶片积分获得总载荷
    thrust = torch.sum(dT_dr * dr)  # 总推力 [N]
    torque = torch.sum(dQ_dr * dr)  # 总扭矩 [N⋅m]
    power = torque * rotor_speed  # 功率 [W]

    # 计算无量纲系数
    A = torch.pi * R ** 2  # 扫掠面积 [m²]
    cp = power / (0.5 * rho * A * wind_speed ** 3)  # 功率系数
    ct_total = thrust / (0.5 * rho * A * wind_speed ** 2)  # 推力系数

    # 返回完整的计算结果
    return {
        'power': power,  # 功率
        'thrust': thrust,  # 推力
        'torque': torque,  # 扭矩
        'cp': cp,  # 功率系数
        'ct': ct_total,  # 推力系数
        'a': a,  # 轴向诱导因子分布
        'a_prime': a_prime,  # 切向诱导因子分布
        'alpha': alpha,  # 攻角分布
        'phi': phi,  # 流入角分布
        'cl': cl,  # 升力系数分布
        'cd': cd,  # 阻力系数分布
        'cm': cm,  # 力矩系数分布
        'r': r,  # 径向位置
        'chord': chord,  # 弦长分布
        'twist': twist,  # 扭转角分布
        'airfoil_ids': torch.tensor(airfoil_ids, dtype=torch.int32),  # 翼型ID分布
        'loads': {'dT_dr': dT_dr, 'dQ_dr': dQ_dr, 'dr': dr},  # 载荷分布
        'flow': {'V_rel': V_rel, 'V_axial': V_axial, 'V_tangential': V_tangential}  # 流动参数
    }


def print_results(result: Dict[str, torch.Tensor], wind_speed: torch.Tensor, rotor_speed: torch.Tensor):
    """打印BEMT计算结果摘要"""
    print(f"\n=== BEMT 计算结果 ===")
    print(f"风速: {wind_speed:.1f} m/s")
    print(f"转速: {rotor_speed:.3f} rad/s ({rotor_speed * 30 / torch.pi:.1f} RPM)")
    print(f"功率: {result['power'] / 1e6:.2f} MW")
    print(f"推力: {result['thrust'] / 1e3:.1f} kN")
    print(f"功率系数 Cp: {result['cp']:.4f}")
    print(f"推力系数 Ct: {result['ct']:.4f}")
    print(f"叶尖速比 TSR: {rotor_speed * 61.5 / wind_speed:.2f}")


def parametric_study(airfoil_db: AirfoilDatabase,
                     blade_geometry: pd.DataFrame,
                     wind_speeds: torch.Tensor,
                     rotor_speeds: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    参数化研究：计算不同风速和转速组合的性能

    Args:
        airfoil_db: 翼型数据库
        blade_geometry: 叶片几何
        wind_speeds: 风速张量
        rotor_speeds: 转速张量

    Returns:
        包含功率、功率系数矩阵的字典
    """
    n_ws = len(wind_speeds)
    n_rs = len(rotor_speeds)

    # 初始化结果张量
    power_matrix = torch.zeros(n_ws, n_rs)
    cp_matrix = torch.zeros(n_ws, n_rs)
    ct_matrix = torch.zeros(n_ws, n_rs)

    print(f"开始参数化研究: {n_ws} × {n_rs} = {n_ws * n_rs} 个工况")

    # 双重循环计算所有工况
    for i, ws in enumerate(wind_speeds):
        for j, rs in enumerate(rotor_speeds):
            try:
                # 调用BEMT求解器
                result = bemt_with_real_airfoils(ws, rs, airfoil_db, blade_geometry)
                # 存储结果
                power_matrix[i, j] = result['power'] / 1e6  # 转换为MW
                cp_matrix[i, j] = result['cp']
                ct_matrix[i, j] = result['ct']
            except Exception as e:
                print(f"工况 (ws={ws:.1f}, rs={rs:.3f}) 计算失败: {e}")
                # 如果计算失败，设为0
                power_matrix[i, j] = 0
                cp_matrix[i, j] = 0
                ct_matrix[i, j] = 0

        print(f"  风速 {ws:.1f} m/s 完成")

    return {
        'power_matrix': power_matrix,
        'cp_matrix': cp_matrix,
        'ct_matrix': ct_matrix,
        'wind_speeds': wind_speeds,
        'rotor_speeds': rotor_speeds
    }


def find_optimal_operating_point(parametric_results: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """寻找最优工作点（最大功率系数）"""
    cp_matrix = parametric_results['cp_matrix']
    wind_speeds = parametric_results['wind_speeds']
    rotor_speeds = parametric_results['rotor_speeds']

    # 找到最大功率系数的索引
    max_cp_idx = torch.argmax(cp_matrix)
    max_cp_i, max_cp_j = divmod(max_cp_idx.item(), len(rotor_speeds))

    optimal_ws = wind_speeds[max_cp_i]
    optimal_rs = rotor_speeds[max_cp_j]
    optimal_cp = cp_matrix[max_cp_i, max_cp_j]
    optimal_power = parametric_results['power_matrix'][max_cp_i, max_cp_j]
    optimal_tsr = optimal_rs * 61.5 / optimal_ws

    return {
        'wind_speed': optimal_ws,
        'rotor_speed': optimal_rs,
        'cp': optimal_cp,
        'power': optimal_power,
        'tsr': optimal_tsr
    }


# === 使用示例 ===
if __name__ == "__main__":
    print("BEMT求解器 - 全PyTorch版本测试")
    print("=" * 50)

    try:
        # 1. 加载翼型数据库（如果失败会直接报错退出）
        print("1. 加载翼型数据库...")
        airfoil_db = AirfoilDatabase("nrel_5mw_airfoils.pkl")

        # 2. 创建叶片几何
        print("\n2. 创建叶片几何...")
        blade_geometry = create_nrel5mw_blade_geometry()
        print(f"叶片几何数据: {len(blade_geometry)} 个节点")

        # 3. 单点BEMT计算
        print("\n3. 进行单点BEMT计算...")
        wind_speed = torch.tensor(11.0, requires_grad=True)  # 启用自动微分
        rotor_speed = torch.tensor(1.2, requires_grad=True)  # 启用自动微分

        result = bemt_with_real_airfoils(
            wind_speed=wind_speed,
            rotor_speed=rotor_speed,
            airfoil_db=airfoil_db,
            blade_geometry=blade_geometry
        )

        # 4. 显示结果
        print_results(result, wind_speed, rotor_speed)

        # 5. 自动微分演示
        print("\n5. 自动微分演示...")
        power = result['power']

        # 计算梯度
        power.backward()

        print(f"功率对风速的梯度: {wind_speed.grad:.2f} W/(m/s)")
        print(f"功率对转速的梯度: {rotor_speed.grad:.2f} W/(rad/s)")

        '''
        # 6. 参数化研究
        print("\n6. 参数化研究...")
        wind_speeds = torch.linspace(6, 14, 5)  # 6-14 m/s, 5个点
        rotor_speeds = torch.linspace(0.8, 1.6, 5)  # 0.8-1.6 rad/s, 5个点

        parametric_results = parametric_study(airfoil_db, blade_geometry, wind_speeds, rotor_speeds)

        # 7. 寻找最优工作点
        optimal_point = find_optimal_operating_point(parametric_results)

        print(f"\n最优工作点:")
        print(f"  风速: {optimal_point['wind_speed']:.1f} m/s")
        print(f"  转速: {optimal_point['rotor_speed']:.3f} rad/s")
        print(f"  功率: {optimal_point['power']:.2f} MW")
        print(f"  功率系数: {optimal_point['cp']:.4f}")
        print(f"  叶尖速比: {optimal_point['tsr']:.2f}")

        print("\n✓ 所有计算完成！")
        
        '''

    except Exception as e:
        print(f"\n✗ 程序执行失败: {e}")
        raise