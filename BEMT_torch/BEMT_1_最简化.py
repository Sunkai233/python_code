"""
极简 BEMT 实现 - 纯 PyTorch 版本
Simple BEMT Implementation using Pure PyTorch
"""

import torch
import numpy as np
import matplotlib.pyplot as plt


def simple_bemt_pytorch(wind_speed, rotor_speed, n_stations=10):
    """
    极简 BEMT 求解器 - 纯 PyTorch 实现
    Simple BEMT solver - Pure PyTorch implementation
    """
    # === 1. 基本参数 / Basic Parameters ===
    R = 50.0  # 转子半径 [m] / Rotor radius [m]
    B = 3  # 叶片数 / Number of blades

    # === 2. 径向站点 / Radial stations ===
    r = torch.linspace(0.2 * R, R, n_stations)  # 径向位置 / Radial positions
    r_R = r / R  # 无量纲半径 / Non-dimensional radius

    # === 3. 叶片几何 (简化) / Blade geometry (simplified) ===
    chord = 3.0 - 2.0 * r_R  # 弦长分布 / Chord distribution
    twist = torch.deg2rad(15.0 * (1 - r_R))  # 扭转角 [rad] / Twist angle [rad]

    # === 4. 初始猜值 / Initial guess ===
    a = torch.zeros_like(r_R)  # 轴向诱导因子 / Axial induction factor
    a_prime = torch.zeros_like(r_R)  # 切向诱导因子 / Tangential induction factor

    # === 5. BEMT 迭代求解 / BEMT iterative solution ===
    for i in range(20):  # 最大20次迭代 / Max 20 iterations
        # 计算诱导速度 / Calculate induced velocities
        V_axial = wind_speed * (1 - a)
        V_tangential = rotor_speed * r * (1 + a_prime)

        # 相对风速和流入角 / Relative wind speed and inflow angle
        V_rel = torch.sqrt(V_axial ** 2 + V_tangential ** 2)
        phi = torch.atan2(V_axial, V_tangential)

        # 攻角 / Angle of attack
        alpha = phi - twist

        # 简化翼型特性 / Simplified airfoil characteristics
        cl = 2 * torch.pi * alpha  # 线性升力系数 / Linear lift coefficient
        cd = 0.01 + 0.1 * alpha ** 2  # 简化阻力系数 / Simplified drag coefficient

        # 法向和切向载荷系数 / Normal and tangential load coefficients
        cn = cl * torch.cos(phi) + cd * torch.sin(phi)
        ct = cl * torch.sin(phi) - cd * torch.cos(phi)

        # 实度 / Solidity
        #sigma = B * chord / (2 * torch.pi * r)

        # 对极小的径向位置进行特殊处理
        r_min = 0.1  # 设置最小径向位置阈值
        r_calc = torch.clamp(r, min=r_min)  # 限制最小值
        sigma = B * chord / (2 * torch.pi * r_calc)
        sigma[0] = 0
        # 更新诱导因子 / Update induction factors
        # 轴向诱导因子 / Axial induction factor
        k_ax = sigma * cn / (4 * torch.sin(phi) ** 2)
        a_new = k_ax / (1 + k_ax)

        # 切向诱导因子 / Tangential induction factor
        k_tan = sigma * ct / (4 * torch.sin(phi) * torch.cos(phi))
        a_prime_new = k_tan / (1 - k_tan)

        # 松弛更新 / Relaxed update
        a = 0.7 * a + 0.3 * a_new
        a_prime = 0.7 * a_prime + 0.3 * a_prime_new

    # === 6. 计算载荷和功率 / Calculate loads and power ===
    rho = 1.225  # 空气密度 / Air density
    dr = (R - 0.2 * R) / (n_stations - 1)  # 径向微元 / Radial element

    # 单位长度载荷 / Loads per unit length
    dT_dr = 0.5 * rho * V_rel ** 2 * chord * cn * B  # 推力 / Thrust
    dQ_dr = 0.5 * rho * V_rel ** 2 * chord * ct * r * B  # 扭矩 / Torque

    # 积分获得总载荷 / Integrate for total loads
    thrust = torch.sum(dT_dr * dr)
    torque = torch.sum(dQ_dr * dr)
    power = torque * rotor_speed

    return {
        'power': power,
        'thrust': thrust,
        'torque': torque,
        'a': a,
        'a_prime': a_prime,
        'alpha': alpha,
        'phi': phi,
        'r': r,
        'loads': {'dT_dr': dT_dr, 'dQ_dr': dQ_dr}
    }


# === 使用示例 / Usage Example ===
if __name__ == "__main__":
    print("PyTorch 简单 BEMT 求解器测试 / PyTorch Simple BEMT Solver Test")

    # 定义工况 / Define conditions
    U = torch.tensor(10.0, requires_grad=True)  # 风速 [m/s] / Wind speed
    Omega = torch.tensor(1.0, requires_grad=True)  # 转速 [rad/s] / Rotor speed

    # 求解 BEMT / Solve BEMT
    result = simple_bemt_pytorch(U, Omega)

    print(f"功率 / Power: {result['power']:.0f} W")
    print(f"推力 / Thrust: {result['thrust']:.0f} N")
    print(f"扭矩 / Torque: {result['torque']:.0f} N⋅m")

    # === 自动微分演示 / Automatic Differentiation Demo ===
    print("\n=== 自动微分演示 / Automatic Differentiation Demo ===")

    # 计算梯度 / Calculate gradients
    power = result['power']

    # 反向传播 / Backward propagation
    power.backward()

    print(f"功率对风速的梯度 / Power gradient w.r.t. wind speed: {U.grad:.2f} W/(m/s)")
    print(f"功率对转速的梯度 / Power gradient w.r.t. rotor speed: {Omega.grad:.2f} W/(rad/s)")

    # === 优化示例 / Optimization Example ===
    print("\n=== 优化示例 / Optimization Example ===")


    def find_optimal_rotor_speed(wind_speed, target_power=500000):
        """
        寻找达到目标功率的最优转速
        Find optimal rotor speed for target power
        """
        # 创建可优化的转速参数 / Create optimizable rotor speed parameter
        omega = torch.tensor(1.0, requires_grad=True)
        optimizer = torch.optim.Adam([omega], lr=0.01)

        print(f"目标功率 / Target power: {target_power:.0f} W")

        for i in range(100):
            optimizer.zero_grad()

            # 计算当前功率 / Calculate current power
            result = simple_bemt_pytorch(wind_speed, omega)
            current_power = result['power']

            # 定义损失函数 / Define loss function
            loss = (current_power - target_power) ** 2

            # 反向传播 / Backward propagation
            loss.backward()
            optimizer.step()

            if i % 20 == 0:
                print(f"迭代 {i:2d}: 转速 = {omega.item():.3f} rad/s, "
                      f"功率 = {current_power.item():.0f} W, "
                      f"误差 = {torch.sqrt(loss).item():.0f} W")

            # 检查收敛 / Check convergence
            if torch.sqrt(loss).item() < 1000:  # 1kW 误差容差 / 1kW error tolerance
                print(f"优化收敛! / Optimization converged!")
                break

        return omega.item(), current_power.item()


    # 寻找产生 500kW 功率的最优转速 / Find optimal speed for 500kW power
    optimal_speed, final_power = find_optimal_rotor_speed(10.0, target_power=500000)

    print(f"\n优化结果 / Optimization Result:")
    print(f"最优转速 / Optimal rotor speed: {optimal_speed:.3f} rad/s")
    print(f"实现功率 / Achieved power: {final_power:.0f} W")

    # === 参数化研究 / Parametric Study ===
    print("\n=== 参数化研究 / Parametric Study ===")

    wind_speeds = torch.linspace(5, 15, 11)
    rotor_speeds = torch.linspace(0.5, 2.0, 16)

    power_matrix = torch.zeros(len(wind_speeds), len(rotor_speeds))
    cp_matrix = torch.zeros(len(wind_speeds), len(rotor_speeds))

    print("正在进行参数化研究... / Conducting parametric study...")

    for i, u in enumerate(wind_speeds):
        for j, omega in enumerate(rotor_speeds):
            result = simple_bemt_pytorch(u, omega)
            power_matrix[i, j] = result['power']

            # 计算功率系数 / Calculate power coefficient
            rho = 1.225
            A = torch.pi * 50.0 ** 2  # 扫掠面积 / Swept area
            cp = result['power'] / (0.5 * rho * A * u ** 3)
            cp_matrix[i, j] = cp

    # 找到最大功率系数点 / Find maximum power coefficient point
    max_cp_idx = torch.argmax(cp_matrix)
    max_cp_i, max_cp_j = divmod(max_cp_idx.item(), len(rotor_speeds))
    max_cp = cp_matrix[max_cp_i, max_cp_j]
    optimal_wind_speed = wind_speeds[max_cp_i]
    optimal_rotor_speed = rotor_speeds[max_cp_j]
    optimal_tsr = optimal_rotor_speed * 50.0 / optimal_wind_speed

    print(f"\n参数化研究结果 / Parametric Study Results:")
    print(f"最大功率系数 / Maximum Cp: {max_cp:.4f}")
    print(f"最优风速 / Optimal wind speed: {optimal_wind_speed:.1f} m/s")
    print(f"最优转速 / Optimal rotor speed: {optimal_rotor_speed:.3f} rad/s")
    print(f"最优叶尖速比 / Optimal TSR: {optimal_tsr:.2f}")

