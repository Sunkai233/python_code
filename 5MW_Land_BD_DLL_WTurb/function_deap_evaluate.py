import os
import pickle
import numpy as np
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
from openfast_toolbox.io import FASTInputFile
from openfast_toolbox.case_generation import runner
from openfast_toolbox.io import FASTOutputFile  # 用于读取 OpenFAST 二进制输出文件
from matplotlib import rcParams

# 设置中文字体（任选一种）
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Zen Hei', 'Arial Unicode MS']
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


class FASTInputHandler:
    def __init__(self, current_dir, parent_dir):
        self.current_dir = current_dir
        self.parent_dir = parent_dir

        # 初始化文件路径
        self.aero_file = os.path.join(current_dir, "NRELOffshrBsline5MW_Onshore_AeroDyn.dat")
        self.blade_file = os.path.join(parent_dir, "5MW_Baseline", "NRELOffshrBsline5MW_AeroDyn_blade.dat")
        self.beamdyn_file = os.path.join(parent_dir, "5MW_Baseline", "NRELOffshrBsline5MW_BeamDyn_Blade.dat")
        self.elastodyn_file = os.path.join(current_dir, "NRELOffshrBsline5MW_Onshore_ElastoDyn_BDoutputs.dat")

        # 加载 FAST 输入文件
        self.aero = FASTInputFile(self.aero_file)
        self.blade = FASTInputFile(self.blade_file)
        self.beamdyn = FASTInputFile(self.beamdyn_file)
        self.elastodyn = FASTInputFile(self.elastodyn_file)

    def load_values_dict(self, pickle_file):
        """从 pickle 文件加载变量字典"""
        with open(pickle_file, "rb") as f:
            return pickle.load(f)

    def generate_variable_values(self, values_dict: dict, variable_flags: dict) -> dict:
        """根据字典和标志计算变量值"""
        result = {}
        for key_v, flag in variable_flags.items():
            key = key_v[:-2]
            if key in values_dict:
                value = values_dict[key]
                if isinstance(value, list):
                    if all(not isinstance(i, (list, tuple)) for i in value):
                        result[key_v] = [i * flag for i in value]
                    else:
                        result[key_v] = (np.array(value) * flag).tolist()
                else:
                    result[key_v] = value * flag
            else:
                result[key_v] = None
        return result

    def apply_variables_to_fast_inputs(self, variable_values: dict):
        """将计算得到的变量值应用到 FAST 输入文件"""
        for key_v, value in variable_values.items():
            key = key_v[:-2]

            if key in ['AirDens', 'KinVisc', 'Patm', 'Pvap']:
                self.aero[key] = value
            elif key == 'TwrCd' and 'TowProp' in self.aero.keys():
                TowProp = np.array(self.aero['TowProp'])
                if TowProp.shape[1] >= 3 and len(value) == TowProp.shape[0]:
                    TowProp[:, 2] = value
                    self.aero['TowProp'] = TowProp.tolist()
            elif key in ['BlCrvAng', 'BlTwist']:
                col_index = 3 if key == 'BlCrvAng' else 4
                BldAeroNodes = np.array(self.blade['BldAeroNodes'])
                if len(value) == BldAeroNodes.shape[0]:
                    BldAeroNodes[:, col_index] = value
                    self.blade['BldAeroNodes'] = BldAeroNodes.tolist()
            elif key == 'DampingCoeffs':
                new_vals = np.array(value)
                old_vals = self.beamdyn['DampingCoeffs']
                if np.shape(old_vals) == np.shape(new_vals):
                    for i in range(len(old_vals)):
                        for j in range(len(old_vals[0])):
                            self.beamdyn['DampingCoeffs'][i][j] = new_vals[i][j]
            elif key == 'K':
                self.beamdyn['BeamProperties']['K'] = value
            elif key == 'M':
                self.beamdyn['BeamProperties']['M'] = value
            elif key in self.elastodyn.keys():
                self.elastodyn[key] = value
            else:
                print(f"⚠️ 未识别或未处理的变量：{key}")

        # 修正 blade 数组
        if isinstance(self.blade['BldAeroNodes'], list):
            self.blade['BldAeroNodes'] = np.array(self.blade['BldAeroNodes'])
        self.blade['NumBlNds'] = self.blade['BldAeroNodes'].shape[0]

    def write_files(self):
        """将修改后的数据写回到文件"""
        self.aero.write(self.aero_file)
        # self.blade.write(self.blade_file)  # 根据需要选择是否覆盖
        self.beamdyn.write(self.beamdyn_file)
        self.elastodyn.write(self.elastodyn_file)


class FASTSimulator:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.fastExe = os.path.join(base_dir, "openfast_x64.exe")
        self.fst_file = os.path.join(base_dir, "5MW_Land_BD_DLL_WTurb.fst")

    def run_simulation(self):
        """调用 OpenFAST 执行仿真"""
        process = runner.run_fast(
            input_file=self.fst_file,
            fastExe=self.fastExe,
            wait=True,
            showOutputs=True,
            showCommand=True
        )
        print("✅ 仿真运行完毕！")

class FASTRunner:
    def __init__(self, current_dir, parent_dir, pickle_file):
        self.input_handler = FASTInputHandler(current_dir, parent_dir)
        self.simulator = FASTSimulator(current_dir)

        # 加载并更新变量字典
        self.values_dict = self.input_handler.load_values_dict(pickle_file)
        self.values_dict.update({
            'AirDens': 1.1225,
            'KinVisc': 1.5e-5,
            'Patm': 101325,
            'Pvap': 0
        })

        # 示例 flags（可外部传入）
        self.variable_flags = {
            'AirDens_v': 1.1,
            'KinVisc_v': 1.1,
            'Patm_v': 1.1,
            'Pvap_v': 1.1,
            'TwrCd_v': 1.1,
            'BlCrvAng_v': 1.1,
            'BlTwist_v': 1.1,
            'DampingCoeffs_v': 1.1,
            'K_v': 1.1,
            'M_v': 1.1,
            'GBoxEff_v': 1,
            'DTTorSpr_v': 1.1,
            'DTTorDmp_v': 1.1,
            'TeetDmp_v': 1.1,
            'TeetCDmp_v': 1,
            'M_CSmax_v': 1,
            'M_CD_v': 1,
            'sig_v_v': 1,
            'sig_v2_v': 1
        }

    def execute(self):
        """执行文件修改与仿真"""
        # 计算变量值
        variable_values = self.input_handler.generate_variable_values(self.values_dict, self.variable_flags)

        # 应用变量并写入文件
        self.input_handler.apply_variables_to_fast_inputs(variable_values)
        self.input_handler.write_files()

        # 运行 OpenFAST 仿真
        self.simulator.run_simulation()

    def process_output_data(self):
        """提取 OpenFAST 输出数据并进行处理，返回总误差值"""
        # 加载并处理原始数据
        fastoutFilename = "5MW_Land_BD_DLL_WTurb.out"
        df = FASTOutputFile(fastoutFilename).toDataFrame()

        # 提取数据列
        time = df['Time_[s]'].values
        rot_speed = df['RotSpeed_[rpm]'].values
        gen_speed = df['GenSpeed_[rpm]'].values
        gen_power = df['GenPwr_[kW]'].values

        # 数据降采样
        target_rows = 30
        if len(time) > target_rows:
            group_size = len(time) // target_rows

            def downsample(data):
                return [np.mean(data[i * group_size:(i + 1) * group_size]) for i in range(target_rows)]

            time = downsample(time)
            rot_speed = downsample(rot_speed)
            gen_speed = downsample(gen_speed)
            gen_power = downsample(gen_power)

        # 归一化函数
        def normalize(data):
            min_val = np.min(data)
            max_val = np.max(data)
            return (data - min_val) / (max_val - min_val + 1e-10)  # 添加小量防止除以零

        # 保存原始数据
        original_data = np.column_stack((time, rot_speed, gen_speed, gen_power))
        np.save('turbine_data.npy', original_data)

        # 归一化原始数据
        norm_original = np.column_stack((
            normalize(time),
            normalize(rot_speed),
            normalize(gen_speed),
            normalize(gen_power)
        ))
        np.save('normalized_turbine_data.npy', norm_original)

        # 初始化总误差为None
        total_error = None

        # 计算调整后数据的误差
        try:
            adjusted_data = np.load('adjusted_turbine_data.npy')
            norm_adjusted = np.column_stack((
                normalize(adjusted_data[:, 0]),
                normalize(adjusted_data[:, 1]),
                normalize(adjusted_data[:, 2]),
                normalize(adjusted_data[:, 3])
            ))

            error_matrix = np.abs(norm_original - norm_adjusted)
            total_error = np.sum(error_matrix)

            np.save('error_matrix.npy', error_matrix)
            print(f"\n总绝对误差: {total_error:.6f}")

        except FileNotFoundError:
            print("\n未找到adjusted_turbine_data.npy，跳过误差计算")

        # 绘图（保持不变）
        self._plot_results(time, rot_speed, gen_speed, gen_power)

        return total_error  # 显式返回总误差值

    def _plot_results(self, time, rot_speed, gen_speed, gen_power):
        """封装绘图逻辑"""
        plt.figure(figsize=(12, 8))

        # 子图1：风轮转速
        plt.subplot(3, 1, 1)
        plt.plot(time, rot_speed, 'b-', linewidth=1.5)
        plt.title(f'Wind Turbine Operational Parameters (显示{len(time)}个数据点)')
        plt.ylabel('Rotor Speed [rpm]')
        plt.grid(True)

        # 子图2：发电机转速
        plt.subplot(3, 1, 2)
        plt.plot(time, gen_speed, 'r-', linewidth=1.5)
        plt.ylabel('Generator Speed [rpm]')
        plt.grid(True)

        # 子图3：发电机功率
        plt.subplot(3, 1, 3)
        plt.plot(time, gen_power, 'g-', linewidth=1.5)
        plt.ylabel('Generator Power [kW]')
        plt.xlabel('Time [s]')
        plt.grid(True)

        plt.tight_layout()
        plt.savefig('测试turbine_performance.jpg', dpi=300, bbox_inches='tight')
        plt.close()


def run_fast_simulation_with_flags(external_variable_flags):
    """
    使用指定的变量参数运行FAST仿真并返回误差值

    参数:
        external_variable_flags: dict - 包含各变量调整系数的字典

    返回:
        float or None - 总误差值（如果存在调整后数据），否则返回None
    """
    # 设置中文字体
    rcParams['font.sans-serif'] = ['SimHei']
    rcParams['axes.unicode_minus'] = False

    # 获取路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    pickle_file = os.path.join(current_dir, "variable_dict.pkl")

    # 创建并配置FASTRunner
    runner = FASTRunner(current_dir, parent_dir, pickle_file)

    # 更新变量参数
    runner.variable_flags.update(external_variable_flags)

    # 执行仿真
    runner.execute()

    # 处理数据并获取误差
    error = runner.process_output_data()

    # 打印并返回结果
    if error is not None:
        print(f"✅ 仿真完成，总误差值: {error:.6f}")
    else:
        print("⚠️ 仿真完成，但未计算误差（缺少adjusted_turbine_data.npy）")

    return error


if __name__ == "__main__":
    # 示例调用
    test_flags = {
        'AirDens_v': 1.2,
        'KinVisc_v': 1.15,
        'Patm_v': 1.1,
        'Pvap_v': 0.95,
        'TwrCd_v': 1.1,
        'BlCrvAng_v': 1.2,
        'BlTwist_v': 1.05,
        'DampingCoeffs_v': 1.1,
        'K_v': 1.1,
        'M_v': 1.2,
        'GBoxEff_v': 1,
        'DTTorSpr_v': 1,
        'DTTorDmp_v': 1,
        'TeetDmp_v': 1,
        'TeetCDmp_v': 1,
        'M_CSmax_v': 1,
        'M_CD_v': 1,
        'sig_v_v': 1,
        'sig_v2_v': 1
    }

    error_value = run_fast_simulation_with_flags(test_flags)