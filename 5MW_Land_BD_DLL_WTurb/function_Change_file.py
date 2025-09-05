import os
import pickle
import numpy as np
from openfast_toolbox.io import FASTInputFile

def generate_variable_values(values_dict: dict, variable_flags: dict) -> dict:
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

def apply_variables_to_fast_inputs(variable_values: dict, aero, blade, beamdyn, elastodyn):
    for key_v, value in variable_values.items():
        key = key_v[:-2]

        if key in ['AirDens', 'KinVisc', 'Patm', 'Pvap']:
            aero[key] = value

        elif key == 'TwrCd' and 'TowProp' in aero.keys():
            TowProp = np.array(aero['TowProp'])
            if TowProp.shape[1] >= 3 and len(value) == TowProp.shape[0]:
                TowProp[:, 2] = value
                aero['TowProp'] = TowProp.tolist()

        elif key in ['BlCrvAng', 'BlTwist']:
            col_index = 3 if key == 'BlCrvAng' else 4
            BldAeroNodes = np.array(blade['BldAeroNodes'])
            if len(value) == BldAeroNodes.shape[0]:
                BldAeroNodes[:, col_index] = value
                blade['BldAeroNodes'] = BldAeroNodes.tolist()

        elif key == 'DampingCoeffs':
            new_vals = np.array(value)
            old_vals = beamdyn['DampingCoeffs']
            if np.shape(old_vals) == np.shape(new_vals):
                for i in range(len(old_vals)):
                    for j in range(len(old_vals[0])):
                        beamdyn['DampingCoeffs'][i][j] = new_vals[i][j]

        elif key == 'K':
            beamdyn['BeamProperties']['K'] = value
        elif key == 'M':
            beamdyn['BeamProperties']['M'] = value
        elif key in elastodyn.keys():
            elastodyn[key] = value
        else:
            print(f"⚠️ 未识别或未处理的变量：{key}")

    # 修正 blade 数组
    if isinstance(blade['BldAeroNodes'], list):
        blade['BldAeroNodes'] = np.array(blade['BldAeroNodes'])
    blade['NumBlNds'] = blade['BldAeroNodes'].shape[0]


def save_fast_files(aero, blade, beamdyn, elastodyn, paths: dict, write_blade: bool = False):
    """
    将修改后的 FAST 模块输入写回文件中。

    参数：
    - aero, blade, beamdyn, elastodyn: FASTInputFile 对象
    - paths: 字典，包含文件路径，如 {'aero': ..., 'blade': ..., ...}
    - write_blade: 是否保存 blade 文件（默认 False）
    """
    try:
        aero.write(paths['aero'])
        if write_blade:
            blade.write(paths['blade'])
        beamdyn.write(paths['beamdyn'])
        elastodyn.write(paths['elastodyn'])
        print("✅ 文件已成功保存")
    except Exception as e:
        print(f"❌ 文件写入错误: {e}")


def load_fast_input_files(base_dir=None):
    """
    加载 OpenFAST 所需输入文件，并设置默认参数值。

    参数：
        base_dir (str): 可选，基础路径（一般为当前脚本目录）

    返回：
        aero, blade, beamdyn, elastodyn, values_dict, paths
    """
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(base_dir)

    paths = {
        'aero': os.path.join(base_dir, "NRELOffshrBsline5MW_Onshore_AeroDyn.dat"),
        'blade': os.path.join(parent_dir, "5MW_Baseline", "NRELOffshrBsline5MW_AeroDyn_blade.dat"),
        'beamdyn': os.path.join(parent_dir, "5MW_Baseline", "NRELOffshrBsline5MW_BeamDyn_Blade.dat"),
        'elastodyn': os.path.join(base_dir, "NRELOffshrBsline5MW_Onshore_ElastoDyn_BDoutputs.dat"),
        'pickle': os.path.join(base_dir, "variable_dict.pkl")
    }

    # 加载输入文件
    aero = FASTInputFile(paths['aero'])
    blade = FASTInputFile(paths['blade'])
    beamdyn = FASTInputFile(paths['beamdyn'])
    elastodyn = FASTInputFile(paths['elastodyn'])

    # 加载变量字典
    with open(paths['pickle'], "rb") as f:
        values_dict = pickle.load(f)

    # 设置默认值
    values_dict.update({
        'AirDens': 1.1225,
        'KinVisc': 1.5e-5,
        'Patm': 101325,
        'Pvap': 0
    })

    return aero, blade, beamdyn, elastodyn, values_dict, paths


def main():
    # 文件路径初始化
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)

    aero_file = os.path.join(current_dir, "NRELOffshrBsline5MW_Onshore_AeroDyn.dat")
    blade_file = os.path.join(parent_dir, "5MW_Baseline", "NRELOffshrBsline5MW_AeroDyn_blade.dat")
    beamdyn_file = os.path.join(parent_dir, "5MW_Baseline", "NRELOffshrBsline5MW_BeamDyn_Blade.dat")
    elastodyn_file = os.path.join(current_dir, "NRELOffshrBsline5MW_Onshore_ElastoDyn_BDoutputs.dat")

    # 加载 FAST 输入文件
    aero = FASTInputFile(aero_file)
    blade = FASTInputFile(blade_file)
    beamdyn = FASTInputFile(beamdyn_file)
    elastodyn = FASTInputFile(elastodyn_file)

    # 加载初始值
    with open(os.path.join(current_dir, "variable_dict.pkl"), "rb") as f:
        values_dict = pickle.load(f)

    # 初始化：修改某些值
    values_dict.update({
        'AirDens': 1.1225,
        'KinVisc': 1.5e-5,
        'Patm': 101325,
        'Pvap': 0
    })

    # 示例 flags（可外部传入）
    variable_flags = {
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

    # 计算变量值
    variable_values = generate_variable_values(values_dict, variable_flags)

    # 应用变量并写入文件
    apply_variables_to_fast_inputs(variable_values, aero, blade, beamdyn, elastodyn)

    # 写回文件
    aero.write(aero_file)
    # blade.write(blade_file)  # 可根据是否想覆盖 blade 决定是否注释
    beamdyn.write(beamdyn_file)
    elastodyn.write(elastodyn_file)

    print("✅ 所有修改已写入输入文件。")

if __name__ == "__main__":
    main()