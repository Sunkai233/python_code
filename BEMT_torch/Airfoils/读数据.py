"""
翼型数据读取器 - 读取AirfoilInfo v1.01.x格式文件
Airfoil Data Reader - Parse AirfoilInfo v1.01.x format files
"""

import os
import re
import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any


class AirfoilDataReader:
    """翼型数据读取器类"""

    def __init__(self, airfoil_directory: str = "./"):
        """
        初始化翼型数据读取器

        Args:
            airfoil_directory: 翼型文件目录路径
        """
        self.airfoil_directory = Path(airfoil_directory)
        self.airfoil_database = {}

        # 翼型文件映射
        self.airfoil_files = {
            1: "Cylinder1.dat",
            2: "Cylinder2.dat",
            3: "DU40_A17.dat",
            4: "DU35_A17.dat",
            5: "DU30_A17.dat",
            6: "DU25_A17.dat",
            7: "DU21_A17.dat",
            8: "NACA64_A17.dat"
        }

    def parse_airfoil_file(self, filepath: Path) -> Dict[str, Any]:
        """
        解析单个翼型数据文件

        Args:
            filepath: 翼型文件路径

        Returns:
            翼型数据字典
        """
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
        except Exception as e:
            print(f"读取文件失败 {filepath}: {e}")
            return {}

        # 解析结果
        airfoil_data = {
            'filename': filepath.name,
            'basic_params': {},
            'ua_params': {},
            'aero_table': None,
            'metadata': {}
        }

        # 解析各个部分
        airfoil_data['basic_params'] = self._parse_basic_parameters(lines)
        airfoil_data['ua_params'] = self._parse_ua_parameters(lines)
        airfoil_data['aero_table'] = self._parse_aerodynamic_table(lines)
        airfoil_data['metadata'] = self._parse_metadata(lines)

        return airfoil_data

    def _parse_basic_parameters(self, lines: List[str]) -> Dict[str, Any]:
        """解析基本参数"""
        params = {}

        for line in lines:
            line = line.strip()
            if not line or line.startswith('!'):
                continue

            # 插值阶数
            if 'InterpOrd' in line:
                match = re.search(r'"?([^"]*)"?\s+InterpOrd', line)
                if match:
                    params['interpolation_order'] = match.group(1)

            # 无量纲面积
            elif 'NonDimArea' in line:
                match = re.search(r'(\d+\.?\d*)\s+NonDimArea', line)
                if match:
                    params['non_dim_area'] = float(match.group(1))

            # 坐标文件
            elif 'NumCoords' in line and '@' in line:
                match = re.search(r'@"([^"]*)"', line)
                if match:
                    params['coords_file'] = match.group(1)

            # 边界层文件
            elif 'BL_file' in line:
                match = re.search(r'"([^"]*)"?\s+BL_file', line)
                if match:
                    params['bl_file'] = match.group(1)

            # 表格数量
            elif 'NumTabs' in line:
                match = re.search(r'(\d+)\s+NumTabs', line)
                if match:
                    params['num_tables'] = int(match.group(1))

            # 雷诺数
            elif 'Re' in line and 'Reynolds' in line:
                match = re.search(r'(\d+\.?\d*)\s+Re', line)
                if match:
                    params['reynolds_number'] = float(match.group(1)) * 1e6

            # 用户属性
            elif 'UserProp' in line:
                match = re.search(r'(\d+)\s+UserProp', line)
                if match:
                    params['user_property'] = int(match.group(1))

            # UA数据包含标志
            elif 'InclUAdata' in line:
                if 'True' in line:
                    params['include_ua_data'] = True
                elif 'False' in line:
                    params['include_ua_data'] = False

        return params

    def _parse_ua_parameters(self, lines: List[str]) -> Dict[str, float]:
        """解析非定常气动参数"""
        ua_params = {}

        # UA参数映射
        ua_param_map = {
            'alpha0': 'zero_lift_aoa',
            'alpha1': 'pos_stall_aoa',
            'alpha2': 'neg_stall_aoa',
            'eta_e': 'recovery_factor',
            'C_nalpha': 'normal_force_slope',
            'T_f0': 'time_const_f0',
            'T_V0': 'time_const_v0',
            'T_p': 'time_const_p',
            'T_VL': 'time_const_vl',
            'b1': 'const_b1',
            'b2': 'const_b2',
            'b5': 'const_b5',
            'A1': 'const_a1',
            'A2': 'const_a2',
            'A5': 'const_a5',
            'S1': 'const_s1',
            'S2': 'const_s2',
            'S3': 'const_s3',
            'S4': 'const_s4',
            'Cn1': 'critical_cn1',
            'Cn2': 'critical_cn2',
            'St_sh': 'strouhal_shedding',
            'Cd0': 'zero_lift_cd',
            'Cm0': 'zero_lift_cm',
            'k0': 'const_k0',
            'k1': 'const_k1',
            'k2': 'const_k2',
            'k3': 'const_k3',
            'k1_hat': 'const_k1_hat',
            'x_cp_bar': 'const_x_cp_bar'
        }

        for line in lines:
            line = line.strip()
            if not line or line.startswith('!'):
                continue

            for param_key, param_name in ua_param_map.items():
                if param_key in line and '!' in line:
                    # 提取数值
                    match = re.search(r'(-?\d+\.?\d*)\s+' + re.escape(param_key), line)
                    if match:
                        try:
                            ua_params[param_name] = float(match.group(1))
                        except ValueError:
                            pass
                        break

        return ua_params

    def _parse_aerodynamic_table(self, lines: List[str]) -> pd.DataFrame:
        """解析气动系数表"""
        # 找到数据表开始位置
        table_start = None
        num_data_lines = 0

        for i, line in enumerate(lines):
            # 查找数据行数
            if 'NumAlf' in line:
                match = re.search(r'(\d+)\s+NumAlf', line)
                if match:
                    num_data_lines = int(match.group(1))

            # 查找表头
            if 'Alpha' in line and 'Cl' in line and 'Cd' in line:
                table_start = i + 1
                break

        if table_start is None:
            print("警告：未找到气动系数表")
            return pd.DataFrame()

        # 解析数据
        data = []
        lines_read = 0

        for line in lines[table_start:]:
            line = line.strip()
            if not line or line.startswith('!'):
                continue

            parts = line.split()
            if len(parts) >= 3:
                try:
                    alpha = float(parts[0])
                    cl = float(parts[1])
                    cd = float(parts[2])
                    cm = float(parts[3]) if len(parts) > 3 else 0.0

                    data.append([alpha, cl, cd, cm])
                    lines_read += 1

                    # 如果指定了数据行数，读取完毕后停止
                    if num_data_lines > 0 and lines_read >= num_data_lines:
                        break

                except ValueError:
                    continue

        if not data:
            print("警告：未找到有效的气动系数数据")
            return pd.DataFrame()

        # 创建DataFrame
        df = pd.DataFrame(data, columns=['Alpha', 'Cl', 'Cd', 'Cm'])

        return df

    def _parse_metadata(self, lines: List[str]) -> Dict[str, str]:
        """解析元数据和注释"""
        metadata = {
            'description': '',
            'source': '',
            'corrections': '',
            'comments': []
        }

        # 提取前几行的注释作为描述
        description_lines = []
        for line in lines[:10]:
            if line.startswith('!') and not line.startswith('! ---'):
                clean_line = line.lstrip('! ').strip()
                if clean_line:
                    description_lines.append(clean_line)

        metadata['description'] = ' '.join(description_lines)

        # 查找特定信息
        full_text = ''.join(lines)

        if 'DOWEC' in full_text:
            metadata['source'] = 'DOWEC document'
        if 'Jason Jonkman' in full_text:
            metadata['corrections'] = 'Corrected by Jason Jonkman'
        if 'AirfoilPrep' in full_text:
            metadata['tool'] = 'AirfoilPrep'

        return metadata

    def load_all_airfoils(self) -> Dict[int, Dict[str, Any]]:
        """加载所有翼型数据"""
        print(f"正在从目录加载翼型数据: {self.airfoil_directory}")

        for airfoil_id, filename in self.airfoil_files.items():
            filepath = self.airfoil_directory / filename

            if filepath.exists():
                print(f"正在读取翼型 {airfoil_id}: {filename}")
                airfoil_data = self.parse_airfoil_file(filepath)

                if airfoil_data:
                    self.airfoil_database[airfoil_id] = airfoil_data
                    # 添加翼型ID
                    self.airfoil_database[airfoil_id]['airfoil_id'] = airfoil_id
                    print(f"  ✓ 成功读取 {len(airfoil_data['aero_table'])} 行气动数据")
                else:
                    print(f"  ✗ 读取失败")
            else:
                print(f"  ✗ 文件不存在: {filepath}")

        print(f"总共加载了 {len(self.airfoil_database)} 个翼型")
        return self.airfoil_database

    def save_to_pickle(self, output_path: str = "nrel_5mw_airfoil_database.pkl"):
        """保存为pickle格式"""
        try:
            with open(output_path, 'wb') as f:
                pickle.dump(self.airfoil_database, f)
            print(f"✓ 数据已保存为pickle格式: {output_path}")
        except Exception as e:
            print(f"✗ 保存pickle失败: {e}")

    def save_to_json(self, output_path: str = "nrel_5mw_airfoil_database.json"):
        """保存为JSON格式（需要转换DataFrame）"""
        try:
            # 转换DataFrame为字典格式
            json_data = {}
            for airfoil_id, data in self.airfoil_database.items():
                json_data[str(airfoil_id)] = {
                    'filename': data['filename'],
                    'airfoil_id': data['airfoil_id'],
                    'basic_params': data['basic_params'],
                    'ua_params': data['ua_params'],
                    'metadata': data['metadata'],
                    'aero_table': data['aero_table'].to_dict('records') if not data['aero_table'].empty else []
                }

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            print(f"✓ 数据已保存为JSON格式: {output_path}")
        except Exception as e:
            print(f"✗ 保存JSON失败: {e}")

    def save_to_numpy(self, output_path: str = "nrel_5mw_airfoil_database.npz"):
        """保存为NumPy格式"""
        try:
            # 准备NumPy数组
            arrays_to_save = {}

            for airfoil_id, data in self.airfoil_database.items():
                prefix = f"airfoil_{airfoil_id}_"

                # 保存气动系数表
                if not data['aero_table'].empty:
                    arrays_to_save[f"{prefix}alpha"] = data['aero_table']['Alpha'].values
                    arrays_to_save[f"{prefix}cl"] = data['aero_table']['Cl'].values
                    arrays_to_save[f"{prefix}cd"] = data['aero_table']['Cd'].values
                    arrays_to_save[f"{prefix}cm"] = data['aero_table']['Cm'].values

                # 保存关键参数
                if 'reynolds_number' in data['basic_params']:
                    arrays_to_save[f"{prefix}reynolds"] = np.array([data['basic_params']['reynolds_number']])

                # 保存UA参数
                for param_name, param_value in data['ua_params'].items():
                    arrays_to_save[f"{prefix}{param_name}"] = np.array([param_value])

            np.savez_compressed(output_path, **arrays_to_save)
            print(f"✓ 数据已保存为NumPy格式: {output_path}")
        except Exception as e:
            print(f"✗ 保存NumPy失败: {e}")

    def print_summary(self):
        """打印数据库摘要"""
        print("\n" + "=" * 60)
        print("NREL 5MW 翼型数据库摘要")
        print("=" * 60)

        for airfoil_id, data in self.airfoil_database.items():
            print(f"\n翼型 {airfoil_id}: {data['filename']}")
            print(f"  描述: {data['metadata']['description'][:80]}...")

            # 基本参数
            basic = data['basic_params']
            if 'reynolds_number' in basic:
                print(f"  雷诺数: {basic['reynolds_number']:.1e}")

            # 气动数据
            if not data['aero_table'].empty:
                aero = data['aero_table']
                print(f"  数据点数: {len(aero)}")
                print(f"  攻角范围: {aero['Alpha'].min():.1f}° ~ {aero['Alpha'].max():.1f}°")
                print(f"  最大Cl: {aero['Cl'].max():.3f}")
                print(f"  最小Cd: {aero['Cd'].min():.4f}")

            # UA参数
            ua = data['ua_params']
            if 'zero_lift_aoa' in ua:
                print(f"  零升攻角: {ua['zero_lift_aoa']:.1f}°")
            if 'pos_stall_aoa' in ua:
                print(f"  失速攻角: +{ua['pos_stall_aoa']:.1f}°/-{abs(ua['neg_stall_aoa']):.1f}°")

    def get_airfoil_data(self, airfoil_id: int) -> Dict[str, Any]:
        """获取指定翼型的数据"""
        return self.airfoil_database.get(airfoil_id, {})

    def get_aero_coefficients(self, airfoil_id: int, alpha_deg: float) -> Tuple[float, float, float]:
        """
        获取指定攻角的气动系数（线性插值）

        Args:
            airfoil_id: 翼型ID
            alpha_deg: 攻角（度）

        Returns:
            (Cl, Cd, Cm) 气动系数
        """
        if airfoil_id not in self.airfoil_database:
            return 0.0, 0.5, 0.0  # 默认值

        aero_table = self.airfoil_database[airfoil_id]['aero_table']
        if aero_table.empty:
            return 0.0, 0.5, 0.0

        # 线性插值
        alpha_array = aero_table['Alpha'].values
        cl_array = aero_table['Cl'].values
        cd_array = aero_table['Cd'].values
        cm_array = aero_table['Cm'].values

        cl = np.interp(alpha_deg, alpha_array, cl_array)
        cd = np.interp(alpha_deg, alpha_array, cd_array)
        cm = np.interp(alpha_deg, alpha_array, cm_array)

        return float(cl), float(cd), float(cm)


def main():
    """主函数 - 使用示例"""
    # 创建翼型数据读取器
    reader = AirfoilDataReader("./")

    # 加载所有翼型数据
    airfoil_database = reader.load_all_airfoils()

    # 打印摘要
    reader.print_summary()

    # 保存为不同格式
    print(f"\n正在保存数据...")
    reader.save_to_pickle("nrel_5mw_airfoils.pkl")
    reader.save_to_json("nrel_5mw_airfoils.json")
    reader.save_to_numpy("nrel_5mw_airfoils.npz")

    # 测试数据访问
    print(f"\n=== 数据访问测试 ===")

    # 测试获取DU21翼型在不同攻角的系数
    airfoil_id = 7  # DU21_A17
    test_angles = [0, 5, 10, 15, 20]

    print(f"DU21翼型气动系数:")
    print("攻角(°)\tCl\tCd\tCm")
    print("-" * 40)

    for alpha in test_angles:
        cl, cd, cm = reader.get_aero_coefficients(airfoil_id, alpha)
        print(f"{alpha:4.0f}\t{cl:.3f}\t{cd:.4f}\t{cm:.3f}")

    print(f"\n✓ 所有数据处理完成！")
    return airfoil_database


# 数据加载函数
def load_airfoil_database(filepath: str = "nrel_5mw_airfoils.pkl") -> Dict[int, Dict[str, Any]]:
    """
    加载已保存的翼型数据库

    Args:
        filepath: 数据库文件路径

    Returns:
        翼型数据库字典
    """
    try:
        with open(filepath, 'rb') as f:
            database = pickle.load(f)
        print(f"✓ 成功加载翼型数据库: {filepath}")
        print(f"  包含 {len(database)} 个翼型")
        return database
    except Exception as e:
        print(f"✗ 加载数据库失败: {e}")
        return {}


if __name__ == "__main__":
    # 运行主程序
    airfoil_data = main()

    print(f"\n=== 使用示例 ===")
    print("# 加载已保存的数据:")
    print("from airfoil_reader import load_airfoil_database")
    print("database = load_airfoil_database('nrel_5mw_airfoils.pkl')")
    print("")
    print("# 获取翼型数据:")
    print("du21_data = database[7]  # DU21翼型")
    print("aero_table = du21_data['aero_table']")
    print("ua_params = du21_data['ua_params']")