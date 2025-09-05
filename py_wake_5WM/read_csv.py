import numpy as np
import pandas as pd


def csv_to_numpy_array(csv_path):
    """
    将CSV文件中的数值数据转换为NumPy数组
    参数:
        csv_path: CSV文件路径
    返回:
        numpy_array: 仅包含数值数据的NumPy数组
    """
    # 读取CSV文件
    df = pd.read_csv(csv_path)

    # 提取所有数值列并转换为NumPy数组
    return df.select_dtypes(include=[np.number]).to_numpy()


# 使用示例
if __name__ == "__main__":
    # 加载数据
    data_array = csv_to_numpy_array("wind_turbine_results.csv")

    # 输出结果
    print("NumPy数组内容:")
    print(data_array)
    print("\n数组形状:", data_array.shape)
    print("数据类型:", data_array.dtype)