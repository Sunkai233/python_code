import re
import os


def find_subroutines_in_f90(file_path):
    """
    读取包含Fortran 90代码的txt文件，找到所有subroutine定义行

    参数:
    file_path (str): txt文件的路径

    返回:
    None (直接打印结果)
    """

    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误：文件 '{file_path}' 不存在!")
        return

    try:
        # 读取文件
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            lines = file.readlines()

        print(f"正在分析文件: {file_path}")
        print(f"文件总行数: {len(lines)}")
        print("=" * 60)

        # 用于匹配subroutine定义的正则表达式
        # 匹配以subroutine开头的行（忽略大小写和前导空格）
        subroutine_pattern = re.compile(r'^\s*subroutine\s+\w+', re.IGNORECASE)

        found_subroutines = []

        # 逐行检查
        for line_number, line in enumerate(lines, 1):
            # 去除行尾的换行符
            clean_line = line.rstrip('\n\r')

            # 检查是否匹配subroutine模式
            if subroutine_pattern.match(clean_line):
                found_subroutines.append((line_number, clean_line))

        # 打印结果
        if found_subroutines:
            print(f"找到 {len(found_subroutines)} 个subroutine定义:")
            print("=" * 60)

            for line_num, line_content in found_subroutines:
                print(f"第 {line_num:4d} 行: {line_content}")

            print("=" * 60)
            print(f"总共找到 {len(found_subroutines)} 个subroutine")
        else:
            print("未找到任何subroutine定义")

    except Exception as e:
        print(f"读取文件时发生错误: {e}")


def find_subroutines_advanced(file_path):
    """
    更高级的subroutine查找函数，可以处理更复杂的情况
    包括注释行、续行等

    参数:
    file_path (str): txt文件的路径
    """

    if not os.path.exists(file_path):
        print(f"错误：文件 '{file_path}' 不存在!")
        return

    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            lines = file.readlines()

        print(f"正在分析文件: {file_path}")
        print(f"文件总行数: {len(lines)}")
        print("=" * 80)

        # 更详细的正则表达式模式
        subroutine_patterns = [
            re.compile(r'^\s*subroutine\s+(\w+)', re.IGNORECASE),  # 标准subroutine
            re.compile(r'^\s*recursive\s+subroutine\s+(\w+)', re.IGNORECASE),  # recursive subroutine
            re.compile(r'^\s*pure\s+subroutine\s+(\w+)', re.IGNORECASE),  # pure subroutine
            re.compile(r'^\s*elemental\s+subroutine\s+(\w+)', re.IGNORECASE),  # elemental subroutine
        ]

        found_subroutines = []

        for line_number, line in enumerate(lines, 1):
            clean_line = line.rstrip('\n\r')

            # 跳过注释行（以C, c, !, *开头的行）
            if re.match(r'^\s*[Cc!*]', clean_line):
                continue

            # 检查所有模式
            for pattern in subroutine_patterns:
                match = pattern.match(clean_line)
                if match:
                    subroutine_name = match.group(1) if match.groups() else "未知"
                    found_subroutines.append((line_number, clean_line, subroutine_name))
                    break

        # 打印结果
        if found_subroutines:
            print(f"找到 {len(found_subroutines)} 个subroutine定义:")
            print("=" * 80)

            for i, (line_num, line_content, sub_name) in enumerate(found_subroutines, 1):
                print(f"{i:2d}. 第 {line_num:4d} 行 - Subroutine名称: {sub_name}")
                print(f"    内容: {line_content}")
                print()

            print("=" * 80)
            print("总结:")
            print(f"- 总共找到 {len(found_subroutines)} 个subroutine")
            print("- Subroutine名称列表:")
            for _, _, name in found_subroutines:
                print(f"  • {name}")
        else:
            print("未找到任何subroutine定义")

    except Exception as e:
        print(f"读取文件时发生错误: {e}")


# 主程序
if __name__ == "__main__":
    # 直接使用指定的文件名
    file_path = "f90文件.txt"

    print("Fortran 90 Subroutine查找器")
    print("=" * 40)
    print(f"分析文件: {file_path}")

    print("\n选择分析模式:")
    print("1. 基础模式（快速查找）")
    print("2. 高级模式（详细分析）")

    choice = input("请选择模式 (1/2, 默认为1): ").strip()

    print("\n" + "=" * 60)

    if choice == "2":
        find_subroutines_advanced(file_path)
    else:
        find_subroutines_in_f90(file_path)

    print("\n分析完成!")

    # 可选：等待用户按键退出
    input("\n按回车键退出...")

# 如果你想直接运行分析，不需要交互，可以使用下面的代码：
"""
# 直接分析指定文件的示例
file_path = "f90文件.txt"
print("=== 基础模式分析 ===")
find_subroutines_in_f90(file_path)
print("\n=== 高级模式分析 ===")
find_subroutines_advanced(file_path)
"""