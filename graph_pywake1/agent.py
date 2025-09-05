# agent.py - AI智能体模块
import os
import json
import requests
from pydantic import BaseModel, Field
from typing import Union, Optional, List, Dict, Any
from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.orm import sessionmaker, declarative_base
from langchain_core.tools import tool
from langchain_deepseek import ChatDeepSeek
from langgraph.prebuilt import create_react_agent
from base_agent import BaseAgent
# PyWake相关导入
from py_wake.examples.data.hornsrev1 import Hornsrev1Site, V80
from py_wake.literature.gaussian_models import Bastankhah_PorteAgel_2014
from py_wake.wind_farm_models.wind_farm_model import SimulationResult
from pywake_me import *


# 设置环境变量
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCHAIN_PROJECT"] = "me_test1"
os.environ["DEEPSEEK_API_KEY"] = "sk-157abd02156e4718b1132b3ed03fd5ce"


route_plots = "./static/plots/"
route_plots_abs = os.path.abspath(route_plots)

# =============================================================================
# Pydantic模型定义
# =============================================================================
class SimulationParameters(BaseModel):
    """风电场仿真参数"""
    wt_x: Optional[List[float]] = Field(None, description="Wind turbine X coordinates")
    wt_y: Optional[List[float]] = Field(None, description="Wind turbine Y coordinates")
    h: Optional[List[float]] = Field(None, description="Hub heights")
    turbine_type: int = Field(0, description="Turbine type index")
    wd: Optional[List[float]] = Field(None, description="Wind directions in degrees")
    ws: Optional[List[float]] = Field(None, description="Wind speeds in m/s")


class PlotParameters(BaseModel):
    """绘图参数"""
    nc_file: str = Field(..., description="NetCDF file path for analysis")
    wt_index: int = Field(35, description="Wind turbine index for specific plots")
    save: bool = Field(True, description="Whether to save the plot")
    show: bool = Field(False, description="Whether to display the plot")


# =============================================================================
# 工具函数定义
# =============================================================================
@tool(args_schema=SimulationParameters)
def run_wind_farm_simulation(
        wt_x: Optional[List[float]] = None,
        wt_y: Optional[List[float]] = None,
        h: Optional[List[float]] = None,
        turbine_type: int = 0,
        wd: Optional[List[float]] = None,
        ws: Optional[List[float]] = None,
):
    """
    运行风电场仿真计算，包括尾流影响分析和AEP计算。

    参数：
    - wt_x, wt_y: 风机坐标位置，如果不提供则使用默认Horns Rev 1布局
    - h: 轮毂高度
    - turbine_type: 风机类型索引
    - wd: 风向角度列表
    - ws: 风速列表
    """
    try:
        sim = WindFarmSimulation()

        # 运行仿真
        sim.run_simulation(wt_x=wt_x, wt_y=wt_y, h=h, turbine_type=turbine_type, wd=wd, ws=ws)

        # 计算AEP
        sim.calculate_and_add_aep()

        # 保存结果
        saved_path = sim.save_to_netcdf("simulation_result.nc")

        # 获取基本统计信息
        total_with_wake = sim.sim_res.aep_with_wake.sum().item()
        total_without_wake = sim.sim_res.aep_without_wake.sum().item()
        wake_loss_pct = (1 - total_with_wake / total_without_wake) * 100

        return {
            "status": "success",
            "message": f"风电场仿真计算完成，结果已保存到 {saved_path}",
            "results": {
                "total_aep_with_wake": f"{total_with_wake:.2f} GWh",
                "total_aep_without_wake": f"{total_without_wake:.2f} GWh",
                "wake_loss_percentage": f"{wake_loss_pct:.2f}%",
                "netcdf_file": saved_path,
                "turbine_count": len(sim.sim_res.wt.values)
            }
        }
    except Exception as e:
        return {"status": "error", "message": f"仿真计算失败: {str(e)}"}


@tool(args_schema=PlotParameters)
def generate_power_map_plot(
        nc_file: str,
        wt_index: int = 35,
        save: bool = True,
        show: bool = False
):
    """
    生成指定风机的功率热图，显示不同风速风向下的发电功率分布。
    """
    try:
        nc_file = os.path.join(os.path.dirname(__file__), "pywake_me", nc_file)
        if not os.path.exists(nc_file):
            return {"status": "error", "message": f"NetCDF文件不存在: {nc_file}"}

        with WindFarmAnalysis(nc_file) as analysis:

            saved_path = analysis.plot_power_map_for_turbine(wt_index=wt_index, save=save, show=show,save_path = route_plots_abs)

        return {
            "status": "success",
            "message": f"风机 {wt_index} 功率热图生成完成",
            "plot_path": saved_path if saved_path else "未保存文件"
        }
    except Exception as e:
        return {"status": "error", "message": f"功率热图生成失败: {str(e)}"}


@tool(args_schema=PlotParameters)
def generate_aep_comparison_plot(
        nc_file: str,
        wt_index: int = 35,  # 此参数在该函数中不使用，但保持接口一致
        save: bool = True,
        show: bool = False
):
    """
    生成AEP对比图，展示考虑尾流和不考虑尾流的总发电量对比。
    """
    try:
        nc_file = os.path.join(os.path.dirname(__file__), "pywake_me", nc_file)
        if not os.path.exists(nc_file):
            return {"status": "error", "message": f"NetCDF文件不存在: {nc_file}"}

        with WindFarmAnalysis(nc_file) as analysis:
            saved_path = analysis.plot_total_aep_comparison(save=save, show=show,save_path = route_plots_abs)

        return {
            "status": "success",
            "message": "总AEP对比图生成完成",
            "plot_path": saved_path if saved_path else "未保存文件"
        }
    except Exception as e:
        return {"status": "error", "message": f"AEP对比图生成失败: {str(e)}"}


@tool(args_schema=PlotParameters)
def generate_turbine_aep_plot(
        nc_file: str,
        wt_index: int = 35,  # 此参数在该函数中不使用
        save: bool = True,
        show: bool = False
):
    """
    生成每台风机的AEP分析图，包括风机布局和发电量曲线。
    """
    try:
        nc_file = os.path.join(os.path.dirname(__file__), "pywake_me", nc_file)
        if not os.path.exists(nc_file):
            return {"status": "error", "message": f"NetCDF文件不存在: {nc_file}"}

        with WindFarmAnalysis(nc_file) as analysis:
            saved_path = analysis.plot_aep_per_turbine( save=save, show=show,save_path = route_plots_abs)

        return {
            "status": "success",
            "message": "单机AEP分析图生成完成",
            "plot_path": saved_path if saved_path else "未保存文件"
        }
    except Exception as e:
        return {"status": "error", "message": f"单机AEP分析图生成失败: {str(e)}"}


@tool(args_schema=PlotParameters)
def generate_wake_loss_heatmap(
        nc_file: str,
        wt_index: int = 35,  # 此参数在该函数中不使用
        save: bool = True,
        show: bool = False
):
    """
    生成尾流损失热图，展示不同风速风向下的尾流影响程度。
    """
    try:
        nc_file = os.path.join(os.path.dirname(__file__), "pywake_me", nc_file)
        if not os.path.exists(nc_file):
            return {"status": "error", "message": f"NetCDF文件不存在: {nc_file}"}

        with WindFarmAnalysis(nc_file) as analysis:
            saved_path = analysis.plot_wake_loss_heatmap(save=save, show=show,save_path = route_plots_abs)

        return {
            "status": "success",
            "message": "尾流损失热图生成完成",
            "plot_path": saved_path if saved_path else "未保存文件"
        }
    except Exception as e:
        return {"status": "error", "message": f"尾流损失热图生成失败: {str(e)}"}


@tool(args_schema=PlotParameters)
def generate_aep_windspeed_plot(
        nc_file: str,
        wt_index: int = 35,  # 此参数在该函数中不使用
        save: bool = True,
        show: bool = False
):
    """
    生成AEP与风速关系图，展示不同风速下的总发电量变化。
    """
    try:
        nc_file = os.path.join(os.path.dirname(__file__), "pywake_me", nc_file)
        if not os.path.exists(nc_file):
            return {"status": "error", "message": f"NetCDF文件不存在: {nc_file}"}

        with WindFarmAnalysis(nc_file) as analysis:
            saved_path = analysis.plot_aep_vs_windspeed(save=save, show=show,save_path = route_plots_abs)

        return {
            "status": "success",
            "message": "AEP-风速关系图生成完成",
            "plot_path": saved_path if saved_path else "未保存文件"
        }
    except Exception as e:
        return {"status": "error", "message": f"AEP-风速关系图生成失败: {str(e)}"}

# =============================================================================
# 智能体类定义
# =============================================================================
class WindFarmAgent(BaseAgent):
    """风电场智能体，支持PyWake仿真计算和结果分析"""

    def __init__(self, api_key=None):
        """初始化风电场智能体"""
        super().__init__()

        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")

        # 初始化LLM
        self.llm = ChatDeepSeek(
            model="deepseek-chat",
            temperature=0.7,
            max_tokens=2048,
            timeout=30.0,
            max_retries=2,
            api_key=self.api_key,
        )

        # 系统提示词
        self.system_prompt = self._get_system_prompt()

        # 工具列表
        self.tools = [
            run_wind_farm_simulation,
            generate_power_map_plot,
            generate_aep_comparison_plot,
            generate_turbine_aep_plot,
            generate_wake_loss_heatmap,
            generate_aep_windspeed_plot
        ]

        # 创建ReAct智能体
        self.graph = create_react_agent(self.llm, tools=self.tools, prompt=self.system_prompt)

    def _get_system_prompt(self):
        """获取系统提示词"""
        return """你是一个专业的风电场分析AI助手，精通PyWake风电场仿真计算和结果分析。你的回复需要使用结构化的HTML格式。

    ## 核心功能：
    1. **风电场仿真计算**：使用PyWake进行风电场尾流建模和AEP（年发电量）计算
    2. **结果可视化分析**：生成多种专业图表，包括功率热图、AEP对比、尾流损失分析等
    3. **专业数据解读**：提供风电场性能评估和优化建议

    ## 重要规则：
    1. 你的回复必须是有效的HTML片段，不需要完整的HTML文档结构
    2. 直接输出HTML内容，不要用代码块包裹
    3. 使用适当的HTML标签来组织内容结构
    4. 优先使用工具进行计算和分析
    5. 提供专业、准确的风电场技术解读

    ## 🖼️ 图片路径处理规则（极其重要）：
    **工具返回的图片路径是绝对路径，你必须将其转换为Web可访问的相对路径！**

    ### 路径转换规则：
    - 🚫 **错误示例**：`<img src="D:\python_code\python_web\graph_pywake1\static\plots\power_map_wt35.png">`
    - ✅ **正确示例**：`<img src="/static/plots/power_map_wt35.png">`

    ### 转换方法：
    1. 从工具返回的绝对路径中提取文件名（最后一部分）
    2. 构造Web路径：`/static/plots/` + 文件名
    3. 例如：`D:\...\plots\power_map_wt35.png` → `/static/plots/power_map_wt35.png`

    ### 图片显示标准格式（每张图片独占一行）：
    ```html
    <div style="margin: 30px 0; text-align: center;">
        <h4 style="margin-bottom: 15px; color: #2c3e50;">🌪️ 单机功率热图</h4>
        <img src="/static/plots/power_map_wt35.png" alt="单机功率热图" 
             style="max-width: 100%; height: auto; border: 2px solid #ddd; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
    </div>

    <div style="margin: 30px 0; text-align: center;">
        <h4 style="margin-bottom: 15px; color: #2c3e50;">📊 总AEP对比图</h4>
        <img src="/static/plots/aep_comparison.png" alt="总AEP对比图" 
             style="max-width: 100%; height: auto; border: 2px solid #ddd; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
    </div>
    ```

    ## 工具调用规则（重要）：
    **必须严格按顺序执行画图工具，每次只调用一个工具，等待结果返回后再调用下一个工具。**

    ### 标准画图顺序：
    1. 首先调用 `generate_power_map_plot` - 生成单机功率热图
    2. 等待结果后调用 `generate_aep_comparison_plot` - 生成总AEP对比图  
    3. 等待结果后调用 `generate_turbine_aep_plot` - 生成各风机AEP分析图
    4. 等待结果后调用 `generate_wake_loss_heatmap` - 生成尾流损失热图
    5. 最后调用 `generate_aep_windspeed_plot` - 生成AEP-风速关系图

    **禁止同时并发调用多个画图工具！** 这会导致文件访问冲突。

    ## 可用工具说明：
    - `run_wind_farm_simulation`: 运行风电场仿真，计算尾流影响和AEP
    - `generate_power_map_plot`: 生成单机功率热图（第1个执行）
    - `generate_aep_comparison_plot`: 生成总AEP对比图（第2个执行）
    - `generate_turbine_aep_plot`: 生成各风机AEP分析图（第3个执行）
    - `generate_wake_loss_heatmap`: 生成尾流损失热图（第4个执行）
    - `generate_aep_windspeed_plot`: 生成AEP-风速关系图（第5个执行）

    ## 📝 重要提醒：
    1. **图片垂直布局**：每张图片独占一行，居中显示，确保清晰可见
    2. **必须手动转换图片路径**：无论工具返回什么绝对路径，都要提取文件名并使用 `/static/plots/文件名` 格式
    3. **图片必须包含alt属性**：提供有意义的描述
    4. **添加图片说明**：每张图片下方添加简短的说明文字
    5. **美观样式**：使用阴影、边框、圆角等提升视觉效果
    6. **保持专业性**：使用合适的颜色和样式来突出重要信息
    7. **按顺序调用工具**：严格按照指定顺序调用画图工具，避免文件冲突

    记住：你的任务是将工具返回的绝对路径转换为 `/static/plots/文件名.png` 格式！每张图片都要独占一行，居中显示！"""

# =============================================================================
# 工厂函数
# =============================================================================
def create_wind_farm_agent(api_key=None):
    """创建风电场智能体实例"""
    return WindFarmAgent(api_key)


# =============================================================================
# 测试代码
# =============================================================================
if __name__ == "__main__":
    # 创建智能体
    agent = create_wind_farm_agent()

    print("=== 测试风电场智能体 ===")

    '''
    print("\n1. 测试仿真计算:")
    for data in agent.chat_stream("请运行一个标准的风电场仿真计算，使用默认的Horns Rev 1风电场布局"):
        print(f"流式数据: {data}")
    '''


    # 测试图表生成
    print("\n2. 测试图表生成:")
    for data in agent.chat_stream("请为simulation_result.nc生成所有类型的分析图表"):
        print(f"流式数据: {data}")

    # 测试完整分析
    print("\n3. 测试完整分析:")
    for data in agent.chat_stream("请进行一次完整的风电场仿真分析，包括计算和所有图表生成，并提供专业的性能评估报告"):
        print(f"流式数据: {data}")