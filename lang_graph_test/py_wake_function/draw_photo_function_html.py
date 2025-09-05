import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
from py_wake.examples.data.hornsrev1 import Hornsrev1Site, V80
from py_wake.literature.gaussian_models import Bastankhah_PorteAgel_2014
import matplotlib
from py_wake.wind_farm_models.wind_farm_model import SimulationResult
import os
import base64
from io import BytesIO

matplotlib.use('QtAgg')


class WindFarmAnalysis:
    def __init__(self, nc_file: str):
        """
        初始化风电场分析器，加载 NetCDF 结果数据并设置基本场景参数。

        Parameters:
        - nc_file: str，NetCDF 文件路径，通常由 SimulationResult.to_netcdf() 保存。
        """
        # 使用 Horns Rev1 场地及 V80 风机类型
        self.site = Hornsrev1Site()
        self.windTurbines = V80()
        self.x, self.y = self.site.initial_position.T

        # 选择尾流模型
        self.wf_model = Bastankhah_PorteAgel_2014(self.site, self.windTurbines, k=0.0324555)

        # 加载仿真结果
        self.sim_res = xr.open_dataset(nc_file)

        # 提取基本信息
        self.wt_ids = self.sim_res.wt.values
        self.x_pos = self.sim_res.x.values
        self.y_pos = self.sim_res.y.values

        # 默认保存目录
        self.default_save_dir = r"D:\python_code\Wind_Turbine2\lang_graph_test\py_wake_function\photo_all"

    def _fig_to_base64(self, fig):
        """
        将 matplotlib figure 转换为 base64 编码的字符串。

        Parameters:
        - fig: matplotlib.figure.Figure 对象

        Returns:
        - str: base64 编码的图片字符串
        """
        buffer = BytesIO()
        fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        buffer.close()
        return img_base64

    def _save_or_show(self, fig, filename=None, show=True):
        """
        辅助函数，保存和/或显示图像，并返回 base64 编码。

        Parameters:
        - fig: matplotlib.figure.Figure 对象
        - filename: str 或 None，保存文件名（含路径），None表示不保存
        - show: bool，是否显示图像

        Returns:
        - str: base64 编码的图片字符串
        """
        # 生成 base64 编码
        img_base64 = self._fig_to_base64(fig)

        if filename:
            # 确保文件夹存在
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            fig.savefig(filename)
        if show:
            plt.show()
        else:
            plt.close(fig)

        return img_base64

    def plot_power_map_for_turbine(self, wt_index=35, save=False, show=True):
        """
        画出单台风机（如 ID=35）在不同风速/风向下的发电功率热图，
        并在右图中高亮该风机位置。

        Parameters:
        - wt_index: int，要显示的风机编号（默认为 35）
        - save: bool，是否保存图片，默认 False
        - show: bool，是否显示图片，默认 True

        Returns:
        - str: base64 编码的图片字符串
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        self.sim_res.Power.sel(wt=wt_index).T.plot(ax=ax1)
        ax1.set_xlabel('wd [deg]')
        ax1.set_ylabel('ws [m/s]')
        ax1.set_title(f'WT {wt_index} Power Map')

        self.windTurbines.plot(self.x_pos, self.y_pos, ax=ax2)
        ax2.plot(self.x_pos[wt_index], self.y_pos[wt_index], 'or')
        ax2.set_xlabel('x [m]')
        ax2.set_ylabel('y [m]')
        ax2.set_title('Wind Farm Layout')

        plt.tight_layout()

        filename = None
        if save:
            filename = os.path.join(self.default_save_dir, f"power_map_wt{wt_index}.png")

        return self._save_or_show(fig, filename, show)

    def plot_total_aep_comparison(self, save=False, show=True):
        """
        比较风电场整体 AEP（总年发电量），考虑与不考虑尾流影响。
        显示柱状图，并计算尾流损失百分比。

        Parameters:
        - save: bool，是否保存图片，默认 False
        - show: bool，是否显示图片，默认 True

        Returns:
        - str: base64 编码的图片字符串
        """
        total_with_wake = self.sim_res.aep_with_wake.sum().item()
        total_without_wake = self.sim_res.aep_without_wake.sum().item()
        wake_loss_pct = (1 - total_with_wake / total_without_wake) * 100

        fig, ax = plt.subplots()
        ax.bar(['With Wake Loss', 'Without Wake Loss'],
               [total_with_wake, total_without_wake],
               color=['orange', 'green'])
        ax.set_ylabel("Total AEP [GWh]")
        ax.set_title(f"Total AEP Comparison\nWake Loss: {wake_loss_pct:.2f}%")
        ax.grid(True)

        filename = None
        if save:
            filename = os.path.join(self.default_save_dir, "total_aep_comparison.png")

        return self._save_or_show(fig, filename, show)

    def plot_aep_per_turbine(self, save=False, show=True):
        """
        画出每台风机的 AEP 值，分别考虑尾流和不考虑尾流。
        包含两个子图：左为 AEP 曲线，右为风机布局图。

        Parameters:
        - save: bool，是否保存图片，默认 False
        - show: bool，是否显示图片，默认 True

        Returns:
        - str: base64 编码的图片字符串
        """
        aep_w = self.sim_res.aep_with_wake.sum(dim=["wd", "ws"])
        aep_wo = self.sim_res.aep_without_wake.sum(dim=["wd", "ws"])

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        ax1.plot(self.wt_ids, aep_w, label="With Wake", marker='o')
        ax1.plot(self.wt_ids, aep_wo, label="Without Wake", marker='x')
        ax1.set_xlabel("Turbine ID")
        ax1.set_ylabel("AEP [GWh]")
        ax1.set_title("Per-Turbine AEP with vs. without Wake Loss")
        ax1.legend()
        ax1.grid(True)

        for i, wt in enumerate(self.wt_ids):
            ax1.text(wt, aep_w[i].item(), str(wt), fontsize=7, ha='right', va='bottom', rotation=45)

        self.windTurbines.plot(self.x_pos, self.y_pos, ax=ax2)
        ax2.set_title("Wind Turbine Layout")
        ax2.set_xlabel("x [m]")
        ax2.set_ylabel("y [m]")
        ax2.axis("equal")
        ax2.grid(True)

        plt.tight_layout()

        filename = None
        if save:
            filename = os.path.join(self.default_save_dir, "aep_per_turbine.png")

        return self._save_or_show(fig, filename, show)

    def plot_wake_loss_heatmap(self, save=False, show=True):
        """
        展示尾流损失热图，按风速-风向二维矩阵展开。
        结果按风机平均（mean over 'wt'）。

        Parameters:
        - save: bool，是否保存图片，默认 False
        - show: bool，是否显示图片，默认 True

        Returns:
        - str: base64 编码的图片字符串
        """
        aep_loss_pct = 100 * (1 - self.sim_res.aep_with_wake / self.sim_res.aep_without_wake)
        aep_loss_mean = aep_loss_pct.mean(dim="wt")

        fig = plt.figure(figsize=(10, 6))
        ax = fig.gca()
        aep_loss_mean.plot(cmap="Reds", ax=ax)
        ax.set_title("Wake Loss [%] vs Wind Direction and Speed")
        ax.set_xlabel("Wind Speed [m/s]")
        ax.set_ylabel("Wind Direction [°]")

        filename = None
        if save:
            filename = os.path.join(self.default_save_dir, "wake_loss_heatmap.png")

        return self._save_or_show(fig, filename, show)

    def plot_aep_vs_windspeed(self, save=False, show=True):
        """
        展示风速与整体 AEP 的关系（有尾流影响），
        展现风速在多少时发电量最高。

        Parameters:
        - save: bool，是否保存图片，默认 False
        - show: bool，是否显示图片，默认 True

        Returns:
        - str: base64 编码的图片字符串
        """
        aep_by_ws = self.sim_res.aep_with_wake.sum(dim=["wt", "wd"])

        fig, ax = plt.subplots()
        ax.plot(self.sim_res.ws, aep_by_ws, marker='o')
        ax.set_xlabel("Wind Speed [m/s]")
        ax.set_ylabel("AEP [GWh]")
        ax.set_title("Total AEP vs Wind Speed (with Wake Loss)")
        ax.grid(True)

        filename = None
        if save:
            filename = os.path.join(self.default_save_dir, "aep_vs_windspeed.png")

        return self._save_or_show(fig, filename, show)


if __name__ == "__main__":
    analysis = WindFarmAnalysis("simulation_result.nc")

    # 所有方法都同时保存图片、显示并返回 base64 编码
    base64_power_map = analysis.plot_power_map_for_turbine(wt_index=35, save=True, show=True)
    base64_aep_comparison = analysis.plot_total_aep_comparison(save=True, show=True)
    base64_aep_per_turbine = analysis.plot_aep_per_turbine(save=True, show=True)
    base64_wake_loss = analysis.plot_wake_loss_heatmap(save=True, show=True)
    base64_aep_windspeed = analysis.plot_aep_vs_windspeed(save=True, show=True)

    # 可以打印或使用这些 base64 编码的字符串
    print(f"Power map base64 length: {len(base64_power_map)}")
    print(f"AEP comparison base64 length: {len(base64_aep_comparison)}")
    print(f"AEP per turbine base64 length: {len(base64_aep_per_turbine)}")
    print(f"Wake loss heatmap base64 length: {len(base64_wake_loss)}")
    print(f"AEP vs windspeed base64 length: {len(base64_aep_windspeed)}")