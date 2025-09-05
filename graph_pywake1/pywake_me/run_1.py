import numpy as np
import xarray as xr
import os

# 默认导入，可选传入时使用
from py_wake.examples.data.hornsrev1 import Hornsrev1Site, V80
from py_wake.literature.gaussian_models import Bastankhah_PorteAgel_2014


class WindFarmSimulation:
    def __init__(self, site=None, windTurbines=None, wf_model_cls=None, wf_model_kwargs=None):
        """
        初始化风电场场景及尾流模型。

        参数：
        - site: 风场场地对象，默认 Hornsrev1Site()
        - windTurbines: 风机对象，默认 V80()
        - wf_model_cls: 尾流模型类，默认 Bastankhah_PorteAgel_2014
        - wf_model_kwargs: 尾流模型初始化参数字典，如 {'k':0.0324555}
        """
        # 设置默认场地和风机
        self.site = site if site is not None else Hornsrev1Site()
        self.windTurbines = windTurbines if windTurbines is not None else V80()
        self.x, self.y = self.site.initial_position.T

        # 设置尾流模型类和初始化参数
        self.wf_model_cls = wf_model_cls if wf_model_cls is not None else Bastankhah_PorteAgel_2014
        self.wf_model_kwargs = wf_model_kwargs if wf_model_kwargs is not None else {'k': 0.0324555}

        # 初始化尾流模型实例
        self.wf_model = self.wf_model_cls(self.site, self.windTurbines, **self.wf_model_kwargs)

        self.sim_res = None

    def run_simulation(self, wt_x=None, wt_y=None, h=None, turbine_type=0, wd=None, ws=None):
        if wt_x is None or wt_y is None:
            wt_x, wt_y = self.x, self.y

        self.sim_res = self.wf_model(wt_x, wt_y, h=h, type=turbine_type, wd=wd, ws=ws)
        return self.sim_res

    def calculate_and_add_aep(self):
        if self.sim_res is None:
            raise RuntimeError("请先调用 run_simulation 运行仿真")

        aep_with_wake = self.sim_res.aep()
        aep_without_wake = self.sim_res.aep(with_wake_loss=False)

        self.sim_res["aep_with_wake"] = aep_with_wake
        self.sim_res["aep_without_wake"] = aep_without_wake

    def save_to_netcdf(self, filename="simulation_result.nc"):
        if self.sim_res is None:
            raise RuntimeError("没有仿真结果，无法保存")

        # 获取当前脚本文件所在目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # 拼接完整路径
        full_path = os.path.join(current_dir, filename)

        self.sim_res.to_netcdf(full_path)
        print(f"仿真结果已保存到 {full_path}")
        return full_path


if __name__ == "__main__":
    sim = WindFarmSimulation()
    sim.run_simulation()
    sim.calculate_and_add_aep()
    sim.save_to_netcdf()





