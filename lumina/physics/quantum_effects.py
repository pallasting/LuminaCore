"""
量子效应和纳米尺度物理建模

实现稀土纳米晶阵列的量子效应建模，包括：
- 量子点效应和量子阱效应
- 纳米尺度几何效应
- 表面等离子体激元效应
- 量子噪声和量子干涉
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class QuantumState:
    """量子状态数据类"""

    energy_levels: torch.Tensor  # 能级结构
    wavefunctions: torch.Tensor  # 波函数
    transition_matrix: torch.Tensor  # 跃迁矩阵
    coherence_time: float  # 相干时间
    dephasing_rate: float  # 退相干率


@dataclass
class NanoGeometry:
    """纳米几何结构数据类"""

    particle_size: float  # 纳米晶粒尺寸 (nm)
    lattice_constant: float  # 晶格常数 (nm)
    surface_area: float  # 表面积 (nm²)
    volume: float  # 体积 (nm³)
    aspect_ratio: float  # 长宽比
    doping_concentration: float  # 掺杂浓度


class QuantumDotModel(nn.Module):
    """
    量子点效应建模

    模拟稀土离子在纳米晶中的量子点行为，包括：
    - 能级结构和跃迁
    - 量子限制效应
    - 表面效应
    - 温度依赖性
    """

    def __init__(
        self,
        num_energy_levels: int = 8,
        temperature: float = 300.0,  # K
        crystal_size: float = 10.0,  # nm
        rare_earth_ion: str = "Er3+",
    ):
        super(QuantumDotModel, self).__init__()

        self.num_levels = num_energy_levels
        self.temperature = temperature
        self.crystal_size = crystal_size
        self.ion_type = rare_earth_ion

        # 能级结构参数（基于不同稀土离子）
        self.energy_levels = self._initialize_energy_levels()

        # 跃迁矩阵（可学习的）
        self.transition_matrix = nn.Parameter(
            torch.randn(num_energy_levels, num_energy_levels) * 0.1
        )

        # 量子限制效应参数
        self.confinement_strength = nn.Parameter(torch.tensor(1.0))

        # 表面效应参数
        self.surface_trap_density = nn.Parameter(torch.tensor(1e12))  # cm⁻²

        # 温度相关参数
        self.thermal_broadening = nn.Parameter(torch.tensor(0.1))

        # 相干时间（基于晶体尺寸和温度）
        self.coherence_time = self._calculate_coherence_time()

    def _initialize_energy_levels(self) -> torch.Tensor:
        """初始化能级结构（基于稀土离子类型）"""
        if self.ion_type == "Er3+":
            # Er³⁺的典型能级（相对于基态，单位：cm⁻¹）
            levels = torch.tensor(
                [
                    0.0,  # ⁴I₁₅/₂
                    6500.0,  # ⁴I₁₃/₂
                    10200.0,  # ⁴I₁₁/₂
                    12400.0,  # ⁴I₉/₂
                    15300.0,  # ⁴F₉/₂
                    18500.0,  # ⁴S₃/₂
                    20500.0,  # ²H₁₁/₂
                    22100.0,  # ⁴F₇/₂
                ]
            )
        elif self.ion_type == "Tm3+":
            # Tm³⁺的典型能级
            levels = torch.tensor(
                [
                    0.0,  # ³H₆
                    5700.0,  # ³F₄
                    8200.0,  # ³H₅
                    12500.0,  # ³H₄
                    14500.0,  # ³F₃
                    21000.0,  # ¹G₄
                    26000.0,  # ³P₂
                    28000.0,  # ¹D₂
                ]
            )
        else:
            # 默认能级结构
            levels = torch.linspace(0.0, 25000.0, self.num_levels)

        return levels.float()

    def _calculate_coherence_time(self) -> float:
        """计算相干时间（基于晶体尺寸和温度）"""
        # 相干时间与晶体尺寸成正比，与温度成反比
        size_factor = self.crystal_size / 10.0  # 归一化到10nm
        temp_factor = 300.0 / self.temperature  # 归一化到室温

        # 典型相干时间范围：纳秒到微秒
        base_coherence = 100e-9  # 100ns
        coherence_time = base_coherence * size_factor * temp_factor

        return max(1e-9, min(1e-6, coherence_time))  # 限制在1ns到1μs之间

    def forward(
        self, input_power: torch.Tensor, wavelength: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播：计算量子效应对光信号的影响

        Args:
            input_power: 输入光功率
            wavelength: 波长 (nm)

        Returns:
            量子效应处理结果
        """
        # 计算量子限制效应
        confinement_effect = self._quantum_confinement_effect(wavelength)

        # 计算表面效应
        surface_effect = self._surface_trap_effect(input_power)

        # 计算温度效应
        thermal_effect = self._thermal_broadening_effect()

        # 计算总的量子效应
        quantum_gain = confinement_effect * (1 + surface_effect) * thermal_effect

        # 计算相干性损失
        coherence_loss = self._calculate_coherence_loss(input_power)

        # 计算量子噪声
        quantum_noise = self._calculate_quantum_noise(input_power)

        return {
            "quantum_gain": quantum_gain,
            "coherence_loss": coherence_loss,
            "quantum_noise": quantum_noise,
            "effective_power": input_power * quantum_gain * (1 - coherence_loss),
        }

    def _quantum_confinement_effect(self, wavelength: torch.Tensor) -> torch.Tensor:
        """量子限制效应"""
        # 量子限制导致的能级分裂和光学性质变化
        confinement_energy = 1240.0 / (self.crystal_size * wavelength)  # eV
        confinement_factor = 1.0 + self.confinement_strength * confinement_energy

        return torch.clamp(confinement_factor, 0.5, 2.0)

    def _surface_trap_effect(self, input_power: torch.Tensor) -> torch.Tensor:
        """表面陷阱效应"""
        # 表面缺陷导致的非辐射跃迁
        trap_probability = self.surface_trap_density * 1e-14  # 归一化
        surface_loss = trap_probability * torch.sqrt(input_power + 1e-10)

        return torch.clamp(surface_loss, 0.0, 0.5)

    def _thermal_broadening_effect(self) -> torch.Tensor:
        """温度展宽效应"""
        # 温度导致的谱线展宽
        kT = 4.14e-21 * self.temperature  # 热能 (eV)
        broadening_factor = 1.0 - self.thermal_broadening * kT

        return torch.clamp(broadening_factor, 0.5, 1.0)

    def _calculate_coherence_loss(self, input_power: torch.Tensor) -> torch.Tensor:
        """计算相干性损失"""
        # 高功率导致的退相干
        power_factor = torch.sqrt(input_power) / 100.0  # 归一化
        coherence_loss = 1.0 - torch.exp(-power_factor / self.coherence_time)

        return torch.clamp(coherence_loss, 0.0, 0.9)

    def _calculate_quantum_noise(self, input_power: torch.Tensor) -> torch.Tensor:
        """计算量子噪声"""
        # 量子限制导致的噪声
        photon_number = input_power * self.crystal_size * 1e-9  # 近似光子数
        quantum_noise_std = torch.sqrt(photon_number + 1e-10)

        return torch.randn_like(input_power) * quantum_noise_std * 0.01

    def get_quantum_state(self) -> QuantumState:
        """获取当前量子状态"""
        # 计算波函数（简化模型）
        wavefunctions = torch.randn(
            self.num_levels, self.num_levels, dtype=torch.complex64
        )

        return QuantumState(
            energy_levels=self.energy_levels,
            wavefunctions=wavefunctions,
            transition_matrix=self.transition_matrix,
            coherence_time=self.coherence_time,
            dephasing_rate=1.0 / self.coherence_time,
        )


class SurfacePlasmonModel(nn.Module):
    """
    表面等离子体激元效应建模

    模拟金属纳米结构与稀土纳米晶的耦合效应：
    - 局域场增强
    - 表面等离子体共振
    - 能量转移增强
    """

    def __init__(
        self,
        metal_type: str = "Au",
        nanoparticle_size: float = 50.0,  # nm
        gap_distance: float = 5.0,  # nm
        dielectric_constant: float = 2.25,  # 基底介电常数
    ):
        super(SurfacePlasmonModel, self).__init__()

        self.metal_type = metal_type
        self.particle_size = nanoparticle_size
        self.gap_distance = gap_distance
        self.dielectric_constant = dielectric_constant

        # 表面等离子体共振参数
        self.plasmon_frequency = self._get_plasmon_frequency()
        self.damping_rate = nn.Parameter(torch.tensor(0.1))

        # 局域场增强因子
        self.field_enhancement = nn.Parameter(torch.tensor(10.0))

        # 耦合强度
        self.coupling_strength = nn.Parameter(torch.tensor(0.5))

    def _get_plasmon_frequency(self) -> float:
        """获取等离子体共振频率（基于金属类型）"""
        if self.metal_type == "Au":
            return 2.4e15  # Hz (约620nm)
        elif self.metal_type == "Ag":
            return 2.8e15  # Hz (约540nm)
        elif self.metal_type == "Cu":
            return 2.1e15  # Hz (约710nm)
        else:
            return 2.4e15  # 默认值

    def forward(
        self, input_field: torch.Tensor, wavelength: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        计算表面等离子体激元效应

        Args:
            input_field: 输入电场
            wavelength: 波长

        Returns:
            表面等离子体效应结果
        """
        # 计算共振条件
        frequency = 3e8 / (wavelength * 1e-9)  # Hz
        detuning = (
            torch.abs(frequency - self.plasmon_frequency) / self.plasmon_frequency
        )

        # 计算局域场增强
        local_field = self._calculate_local_field_enhancement(detuning)

        # 计算能量转移效率
        energy_transfer = self._calculate_energy_transfer(input_field, detuning)

        # 计算散射和吸收
        scattering = self._calculate_scattering(input_field, detuning)
        absorption = self._calculate_absorption(input_field, detuning)

        # 计算总的场增强
        total_enhancement = local_field * (1 + energy_transfer)

        return {
            "field_enhancement": total_enhancement,
            "energy_transfer": energy_transfer,
            "scattering": scattering,
            "absorption": absorption,
            "plasmon_resonance": 1.0 / (1.0 + detuning**2),
        }

    def _calculate_local_field_enhancement(
        self, detuning: torch.Tensor
    ) -> torch.Tensor:
        """计算局域场增强"""
        # 洛伦兹型共振
        resonance_factor = 1.0 / (1.0 + (2 * detuning / self.damping_rate) ** 2)
        enhancement = self.field_enhancement * resonance_factor

        return enhancement

    def _calculate_energy_transfer(
        self, input_field: torch.Tensor, detuning: torch.Tensor
    ) -> torch.Tensor:
        """计算能量转移效率"""
        # 福斯特共振能量转移
        transfer_rate = self.coupling_strength / (1.0 + detuning**2)
        field_strength = torch.abs(input_field)

        # 能量转移与场强成正比
        energy_transfer = transfer_rate * torch.sqrt(field_strength + 1e-10)

        return torch.clamp(energy_transfer, 0.0, 0.9)

    def _calculate_scattering(
        self, input_field: torch.Tensor, detuning: torch.Tensor
    ) -> torch.Tensor:
        """计算散射截面"""
        # 基于准静态近似
        size_factor = (self.particle_size / 100.0) ** 3  # 归一化
        scattering_cross_section = size_factor * (1.0 + detuning**2)

        return scattering_cross_section * torch.abs(input_field)

    def _calculate_absorption(
        self, input_field: torch.Tensor, detuning: torch.Tensor
    ) -> torch.Tensor:
        """计算吸收截面"""
        # 基于等离子体共振
        absorption_cross_section = self.damping_rate / (1.0 + detuning**2)

        return absorption_cross_section * torch.abs(input_field)


class NanoGeometryModel(nn.Module):
    """
    纳米结构几何效应建模

    模拟纳米晶阵列的几何效应：
    - 尺寸效应
    - 形状效应
    - 排列效应
    - 近场耦合
    """

    def __init__(
        self,
        array_size: Tuple[int, int] = (10, 10),
        particle_spacing: float = 20.0,  # nm
        particle_shape: str = "spherical",
    ):
        super(NanoGeometryModel, self).__init__()

        self.array_size = array_size
        self.particle_spacing = particle_spacing
        self.particle_shape = particle_shape

        # 几何参数
        self.aspect_ratio = nn.Parameter(torch.tensor(1.0))  # 长宽比
        self.surface_roughness = nn.Parameter(torch.tensor(0.1))  # 表面粗糙度

        # 耦合参数
        self.near_field_coupling = nn.Parameter(torch.tensor(0.2))
        self.far_field_coupling = nn.Parameter(torch.tensor(0.05))

        # 形状因子
        self.shape_factor = self._get_shape_factor()

    def _get_shape_factor(self) -> float:
        """获取形状因子"""
        if self.particle_shape == "spherical":
            return 1.0
        elif self.particle_shape == "rod":
            return 1.5
        elif self.particle_shape == "plate":
            return 0.8
        else:
            return 1.0

    def forward(self, input_signal: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        计算几何效应

        Args:
            input_signal: 输入信号

        Returns:
            几何效应结果
        """
        # 计算尺寸效应
        size_effect = self._calculate_size_effect(input_signal)

        # 计算形状效应
        shape_effect = self._calculate_shape_effect(input_signal)

        # 计算排列效应
        array_effect = self._calculate_array_effect(input_signal)

        # 计算耦合效应
        coupling_effect = self._calculate_coupling_effect(input_signal)

        # 综合效应
        total_effect = size_effect * shape_effect * array_effect * (1 + coupling_effect)

        return {
            "size_effect": size_effect,
            "shape_effect": shape_effect,
            "array_effect": array_effect,
            "coupling_effect": coupling_effect,
            "total_effect": total_effect,
        }

    def _calculate_size_effect(self, input_signal: torch.Tensor) -> torch.Tensor:
        """计算尺寸效应"""
        # 量子尺寸效应导致的光学性质变化
        signal_strength = torch.abs(input_signal)
        size_factor = 1.0 + 0.1 * torch.log(signal_strength + 1e-10)

        return torch.clamp(size_factor, 0.5, 2.0)

    def _calculate_shape_effect(self, input_signal: torch.Tensor) -> torch.Tensor:
        """计算形状效应"""
        # 不同形状的散射特性
        shape_modulation = self.shape_factor * (1 + self.aspect_ratio * 0.1)

        return torch.full_like(input_signal, shape_modulation, dtype=input_signal.dtype)

    def _calculate_array_effect(self, input_signal: torch.Tensor) -> torch.Tensor:
        """计算阵列排列效应"""
        # 周期性排列导致的衍射效应
        array_factor = 1.0 + 0.05 * torch.sin(
            2 * np.pi * torch.arange(len(input_signal)) / 10.0
        )

        return array_factor.to(input_signal.device)

    def _calculate_coupling_effect(self, input_signal: torch.Tensor) -> torch.Tensor:
        """计算近场耦合效应"""
        # 纳米粒子间的近场耦合
        coupling_strength = self.near_field_coupling * torch.exp(
            -self.particle_spacing / 50.0
        )

        return coupling_strength


class ThermalOpticsModel(nn.Module):
    """
    热光效应建模

    模拟温度对光学性质的影响：
    - 热透镜效应
    - 热膨胀
    - 温度依赖的折射率变化
    - 热噪声
    """

    def __init__(
        self,
        thermal_conductivity: float = 1.0,  # W/m·K
        thermal_expansion_coeff: float = 1e-5,  # K⁻¹
        dn_dT: float = 1e-5,  # K⁻¹ (折射率温度系数)
    ):
        super(ThermalOpticsModel, self).__init__()

        self.thermal_conductivity = thermal_conductivity
        self.thermal_expansion_coeff = thermal_expansion_coeff
        self.dn_dT = dn_dT

        # 热参数
        self.heat_capacity = nn.Parameter(torch.tensor(0.7))  # J/g·K
        self.thermal_time_constant = nn.Parameter(torch.tensor(1e-3))  # s

        # 热噪声参数
        self.thermal_noise_amplitude = nn.Parameter(torch.tensor(1e-6))

    def forward(
        self,
        input_field: torch.Tensor,
        temperature: torch.Tensor,
        time_step: float = 1e-6,
    ) -> Dict[str, torch.Tensor]:
        """
        计算热光效应

        Args:
            input_field: 输入电场
            temperature: 温度分布
            time_step: 时间步长

        Returns:
            热光效应结果
        """
        # 计算热透镜效应
        thermal_lens = self._calculate_thermal_lens(input_field, temperature)

        # 计算热膨胀效应
        thermal_expansion = self._calculate_thermal_expansion(temperature)

        # 计算折射率变化
        refractive_index_change = self._calculate_refractive_change(temperature)

        # 计算热噪声
        thermal_noise = self._calculate_thermal_noise(input_field, temperature)

        # 计算相位变化
        phase_change = thermal_lens + refractive_index_change

        return {
            "thermal_lens": thermal_lens,
            "thermal_expansion": thermal_expansion,
            "refractive_change": refractive_index_change,
            "thermal_noise": thermal_noise,
            "phase_change": phase_change,
        }

    def _calculate_thermal_lens(
        self, input_field: torch.Tensor, temperature: torch.Tensor
    ) -> torch.Tensor:
        """计算热透镜效应"""
        # 温度梯度导致的折射率梯度
        power_density = torch.abs(input_field) ** 2
        temperature_rise = power_density * self.thermal_time_constant

        # 热透镜相位
        lens_phase = temperature_rise * self.dn_dT * 2 * np.pi

        return lens_phase

    def _calculate_thermal_expansion(self, temperature: torch.Tensor) -> torch.Tensor:
        """计算热膨胀效应"""
        # 热膨胀导致的光学路径变化
        expansion_factor = temperature * self.thermal_expansion_coeff
        path_change = expansion_factor * 2 * np.pi  # 转换为相位

        return path_change

    def _calculate_refractive_change(self, temperature: torch.Tensor) -> torch.Tensor:
        """计算折射率温度变化"""
        # dn/dT效应
        refractive_change = temperature * self.dn_dT * 2 * np.pi

        return refractive_change

    def _calculate_thermal_noise(
        self, input_field: torch.Tensor, temperature: torch.Tensor
    ) -> torch.Tensor:
        """计算热噪声"""
        # 热起伏导致的相位噪声
        thermal_fluctuation = (
            torch.randn_like(input_field) * self.thermal_noise_amplitude
        )
        temperature_factor = torch.sqrt(temperature / 300.0)  # 温度归一化

        return thermal_fluctuation * temperature_factor


class AdvancedPhysicsEngine:
    """
    高级物理效应引擎

    集成所有物理效应模型，提供统一的接口
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        # 初始化各个物理模型
        self.quantum_model = QuantumDotModel(
            num_energy_levels=config.get("num_energy_levels", 8),
            temperature=config.get("temperature", 300.0),
            crystal_size=config.get("crystal_size", 10.0),
            rare_earth_ion=config.get("rare_earth_ion", "Er3+"),
        )

        self.plasmon_model = SurfacePlasmonModel(
            metal_type=config.get("metal_type", "Au"),
            nanoparticle_size=config.get("nanoparticle_size", 50.0),
            gap_distance=config.get("gap_distance", 5.0),
        )

        self.geometry_model = NanoGeometryModel(
            array_size=config.get("array_size", (10, 10)),
            particle_spacing=config.get("particle_spacing", 20.0),
            particle_shape=config.get("particle_shape", "spherical"),
        )

        self.thermal_model = ThermalOpticsModel(
            thermal_conductivity=config.get("thermal_conductivity", 1.0),
            dn_dT=config.get("dn_dT", 1e-5),
        )

        # 效应权重
        self.effect_weights = {
            "quantum": config.get("quantum_weight", 1.0),
            "plasmon": config.get("plasmon_weight", 0.8),
            "geometry": config.get("geometry_weight", 0.6),
            "thermal": config.get("thermal_weight", 0.4),
        }

    def simulate_complete_physics(
        self,
        input_field: torch.Tensor,
        wavelength: torch.Tensor,
        temperature: torch.Tensor,
        time_step: float = 1e-6,
    ) -> Dict[str, Any]:
        """
        模拟完整的物理效应

        Args:
            input_field: 输入电场
            wavelength: 波长
            temperature: 温度
            time_step: 时间步长

        Returns:
            完整的物理效应结果
        """
        results = {}

        # 量子效应
        if self.effect_weights["quantum"] > 0:
            quantum_results = self.quantum_model(input_field, wavelength)
            results.update({f"quantum_{k}": v for k, v in quantum_results.items()})

        # 表面等离子体效应
        if self.effect_weights["plasmon"] > 0:
            plasmon_results = self.plasmon_model(input_field, wavelength)
            results.update({f"plasmon_{k}": v for k, v in plasmon_results.items()})

        # 几何效应
        if self.effect_weights["geometry"] > 0:
            geometry_results = self.geometry_model(input_field)
            results.update({f"geometry_{k}": v for k, v in geometry_results.items()})

        # 热光效应
        if self.effect_weights["thermal"] > 0:
            thermal_results = self.thermal_model(input_field, temperature, time_step)
            results.update({f"thermal_{k}": v for k, v in thermal_results.items()})

        # 计算综合效应
        total_gain = self._calculate_total_gain(results)
        total_phase = self._calculate_total_phase(results)
        total_noise = self._calculate_total_noise(results)

        results.update(
            {
                "total_gain": total_gain,
                "total_phase": total_phase,
                "total_noise": total_noise,
                "final_field": input_field * total_gain * torch.exp(1j * total_phase)
                + total_noise,
            }
        )

        return results

    def _calculate_total_gain(self, results: Dict[str, torch.Tensor]) -> torch.Tensor:
        """计算总增益"""
        gains = []

        if "quantum_quantum_gain" in results:
            gains.append(
                results["quantum_quantum_gain"] * self.effect_weights["quantum"]
            )

        if "plasmon_field_enhancement" in results:
            gains.append(
                results["plasmon_field_enhancement"] * self.effect_weights["plasmon"]
            )

        if "geometry_total_effect" in results:
            gains.append(
                results["geometry_total_effect"] * self.effect_weights["geometry"]
            )

        if gains:
            total_gain = torch.stack(gains).mean(dim=0)
        else:
            total_gain = torch.ones_like(list(results.values())[0])

        return total_gain

    def _calculate_total_phase(self, results: Dict[str, torch.Tensor]) -> torch.Tensor:
        """计算总相位"""
        phases = []

        if "thermal_phase_change" in results:
            phases.append(
                results["thermal_phase_change"] * self.effect_weights["thermal"]
            )

        if phases:
            total_phase = torch.stack(phases).sum(dim=0)
        else:
            total_phase = torch.zeros_like(list(results.values())[0])

        return total_phase

    def _calculate_total_noise(self, results: Dict[str, torch.Tensor]) -> torch.Tensor:
        """计算总噪声"""
        noises = []

        if "quantum_quantum_noise" in results:
            noises.append(
                results["quantum_quantum_noise"] * self.effect_weights["quantum"]
            )

        if "thermal_thermal_noise" in results:
            noises.append(
                results["thermal_thermal_noise"] * self.effect_weights["thermal"]
            )

        if noises:
            total_noise = torch.stack(noises).sum(dim=0)
        else:
            total_noise = torch.zeros_like(list(results.values())[0])

        return total_noise

    def get_model_parameters(self) -> Dict[str, Any]:
        """获取所有模型参数"""
        return {
            "quantum_state": self.quantum_model.get_quantum_state(),
            "plasmon_params": {
                "frequency": self.plasmon_model.plasmon_frequency,
                "damping": self.plasmon_model.damping_rate.item(),
                "enhancement": self.plasmon_model.field_enhancement.item(),
            },
            "geometry_params": {
                "array_size": self.geometry_model.array_size,
                "spacing": self.geometry_model.particle_spacing,
                "shape": self.geometry_model.particle_shape,
            },
            "thermal_params": {
                "conductivity": self.thermal_model.thermal_conductivity,
                "expansion_coeff": self.thermal_model.thermal_expansion_coeff,
                "dn_dT": self.thermal_model.dn_dT,
            },
            "effect_weights": self.effect_weights,
        }
