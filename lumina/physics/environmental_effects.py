"""
环境因素影响建模

模拟各种环境因素对光子芯片性能的影响：
- 温度和湿度变化
- 机械振动和应力
- 电磁干扰
- 光照和辐射效应
- 材料老化
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


@dataclass
class EnvironmentalConditions:
    """环境条件数据类"""

    temperature: float  # °C
    humidity: float  # %
    vibration_amplitude: float  # m
    vibration_frequency: float  # Hz
    emi_strength: float  # V/m
    radiation_dose: float  # Gy
    illumination: float  # lux
    aging_time: float  # hours


@dataclass
class EnvironmentalImpact:
    """环境影响数据类"""

    performance_degradation: Dict[str, float]
    noise_increase: Dict[str, float]
    reliability_reduction: float
    lifetime_reduction: float
    mitigation_strategies: List[str]


class ThermalEnvironmentalModel(nn.Module):
    """
    热环境效应建模

    包括温度梯度、热应力、热噪声等效应
    """

    def __init__(self, chip_area: float = 1e-6, thermal_conductivity: float = 1.5):
        super(ThermalEnvironmentalModel, self).__init__()

        self.chip_area = chip_area  # m²
        self.thermal_conductivity = thermal_conductivity  # W/m·K

        # 热参数
        self.heat_capacity = nn.Parameter(torch.tensor(0.7))  # J/g·K
        self.thermal_expansion_coeff = nn.Parameter(torch.tensor(1e-5))  # K⁻¹
        self.temperature_gradient_sensitivity = nn.Parameter(torch.tensor(0.1))

        # 热噪声参数
        self.johnson_noise_coeff = nn.Parameter(torch.tensor(1e-9))  # 约翰逊噪声系数
        self.thermal_fluctuation_coeff = nn.Parameter(torch.tensor(1e-6))

    def forward(
        self, temperature: float, temperature_gradient: float, time_step: float = 1e-3
    ) -> Dict[str, torch.Tensor]:
        """
        计算热环境效应

        Args:
            temperature: 温度 (°C)
            temperature_gradient: 温度梯度 (K/m)
            time_step: 时间步长

        Returns:
            热环境效应结果
        """
        # 热应力效应
        thermal_stress = self._calculate_thermal_stress(
            temperature, temperature_gradient
        )

        # 热噪声
        thermal_noise = self._calculate_thermal_noise(temperature)

        # 温度漂移
        temperature_drift = self._calculate_temperature_drift(
            temperature, temperature_gradient, time_step
        )

        # 热可靠性下降
        reliability_degradation = self._calculate_reliability_degradation(temperature)

        return {
            "thermal_stress": thermal_stress,
            "thermal_noise": thermal_noise,
            "temperature_drift": temperature_drift,
            "reliability_degradation": reliability_degradation,
            "effective_temperature": temperature + temperature_drift,
        }

    def _calculate_thermal_stress(
        self, temperature: float, gradient: float
    ) -> torch.Tensor:
        """计算热应力"""
        # 热膨胀导致的应力
        stress = self.thermal_expansion_coeff * torch.tensor(abs(temperature - 25.0), dtype=torch.float32) * self.chip_area
        stress += self.temperature_gradient_sensitivity * torch.tensor(gradient, dtype=torch.float32)

        return stress

    def _calculate_thermal_noise(self, temperature: float) -> torch.Tensor:
        """计算热噪声"""
        # 约翰逊噪声（与温度成正比的噪声）
        johnson_noise = self.johnson_noise_coeff * torch.sqrt(torch.tensor(temperature + 273.15, dtype=torch.float32))

        # 热起伏噪声（温度偏离25°C的影响）
        thermal_fluctuation = self.thermal_fluctuation_coeff * torch.tensor(temperature - 25.0, dtype=torch.float32)

        total_noise = johnson_noise + thermal_fluctuation

        return total_noise

    def _calculate_temperature_drift(
        self, temperature: float, gradient: float, time_step: float
    ) -> torch.Tensor:
        """计算温度漂移"""
        # 热传导导致的温度变化
        drift_rate = (
            gradient * self.thermal_conductivity / (self.heat_capacity * self.chip_area)
        )
        drift = drift_rate * time_step

        return torch.tensor(drift, dtype=torch.float32)

    def _calculate_reliability_degradation(self, temperature: float) -> torch.Tensor:
        """计算可靠性下降"""
        # Arrhenius模型：可靠性随温度指数下降
        activation_energy = 0.5  # eV (假设值)
        k_boltzmann = 8.617e-5  # eV/K

        degradation_rate = torch.exp(
            -activation_energy / (k_boltzmann * (temperature + 273.15))
        )
        degradation = 1.0 - degradation_rate

        return degradation


class HumidityEnvironmentalModel(nn.Module):
    """
    湿度环境效应建模

    包括水分吸收、表面导电性变化、腐蚀等效应
    """

    def __init__(
        self, surface_area: float = 1e-6, material_permeability: float = 1e-10
    ):
        super(HumidityEnvironmentalModel, self).__init__()

        self.surface_area = surface_area  # m²
        self.material_permeability = material_permeability  # mol/m·s·Pa

        # 湿度参数
        self.moisture_absorption_coeff = nn.Parameter(torch.tensor(0.01))
        self.surface_conductivity_coeff = nn.Parameter(torch.tensor(1e-8))
        self.corrosion_rate_coeff = nn.Parameter(torch.tensor(1e-10))

    def forward(
        self, humidity: float, temperature: float, exposure_time: float
    ) -> Dict[str, torch.Tensor]:
        """
        计算湿度环境效应

        Args:
            humidity: 相对湿度 (%)
            temperature: 温度 (°C)
            exposure_time: 暴露时间 (hours)

        Returns:
            湿度环境效应结果
        """
        # 水分吸收
        moisture_absorption = self._calculate_moisture_absorption(
            humidity, temperature, exposure_time
        )

        # 表面导电性变化
        surface_conductivity = self._calculate_surface_conductivity(
            humidity, temperature
        )

        # 腐蚀效应
        corrosion_effect = self._calculate_corrosion_effect(
            humidity, temperature, exposure_time
        )

        # 绝缘性能下降
        insulation_degradation = self._calculate_insulation_degradation(
            moisture_absorption
        )

        return {
            "moisture_absorption": moisture_absorption,
            "surface_conductivity": surface_conductivity,
            "corrosion_effect": corrosion_effect,
            "insulation_degradation": insulation_degradation,
        }

    def _calculate_moisture_absorption(
        self, humidity: float, temperature: float, exposure_time: float
    ) -> torch.Tensor:
        """计算水分吸收"""
        # 基于Fick定律的扩散模型
        diffusion_coeff = self.material_permeability * torch.exp(
            -0.5 / torch.tensor(temperature + 273.15, dtype=torch.float32)
        )
        absorption = (
            diffusion_coeff * torch.tensor(humidity, dtype=torch.float32) * torch.sqrt(torch.tensor(exposure_time, dtype=torch.float32)) * self.surface_area
        )

        return absorption * self.moisture_absorption_coeff

    def _calculate_surface_conductivity(
        self, humidity: float, temperature: float
    ) -> torch.Tensor:
        """计算表面导电性"""
        # 湿度导致的表面导电性增加
        conductivity = (
            self.surface_conductivity_coeff * torch.tensor(humidity, dtype=torch.float32) * torch.exp(torch.tensor(temperature / 50.0, dtype=torch.float32))
        )

        return conductivity

    def _calculate_corrosion_effect(
        self, humidity: float, temperature: float, exposure_time: float
    ) -> torch.Tensor:
        """计算腐蚀效应"""
        # 湿度加速腐蚀
        corrosion_rate = (
            self.corrosion_rate_coeff * humidity * torch.exp(torch.tensor(temperature / 30.0, dtype=torch.float32))
        )
        corrosion = corrosion_rate * exposure_time

        return torch.clamp(corrosion, 0.0, 1.0)

    def _calculate_insulation_degradation(
        self, moisture_absorption: torch.Tensor
    ) -> torch.Tensor:
        """计算绝缘性能下降"""
        # 水分导致绝缘电阻下降
        degradation = 1.0 - torch.exp(-moisture_absorption / 0.1)

        return torch.clamp(degradation, 0.0, 1.0)


class VibrationEnvironmentalModel(nn.Module):
    """
    振动环境效应建模

    包括机械振动导致的光路变化、应力效应等
    """

    def __init__(
        self, chip_dimensions: Tuple[float, float, float] = (1e-3, 1e-3, 1e-4)
    ):
        super(VibrationEnvironmentalModel, self).__init__()

        self.chip_dimensions = chip_dimensions  # (length, width, height) in meters

        # 振动参数
        self.resonance_frequency = nn.Parameter(torch.tensor(1000.0))  # Hz
        self.damping_coefficient = nn.Parameter(torch.tensor(0.01))
        self.displacement_sensitivity = nn.Parameter(torch.tensor(1e-6))  # m⁻¹

    def forward(
        self,
        vibration_amplitude: float,
        vibration_frequency: float,
        time_step: float = 1e-3,
    ) -> Dict[str, torch.Tensor]:
        """
        计算振动环境效应

        Args:
            vibration_amplitude: 振动振幅 (m)
            vibration_frequency: 振动频率 (Hz)
            time_step: 时间步长

        Returns:
            振动环境效应结果
        """
        # 共振效应
        resonance_effect = self._calculate_resonance_effect(
            vibration_amplitude, vibration_frequency
        )

        # 位移效应
        displacement_effect = self._calculate_displacement_effect(
            vibration_amplitude, vibration_frequency
        )

        # 应力效应
        stress_effect = self._calculate_stress_effect(
            vibration_amplitude, vibration_frequency
        )

        # 相位噪声
        phase_noise = self._calculate_phase_noise(
            vibration_amplitude, vibration_frequency, time_step
        )

        return {
            "resonance_effect": resonance_effect,
            "displacement_effect": displacement_effect,
            "stress_effect": stress_effect,
            "phase_noise": phase_noise,
        }

    def _calculate_resonance_effect(
        self, amplitude: float, frequency: float
    ) -> torch.Tensor:
        """计算共振效应"""
        # 洛伦兹共振
        detuning = (
            abs(frequency - self.resonance_frequency.item())
            / self.resonance_frequency.item()
        )
        resonance_amplitude = amplitude / (
            1.0 + (2 * detuning / self.damping_coefficient.item()) ** 2
        )

        return torch.tensor(resonance_amplitude, dtype=torch.float32)

    def _calculate_displacement_effect(
        self, amplitude: float, frequency: float
    ) -> torch.Tensor:
        """计算位移效应"""
        # 振动导致的光路长度变化
        displacement = amplitude * self.displacement_sensitivity.item()
        path_change = 2 * displacement  # 双程效应

        return torch.tensor(path_change, dtype=torch.float32)

    def _calculate_stress_effect(
        self, amplitude: float, frequency: float
    ) -> torch.Tensor:
        """计算应力效应"""
        # 振动应力
        acceleration = (2 * torch.tensor(np.pi, dtype=torch.float32) * frequency) ** 2 * amplitude
        stress = acceleration * 1000  # 简化的应力计算

        return torch.tensor(stress, dtype=torch.float32)

    def _calculate_phase_noise(
        self, amplitude: float, frequency: float, time_step: float
    ) -> torch.Tensor:
        """计算相位噪声"""
        # 振动导致的相位起伏
        phase_fluctuation = 2 * np.pi * amplitude / (3e8 / frequency)  # 波长相关
        phase_noise = phase_fluctuation * np.sqrt(time_step)

        return torch.tensor(phase_noise, dtype=torch.float32)


class EMIEnvironmentalModel(nn.Module):
    """
    电磁干扰环境效应建模

    包括电磁场耦合、感应噪声等效应
    """

    def __init__(self, antenna_area: float = 1e-8, coupling_coefficient: float = 1e-12):
        super(EMIEnvironmentalModel, self).__init__()

        self.antenna_area = antenna_area  # m²
        self.coupling_coefficient = coupling_coefficient

        # EMI参数
        self.induced_voltage_coeff = nn.Parameter(torch.tensor(1e-6))
        self.crosstalk_induction_coeff = nn.Parameter(torch.tensor(1e-9))

    def forward(
        self, emi_strength: float, frequency: float, distance: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        计算电磁干扰效应

        Args:
            emi_strength: EMI场强 (V/m)
            frequency: EMI频率 (Hz)
            distance: 干扰源距离 (m)

        Returns:
            EMI环境效应结果
        """
        # 感应电压
        induced_voltage = self._calculate_induced_voltage(
            emi_strength, frequency, distance
        )

        # 耦合噪声
        coupling_noise = self._calculate_coupling_noise(emi_strength, frequency)

        # 串扰感应
        crosstalk_induction = self._calculate_crosstalk_induction(
            emi_strength, frequency
        )

        # 屏蔽效果
        shielding_effect = self._calculate_shielding_effect(distance)

        return {
            "induced_voltage": induced_voltage,
            "coupling_noise": coupling_noise,
            "crosstalk_induction": crosstalk_induction,
            "shielding_effect": shielding_effect,
        }

    def _calculate_induced_voltage(
        self, emi_strength: float, frequency: float, distance: float
    ) -> torch.Tensor:
        """计算感应电压"""
        # 基于天线效应
        induced_voltage = self.induced_voltage_coeff * emi_strength * self.antenna_area
        induced_voltage *= 1.0 / distance  # 距离衰减

        return torch.tensor(induced_voltage, dtype=torch.float32)

    def _calculate_coupling_noise(
        self, emi_strength: float, frequency: float
    ) -> torch.Tensor:
        """计算耦合噪声"""
        # 电磁耦合噪声
        coupling_noise = self.coupling_coefficient * emi_strength * frequency

        return torch.tensor(coupling_noise, dtype=torch.float32)

    def _calculate_crosstalk_induction(
        self, emi_strength: float, frequency: float
    ) -> torch.Tensor:
        """计算串扰感应"""
        # EMI导致的通道间串扰
        crosstalk = self.crosstalk_induction_coeff * torch.tensor(emi_strength, dtype=torch.float32) * torch.sqrt(torch.tensor(frequency, dtype=torch.float32))

        return crosstalk

    def _calculate_shielding_effect(self, distance: float) -> torch.Tensor:
        """计算屏蔽效果"""
        # 距离相关的屏蔽衰减
        shielding = torch.exp(-distance / 0.1)  # 经验衰减模型

        return shielding


class RadiationEnvironmentalModel(nn.Module):
    """
    辐射环境效应建模

    包括电离辐射导致的材料损伤、暗电流增加等效应
    """

    def __init__(self, sensitive_volume: float = 1e-12):
        super(RadiationEnvironmentalModel, self).__init__()

        self.sensitive_volume = sensitive_volume  # m³

        # 辐射参数
        self.ionization_coeff = nn.Parameter(torch.tensor(1e-6))  # 对/Gy
        self.displacement_coeff = nn.Parameter(torch.tensor(1e-8))  # dpa/Gy
        self.dark_current_coeff = nn.Parameter(torch.tensor(1e-12))  # A/Gy

    def forward(
        self, radiation_dose: float, dose_rate: float, exposure_time: float
    ) -> Dict[str, torch.Tensor]:
        """
        计算辐射环境效应

        Args:
            radiation_dose: 辐射剂量 (Gy)
            dose_rate: 剂量率 (Gy/s)
            exposure_time: 暴露时间 (s)

        Returns:
            辐射环境效应结果
        """
        # 电离损伤
        ionization_damage = self._calculate_ionization_damage(radiation_dose, dose_rate)

        # 位移损伤
        displacement_damage = self._calculate_displacement_damage(radiation_dose)

        # 暗电流增加
        dark_current_increase = self._calculate_dark_current_increase(
            radiation_dose, exposure_time
        )

        # 材料退化
        material_degradation = self._calculate_material_degradation(
            radiation_dose, exposure_time
        )

        return {
            "ionization_damage": ionization_damage,
            "displacement_damage": displacement_damage,
            "dark_current_increase": dark_current_increase,
            "material_degradation": material_degradation,
        }

    def _calculate_ionization_damage(
        self, dose: float, dose_rate: float
    ) -> torch.Tensor:
        """计算电离损伤"""
        # 电离导致的载流子产生
        ionization_events = self.ionization_coeff * dose * self.sensitive_volume
        ionization_damage = ionization_events * dose_rate

        return torch.tensor(ionization_damage, dtype=torch.float32)

    def _calculate_displacement_damage(self, dose: float) -> torch.Tensor:
        """计算位移损伤"""
        # 原子位移导致的晶格缺陷
        displacement_events = self.displacement_coeff * dose * self.sensitive_volume

        return torch.tensor(displacement_events, dtype=torch.float32)

    def _calculate_dark_current_increase(
        self, dose: float, exposure_time: float
    ) -> torch.Tensor:
        """计算暗电流增加"""
        # 辐射诱导的暗电流
        dark_current = self.dark_current_coeff * torch.tensor(dose, dtype=torch.float32) * torch.sqrt(torch.tensor(exposure_time, dtype=torch.float32))

        return dark_current

    def _calculate_material_degradation(
        self, dose: float, exposure_time: float
    ) -> torch.Tensor:
        """计算材料退化"""
        # 综合退化模型
        degradation = 1.0 - torch.exp(torch.tensor(-dose * exposure_time / 1e6, dtype=torch.float32))

        return torch.clamp(degradation, 0.0, 1.0)


class ComprehensiveEnvironmentalModel:
    """
    综合环境效应模型

    集成所有环境因素的影响
    """

    def __init__(self, chip_parameters: Dict[str, Any]):
        self.chip_params = chip_parameters

        # 初始化各个环境模型
        self.thermal_model = ThermalEnvironmentalModel(
            chip_area=chip_parameters.get("chip_area", 1e-6),
            thermal_conductivity=chip_parameters.get("thermal_conductivity", 1.5),
        )

        self.humidity_model = HumidityEnvironmentalModel(
            surface_area=chip_parameters.get("surface_area", 1e-6),
            material_permeability=chip_parameters.get("material_permeability", 1e-10),
        )

        self.vibration_model = VibrationEnvironmentalModel(
            chip_dimensions=chip_parameters.get("chip_dimensions", (1e-3, 1e-3, 1e-4))
        )

        self.emi_model = EMIEnvironmentalModel(
            antenna_area=chip_parameters.get("antenna_area", 1e-8),
            coupling_coefficient=chip_parameters.get("coupling_coefficient", 1e-12),
        )

        self.radiation_model = RadiationEnvironmentalModel(
            sensitive_volume=chip_parameters.get("sensitive_volume", 1e-12)
        )

        # 环境效应权重
        self.effect_weights = {
            "thermal": chip_parameters.get("thermal_weight", 0.4),
            "humidity": chip_parameters.get("humidity_weight", 0.2),
            "vibration": chip_parameters.get("vibration_weight", 0.2),
            "emi": chip_parameters.get("emi_weight", 0.1),
            "radiation": chip_parameters.get("radiation_weight", 0.1),
        }

    def simulate_environmental_impact(
        self, conditions: EnvironmentalConditions, time_step: float = 1e-3
    ) -> EnvironmentalImpact:
        """
        模拟综合环境影响

        Args:
            conditions: 环境条件
            time_step: 时间步长

        Returns:
            环境影响结果
        """
        # 计算各个环境效应的影响
        impacts = {}

        # 热效应
        if self.effect_weights["thermal"] > 0:
            thermal_impact = self.thermal_model(
                conditions.temperature,
                conditions.vibration_amplitude * 1000,  # 转换为温度梯度
                time_step,
            )
            impacts["thermal"] = thermal_impact

        # 湿度效应
        if self.effect_weights["humidity"] > 0:
            humidity_impact = self.humidity_model(
                conditions.humidity, conditions.temperature, conditions.aging_time
            )
            impacts["humidity"] = humidity_impact

        # 振动效应
        if self.effect_weights["vibration"] > 0:
            vibration_impact = self.vibration_model(
                conditions.vibration_amplitude,
                conditions.vibration_frequency,
                time_step,
            )
            impacts["vibration"] = vibration_impact

        # EMI效应
        if self.effect_weights["emi"] > 0:
            emi_impact = self.emi_model(
                conditions.emi_strength,
                conditions.vibration_frequency,  # 使用振动频率作为EMI频率近似
            )
            impacts["emi"] = emi_impact

        # 辐射效应
        if self.effect_weights["radiation"] > 0:
            radiation_impact = self.radiation_model(
                conditions.radiation_dose,
                conditions.radiation_dose / max(conditions.aging_time, 1.0),  # 剂量率
                conditions.aging_time,
            )
            impacts["radiation"] = radiation_impact

        # 综合影响评估
        overall_impact = self._calculate_overall_impact(impacts)

        # 生成缓解策略
        mitigation_strategies = self._generate_mitigation_strategies(overall_impact)

        return EnvironmentalImpact(
            performance_degradation=overall_impact["performance_degradation"],
            noise_increase=overall_impact["noise_increase"],
            reliability_reduction=overall_impact["reliability_reduction"],
            lifetime_reduction=overall_impact["lifetime_reduction"],
            mitigation_strategies=mitigation_strategies,
        )

    def _calculate_overall_impact(self, impacts: Dict[str, Any]) -> Dict[str, Any]:
        """计算综合影响"""
        performance_degradation = {}
        noise_increase = {}
        reliability_reduction = 0.0
        lifetime_reduction = 0.0

        # 性能下降
        if "thermal" in impacts:
            performance_degradation["thermal"] = impacts["thermal"][
                "reliability_degradation"
            ].item()

        if "humidity" in impacts:
            performance_degradation["humidity"] = impacts["humidity"][
                "insulation_degradation"
            ].item()

        if "vibration" in impacts:
            performance_degradation["vibration"] = (
                impacts["vibration"]["resonance_effect"].item() * 0.1
            )

        if "emi" in impacts:
            performance_degradation["emi"] = (
                impacts["emi"]["coupling_noise"].item() * 1000
            )

        if "radiation" in impacts:
            performance_degradation["radiation"] = impacts["radiation"][
                "material_degradation"
            ].item()

        # 噪声增加
        if "thermal" in impacts:
            noise_increase["thermal"] = impacts["thermal"]["thermal_noise"].item()

        if "vibration" in impacts:
            noise_increase["vibration"] = impacts["vibration"]["phase_noise"].item()

        if "emi" in impacts:
            noise_increase["emi"] = impacts["emi"]["induced_voltage"].item()

        # 可靠性下降
        reliability_factors = []
        for impact in impacts.values():
            if "reliability_degradation" in impact:
                reliability_factors.append(impact["reliability_degradation"].item())
            elif "insulation_degradation" in impact:
                reliability_factors.append(impact["insulation_degradation"].item())
            elif "corrosion_effect" in impact:
                reliability_factors.append(impact["corrosion_effect"].item())

        if reliability_factors:
            reliability_reduction = 1.0 - np.prod(
                [1.0 - f for f in reliability_factors]
            )

        # 寿命下降
        lifetime_factors = [
            1.0 - degradation for degradation in performance_degradation.values()
        ]
        lifetime_reduction = 1.0 - np.prod(lifetime_factors)

        return {
            "performance_degradation": performance_degradation,
            "noise_increase": noise_increase,
            "reliability_reduction": reliability_reduction,
            "lifetime_reduction": lifetime_reduction,
        }

    def _generate_mitigation_strategies(
        self, overall_impact: Dict[str, Any]
    ) -> List[str]:
        """生成缓解策略"""
        strategies = []

        # 基于性能下降的策略
        perf_degradation = overall_impact["performance_degradation"]

        if perf_degradation.get("thermal", 0) > 0.1:
            strategies.append("改善散热系统，降低工作温度")

        if perf_degradation.get("humidity", 0) > 0.1:
            strategies.append("增加防水密封，提高湿度防护等级")

        if perf_degradation.get("vibration", 0) > 0.1:
            strategies.append("增加减震装置，优化机械结构")

        if perf_degradation.get("emi", 0) > 0.1:
            strategies.append("加强电磁屏蔽，优化电路布局")

        if perf_degradation.get("radiation", 0) > 0.1:
            strategies.append("使用辐射硬化材料，增加辐射防护")

        # 通用策略
        if overall_impact["reliability_reduction"] > 0.2:
            strategies.append("实施定期维护和性能监控")

        if overall_impact["lifetime_reduction"] > 0.3:
            strategies.append("考虑冗余设计和故障转移机制")

        return strategies if strategies else ["保持当前环境控制措施"]
