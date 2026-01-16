"""
Digital Twin System - 光子芯片数字孪生系统

实现光子芯片的数字化镜像，包括：
- 实时物理参数监控
- 性能预测和优化
- 故障诊断和预警
- 自适应参数调整
"""

import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Deque

import numpy as np
import torch
import torch.nn as nn

from lumina.layers.optical_linear import OpticalLinear
from lumina.layers.wdm_mapping import WDMChannelMapper
from lumina.exceptions import InvalidParameterError, ValidationError


@dataclass
class PhysicalState:
    """物理状态数据类"""

    timestamp: float
    temperature: float
    power_consumption: float
    optical_power: float
    snr: float
    crosstalk_level: float
    error_rate: float
    channel_utilization: List[float]
    performance_metrics: Dict[str, float]


@dataclass
class PredictionResult:
    """预测结果数据类"""

    predicted_performance: Dict[str, float]
    confidence: float
    recommendations: List[str]
    risk_assessment: Dict[str, float]


class PhotonicChipDigitalTwin:
    """
    光子芯片数字孪生系统

    实时监控和预测光子芯片的物理状态和性能，
    提供智能优化和故障预警功能。

    Args:
        optical_layer: OpticalLinear层实例
        wdm_mapper: WDMChannelMapper实例
        monitoring_window: 监控窗口大小
        prediction_horizon: 预测时间范围（步数）
    """

    def __init__(
        self,
        optical_layer: OpticalLinear,
        wdm_mapper: Optional[WDMChannelMapper] = None,
        monitoring_window: int = 1000,
        prediction_horizon: int = 50,
    ):
        self.optical_layer = optical_layer
        self.wdm_mapper = wdm_mapper

        # 监控参数
        self.monitoring_window = monitoring_window
        self.prediction_horizon = prediction_horizon

        # 历史数据存储
        self.state_history: Deque[PhysicalState] = deque(maxlen=monitoring_window)
        self.performance_history: Deque[Dict[str, float]] = deque(maxlen=monitoring_window)

        # 当前状态
        self.current_state: Optional[PhysicalState] = None
        self.last_update_time = time.time()

        # 预测模型（简单的线性趋势预测）
        self.performance_predictor = nn.Linear(10, 5)  # 输入10个特征，输出5个性能指标
        self.risk_assessor = nn.Linear(5, 3)  # 评估3种风险类型

        # 初始化预测模型权重
        self._initialize_predictors()

        # 阈值配置
        self.thresholds = {
            "temperature_high": 70.0,  # °C
            "power_high": 50.0,  # W
            "snr_low": 15.0,  # dB
            "crosstalk_high": 0.1,  # 相对值
            "error_rate_high": 1e-6,  # 误码率
        }

        # 预警状态
        self.alert_level = "NORMAL"  # NORMAL, WARNING, CRITICAL
        self.active_alerts: List[str] = []

    def _initialize_predictors(self):
        """初始化预测模型"""
        # 性能预测器 - 基于历史趋势
        nn.init.xavier_uniform_(self.performance_predictor.weight)
        nn.init.zeros_(self.performance_predictor.bias)

        # 风险评估器
        nn.init.xavier_uniform_(self.risk_assessor.weight)
        nn.init.zeros_(self.risk_assessor.bias)

        # 冻结参数（简单实现，实际中可能需要训练）
        for param in self.performance_predictor.parameters():
            param.requires_grad = False
        for param in self.risk_assessor.parameters():
            param.requires_grad = False

    def update_physical_state(
        self,
        temperature: float,
        power_consumption: float,
        optical_power: float,
        error_rate: float,
        channel_utilization: Optional[List[float]] = None,
    ) -> PhysicalState:
        """
        更新物理状态

        Args:
            temperature: 温度 (°C)
            power_consumption: 功耗 (W)
            optical_power: 光功率
            error_rate: 误码率
            channel_utilization: 通道利用率列表

        Returns:
            当前物理状态
        """
        current_time = time.time()

        # 从WDM映射器获取当前参数
        if self.wdm_mapper is not None:
            wdm_params = self.wdm_mapper.get_physical_parameters()
            snr = wdm_params["snr_estimate"]
            crosstalk_level = wdm_params["crosstalk_level"]
        else:
            snr = 25.0  # 默认值
            crosstalk_level = 0.05  # 默认值

        # 计算性能指标
        performance_metrics = self._calculate_performance_metrics(
            snr, crosstalk_level, error_rate, optical_power
        )

        # 创建物理状态
        state = PhysicalState(
            timestamp=current_time,
            temperature=temperature,
            power_consumption=power_consumption,
            optical_power=optical_power,
            snr=snr,
            crosstalk_level=crosstalk_level,
            error_rate=error_rate,
            channel_utilization=channel_utilization or [1.0] * 3,
            performance_metrics=performance_metrics,
        )

        # 更新状态
        self.current_state = state
        self.state_history.append(state)

        # 检查预警条件
        self._check_alerts(state)

        return state

    def _calculate_performance_metrics(
        self, snr: float, crosstalk: float, error_rate: float, optical_power: float
    ) -> Dict[str, float]:
        """计算性能指标"""
        # 计算数据速率（基于SNR和误码率）
        max_data_rate = 10.0  # Gbps
        snr_factor = max(0, min(1, (snr - 10) / 20))  # SNR 10-30dB映射到0-1
        error_factor = max(0, 1 - np.log10(error_rate + 1e-10) / 6)  # 误码率影响

        actual_data_rate = max_data_rate * snr_factor * error_factor

        # 计算能效
        power_efficiency = (
            optical_power / (self.current_state.power_consumption + 1e-10)
            if self.current_state
            else 0.5
        )

        # 计算可靠性分数
        reliability_score = max(0, 1 - error_rate * 1e6)  # 误码率转换为可靠性分数

        return {
            "data_rate_gbps": actual_data_rate,
            "power_efficiency": power_efficiency,
            "reliability_score": reliability_score,
            "spectral_efficiency": snr_factor * 4.0,  # 假设最大4 bits/s/Hz
            "throughput_gbps": actual_data_rate * 0.9,  # 考虑开销
        }

    def _check_alerts(self, state: PhysicalState):
        """检查预警条件"""
        self.active_alerts = []

        # 温度预警
        if state.temperature > self.thresholds["temperature_high"]:
            self.active_alerts.append(f"温度过高: {state.temperature:.1f}°C")
            self.alert_level = "WARNING"

        # 功耗预警
        if state.power_consumption > self.thresholds["power_high"]:
            self.active_alerts.append(f"功耗过高: {state.power_consumption:.1f}W")
            self.alert_level = "WARNING"

        # SNR预警
        if state.snr < self.thresholds["snr_low"]:
            self.active_alerts.append(f"SNR过低: {state.snr:.1f}dB")
            self.alert_level = "WARNING"

        # 串扰预警
        if state.crosstalk_level > self.thresholds["crosstalk_high"]:
            self.active_alerts.append(f"串扰过高: {state.crosstalk_level:.3f}")
            self.alert_level = "CRITICAL"

        # 误码率预警
        if state.error_rate > self.thresholds["error_rate_high"]:
            self.active_alerts.append(f"误码率过高: {state.error_rate:.2e}")
            self.alert_level = "CRITICAL"

        # 如果没有预警，设置为正常
        if not self.active_alerts:
            self.alert_level = "NORMAL"

    def predict_performance(self, steps_ahead: int = 10) -> PredictionResult:
        """
        预测未来性能

        Args:
            steps_ahead: 预测步数

        Returns:
            预测结果
        """
        if len(self.state_history) < 10:
            # 历史数据不足，返回默认预测
            return self._default_prediction()

        # 准备预测特征
        recent_states = list(self.state_history)[-10:]  # 最近10个状态

        # 提取特征
        features = self._extract_prediction_features(recent_states)

        # 执行预测
        with torch.no_grad():
            if torch.is_tensor(features):
                features = features.unsqueeze(0)  # 添加batch维度
                predicted_metrics = self.performance_predictor(features)
                risk_scores = self.risk_assessor(predicted_metrics)

                # 转换为字典
                metric_names = [
                    "data_rate",
                    "power_efficiency",
                    "reliability",
                    "spectral_efficiency",
                    "throughput",
                ]
                predicted_performance = {
                    name: pred.item()
                    for name, pred in zip(metric_names, predicted_metrics[0])
                }

                risk_names = ["thermal_risk", "optical_risk", "electrical_risk"]
                risk_assessment = {
                    name: risk.item() for name, risk in zip(risk_names, risk_scores[0])
                }
            else:
                predicted_performance, risk_assessment = self._trend_based_prediction(
                    recent_states
                )

        # 生成建议
        recommendations = self._generate_recommendations(
            predicted_performance, risk_assessment
        )

        # 计算置信度（基于历史数据的稳定性）
        confidence = self._calculate_prediction_confidence(recent_states)

        return PredictionResult(
            predicted_performance=predicted_performance,
            confidence=confidence,
            recommendations=recommendations,
            risk_assessment=risk_assessment,
        )

    def _extract_prediction_features(self, states: List[PhysicalState]) -> torch.Tensor:
        """提取预测特征"""
        if not states:
            return torch.zeros(10)

        # 提取关键特征
        features = []

        # 最近状态特征
        latest = states[-1]
        features.extend(
            [
                latest.temperature,
                latest.power_consumption,
                latest.optical_power,
                latest.snr,
                latest.crosstalk_level,
                latest.error_rate,
            ]
        )

        # 趋势特征（基于最近几个状态）
        if len(states) >= 3:
            recent_3 = states[-3:]
            temp_trend = np.polyfit(
                range(len(recent_3)), [s.temperature for s in recent_3], 1
            )[0]
            power_trend = np.polyfit(
                range(len(recent_3)), [s.power_consumption for s in recent_3], 1
            )[0]
            snr_trend = np.polyfit(range(len(recent_3)), [s.snr for s in recent_3], 1)[
                0
            ]

            features.extend([temp_trend, power_trend, snr_trend])
        else:
            features.extend([0.0, 0.0, 0.0])

        # 最后一个特征：时间间隔
        if len(states) >= 2:
            time_diff = states[-1].timestamp - states[-2].timestamp
            features.append(time_diff)
        else:
            features.append(1.0)

        return torch.tensor(features, dtype=torch.float32)

    def _trend_based_prediction(
        self, states: List[PhysicalState]
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """基于趋势的简单预测"""
        if len(states) < 3:
            # 默认预测
            return (
                {
                    "data_rate": 8.0,
                    "power_efficiency": 0.5,
                    "reliability": 0.95,
                    "spectral_efficiency": 3.0,
                    "throughput": 7.2,
                },
                {"thermal_risk": 0.1, "optical_risk": 0.1, "electrical_risk": 0.1},
            )

        # 提取性能指标历史
        data_rates = [
            self._calculate_performance_metrics(
                s.snr, s.crosstalk_level, s.error_rate, s.optical_power
            )["data_rate_gbps"]
            for s in states[-5:]
        ]
        power_efficiencies = [
            self._calculate_performance_metrics(
                s.snr, s.crosstalk_level, s.error_rate, s.optical_power
            )["power_efficiency"]
            for s in states[-5:]
        ]

        # 线性趋势预测
        x = np.arange(len(data_rates))

        # 预测下一个值
        next_data_rate = (
            np.polyfit(x, data_rates, 1)[0] * len(data_rates)
            + np.polyfit(x, data_rates, 1)[1]
        )
        next_power_eff = (
            np.polyfit(x, power_efficiencies, 1)[0] * len(power_efficiencies)
            + np.polyfit(x, power_efficiencies, 1)[1]
        )

        # 限制预测值范围
        next_data_rate = max(0, min(10, next_data_rate))
        next_power_eff = max(0, min(1, next_power_eff))

        predicted_performance = {
            "data_rate": next_data_rate,
            "power_efficiency": next_power_eff,
            "reliability": 0.95,  # 简化处理
            "spectral_efficiency": 3.0,  # 简化处理
            "throughput": next_data_rate * 0.9,
        }

        # 风险评估
        risk_assessment = {
            "thermal_risk": 0.1 if states[-1].temperature < 50 else 0.5,
            "optical_risk": 0.1 if states[-1].snr > 20 else 0.4,
            "electrical_risk": 0.1 if states[-1].power_consumption < 30 else 0.3,
        }

        return predicted_performance, risk_assessment

    def _generate_recommendations(
        self, performance: Dict[str, float], risk: Dict[str, float]
    ) -> List[str]:
        """生成优化建议"""
        recommendations = []

        # 基于风险的建议
        if risk["thermal_risk"] > 0.5:
            recommendations.append("建议降低工作温度，增加散热措施")

        if risk["optical_risk"] > 0.5:
            recommendations.append("建议优化WDM通道配置，降低串扰")

        if risk["electrical_risk"] > 0.5:
            recommendations.append("建议优化功耗配置，降低能耗")

        # 基于性能的建议
        if performance.get("data_rate", 0) < 5.0:
            recommendations.append("建议提升SNR或降低误码率以提高数据速率")

        if performance.get("power_efficiency", 0) < 0.3:
            recommendations.append("建议优化光学器件配置，提升能效")

        # WDM优化建议
        if self.wdm_mapper is not None:
            recommendations.append("建议启用自适应通道分配以优化性能")

        return recommendations

    def _calculate_prediction_confidence(self, states: List[PhysicalState]) -> float:
        """计算预测置信度"""
        if len(states) < 3:
            return 0.5

        # 计算历史数据的稳定性
        snr_values = [s.snr for s in states]
        snr_std = np.std(snr_values)
        snr_mean = np.mean(snr_values)

        # 基于SNR的稳定性计算置信度
        cv = snr_std / (snr_mean + 1e-10)  # 变异系数
        confidence = max(0.1, min(1.0, 1.0 - cv))

        return confidence

    def _default_prediction(self) -> PredictionResult:
        """默认预测（当数据不足时）"""
        return PredictionResult(
            predicted_performance={
                "data_rate": 8.0,
                "power_efficiency": 0.5,
                "reliability": 0.95,
                "spectral_efficiency": 3.0,
                "throughput": 7.2,
            },
            confidence=0.5,
            recommendations=["需要更多历史数据以提供准确预测"],
            risk_assessment={
                "thermal_risk": 0.2,
                "optical_risk": 0.2,
                "electrical_risk": 0.2,
            },
        )

    def optimize_parameters(
        self, target_performance: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        基于目标性能优化参数

        Args:
            target_performance: 目标性能指标

        Returns:
            优化建议
        """
        optimizations: Dict[str, Dict[str, Any]] = {}

        # 温度优化
        if self.current_state and self.current_state.temperature > 50:
            optimizations["temperature"] = {
                "current": self.current_state.temperature,
                "target": 45.0,
                "action": "reduce_power_or_improve_cooling",
            }

        # SNR优化
        if self.current_state and self.current_state.snr < 20:
            optimizations["snr"] = {
                "current": self.current_state.snr,
                "target": 25.0,
                "action": "optimize_wdm_channels",
            }

            # 建议WDM参数调整
            if self.wdm_mapper is not None:
                current_params = self.wdm_mapper.get_physical_parameters()
                optimizations["wdm_optimization"] = {
                    "channel_gains": (current_params["channel_gains"] * 1.1).tolist(),
                    "dispersion_compensation": True,
                    "crosstalk_reduction": True,
                }

        # 功耗优化
        if self.current_state and self.current_state.power_consumption > 40:
            optimizations["power"] = {
                "current": self.current_state.power_consumption,
                "target": 30.0,
                "action": "enable_power_saving_mode",
            }

        return optimizations

    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态摘要"""
        status = {
            "alert_level": self.alert_level,
            "active_alerts": self.active_alerts,
            "monitoring_duration": time.time() - self.last_update_time,
            "data_points_collected": len(self.state_history),
            "current_state": None,
        }

        if self.current_state:
            status["current_state"] = {
                "temperature": self.current_state.temperature,
                "power_consumption": self.current_state.power_consumption,
                "snr": self.current_state.snr,
                "crosstalk_level": self.current_state.crosstalk_level,
                "error_rate": self.current_state.error_rate,
            }

        return status
