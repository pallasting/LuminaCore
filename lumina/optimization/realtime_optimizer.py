"""
实时优化和动态调整算法

实现运行时的自适应优化：
- 实时性能监控
- 动态参数调整
- 自适应控制算法
- 预测性优化
"""

import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


@dataclass
class SystemState:
    """系统状态数据类"""

    timestamp: float
    performance_metrics: Dict[str, float]
    environmental_conditions: Dict[str, float]
    resource_utilization: Dict[str, float]
    error_rates: Dict[str, float]


@dataclass
class OptimizationAction:
    """优化动作数据类"""

    parameter_changes: Dict[str, float]
    expected_improvement: float
    confidence: float
    execution_time: float
    rollback_plan: Dict[str, float]


@dataclass
class AdaptiveControlResult:
    """自适应控制结果"""

    actions_taken: List[OptimizationAction]
    performance_improvement: float
    stability_score: float
    convergence_time: float


class RealTimeMonitor(nn.Module):
    """
    实时性能监控器

    持续监控系统性能指标，检测异常和趋势
    """

    def __init__(self, monitoring_window: int = 100, anomaly_threshold: float = 2.0):
        super(RealTimeMonitor, self).__init__()

        self.monitoring_window = monitoring_window
        self.anomaly_threshold = anomaly_threshold

        # 监控数据存储
        self.metric_history = {}
        self.anomaly_detector = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

        # 趋势分析器
        self.trend_analyzer = nn.LSTM(5, 16, batch_first=True)

        # 基准性能
        self.baseline_performance = {}

    def update_metrics(
        self, metrics: Dict[str, float], timestamp: float
    ) -> Dict[str, Any]:
        """
        更新性能指标

        Args:
            metrics: 当前性能指标
            timestamp: 时间戳

        Returns:
            监控结果
        """
        # 存储历史数据
        for metric_name, value in metrics.items():
            if metric_name not in self.metric_history:
                self.metric_history[metric_name] = deque(maxlen=self.monitoring_window)

            self.metric_history[metric_name].append((timestamp, value))

        # 检测异常
        anomalies = self._detect_anomalies(metrics)

        # 分析趋势
        trends = self._analyze_trends()

        # 计算性能评分
        performance_score = self._calculate_performance_score(metrics)

        return {
            "anomalies": anomalies,
            "trends": trends,
            "performance_score": performance_score,
            "health_status": self._assess_health_status(anomalies, trends),
        }

    def _detect_anomalies(self, current_metrics: Dict[str, float]) -> Dict[str, bool]:
        """检测异常"""
        anomalies = {}

        for metric_name, current_value in current_metrics.items():
            if (
                metric_name in self.metric_history
                and len(self.metric_history[metric_name]) > 10
            ):
                # 计算历史均值和标准差
                values = [v for _, v in self.metric_history[metric_name]]
                mean_val = np.mean(values)
                std_val = np.std(values)

                if std_val > 0:
                    z_score = abs(current_value - mean_val) / std_val
                    anomalies[metric_name] = z_score > self.anomaly_threshold

        return anomalies

    def _analyze_trends(self) -> Dict[str, str]:
        """分析趋势"""
        trends = {}

        for metric_name, history in self.metric_history.items():
            if len(history) >= 5:
                # 简单的线性趋势分析
                values = [v for _, v in history[-5:]]
                slope = np.polyfit(range(len(values)), values, 1)[0]

                if abs(slope) < 0.01:
                    trends[metric_name] = "stable"
                elif slope > 0:
                    trends[metric_name] = (
                        "improving"
                        if metric_name.endswith("_score")
                        or metric_name.endswith("_rate")
                        else "degrading"
                    )
                else:
                    trends[metric_name] = (
                        "degrading"
                        if metric_name.endswith("_score")
                        or metric_name.endswith("_rate")
                        else "improving"
                    )

        return trends

    def _calculate_performance_score(self, metrics: Dict[str, float]) -> float:
        """计算综合性能评分"""
        # 权重配置
        weights = {
            "throughput": 0.3,
            "latency": -0.2,  # 负权重，因为低延迟更好
            "error_rate": -0.3,  # 负权重，因为低错误率更好
            "efficiency": 0.2,
        }

        score = 0.0
        total_weight = 0.0

        for metric, weight in weights.items():
            if metric in metrics:
                # 归一化处理
                normalized_value = self._normalize_metric(metric, metrics[metric])
                score += weight * normalized_value
                total_weight += abs(weight)

        return score / total_weight if total_weight > 0 else 0.5

    def _normalize_metric(self, metric_name: str, value: float) -> float:
        """归一化指标值"""
        # 基于经验值进行归一化
        if metric_name == "throughput":
            return min(value / 100.0, 1.0)  # 假设100为满分
        elif metric_name == "latency":
            return max(0, 1.0 - value / 10.0)  # 假设10ms为差
        elif metric_name == "error_rate":
            return max(0, 1.0 - value * 100)  # 假设1%错误率为差
        elif metric_name == "efficiency":
            return min(value, 1.0)
        else:
            return 0.5

    def _assess_health_status(
        self, anomalies: Dict[str, bool], trends: Dict[str, str]
    ) -> str:
        """评估系统健康状态"""
        anomaly_count = sum(anomalies.values())
        degrading_trends = sum(1 for trend in trends.values() if trend == "degrading")

        if anomaly_count > 2 or degrading_trends > 3:
            return "CRITICAL"
        elif anomaly_count > 0 or degrading_trends > 1:
            return "WARNING"
        else:
            return "HEALTHY"


class AdaptiveController(nn.Module):
    """
    自适应控制器

    基于系统状态动态调整参数
    """

    def __init__(
        self,
        parameter_bounds: Dict[str, Tuple[float, float]],
        adaptation_rate: float = 0.1,
    ):
        super(AdaptiveController, self).__init__()

        self.parameter_bounds = parameter_bounds
        self.adaptation_rate = adaptation_rate

        # 状态评估网络
        self.state_evaluator = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
        )

        # 参数调整网络
        self.parameter_adjuster = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, len(parameter_bounds)),
            nn.Tanh(),  # 输出在[-1, 1]范围
        )

        # 置信度评估
        self.confidence_estimator = nn.Sequential(
            nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid()
        )

        # 当前参数值
        self.current_parameters = {
            param: (bounds[0] + bounds[1]) / 2  # 初始化为中间值
            for param, bounds in parameter_bounds.items()
        }

    def adapt_parameters(self, system_state: SystemState) -> OptimizationAction:
        """
        自适应调整参数

        Args:
            system_state: 当前系统状态

        Returns:
            优化动作
        """
        # 提取状态特征
        state_features = self._extract_state_features(system_state)

        # 评估当前状态
        state_embedding = self.state_evaluator(state_features)

        # 计算参数调整
        parameter_adjustments = self.parameter_adjuster(state_embedding)

        # 计算置信度
        confidence = self.confidence_estimator(state_embedding)

        # 生成参数变化
        parameter_changes = {}
        expected_improvement = 0.0

        for i, (param_name, bounds) in enumerate(self.parameter_bounds.items()):
            adjustment = parameter_adjustments[0, i].item() * self.adaptation_rate
            current_value = self.current_parameters[param_name]

            # 限制在边界内
            new_value = np.clip(current_value + adjustment, bounds[0], bounds[1])
            parameter_changes[param_name] = new_value - current_value

            # 估算改进效果
            expected_improvement += abs(adjustment) * confidence.item()

        # 更新当前参数
        for param, change in parameter_changes.items():
            self.current_parameters[param] += change

        # 生成回滚计划
        rollback_plan = {param: -change for param, change in parameter_changes.items()}

        return OptimizationAction(
            parameter_changes=parameter_changes,
            expected_improvement=expected_improvement,
            confidence=confidence.item(),
            execution_time=time.time(),
            rollback_plan=rollback_plan,
        )

    def _extract_state_features(self, system_state: SystemState) -> torch.Tensor:
        """提取状态特征"""
        features = []

        # 性能指标
        for metric in ["throughput", "latency", "error_rate", "efficiency"]:
            value = system_state.performance_metrics.get(metric, 0.5)
            features.append(self._normalize_metric(metric, value))

        # 环境条件
        for condition in ["temperature", "humidity", "vibration"]:
            value = system_state.environmental_conditions.get(condition, 25.0)
            features.append(value / 100.0)  # 归一化

        # 资源利用率
        for resource in ["cpu", "memory", "power"]:
            value = system_state.resource_utilization.get(resource, 0.5)
            features.append(value)

        # 错误率
        for error_type in ["bit_error", "packet_loss"]:
            value = system_state.error_rates.get(error_type, 0.0)
            features.append(value)

        # 填充到固定长度
        while len(features) < 20:
            features.append(0.5)

        return torch.tensor(features[:20], dtype=torch.float32).unsqueeze(0)

    def _normalize_metric(self, metric_name: str, value: float) -> float:
        """归一化指标"""
        if metric_name == "throughput":
            return min(value / 100.0, 1.0)
        elif metric_name == "latency":
            return max(0, 1.0 - value / 10.0)
        elif metric_name == "error_rate":
            return max(0, 1.0 - value * 100)
        elif metric_name == "efficiency":
            return min(value, 1.0)
        else:
            return value


class PredictiveOptimizer(nn.Module):
    """
    预测性优化器

    基于预测模型进行前瞻性优化
    """

    def __init__(self, prediction_horizon: int = 10, optimization_frequency: int = 5):
        super(PredictiveOptimizer, self).__init__()

        self.prediction_horizon = prediction_horizon
        self.optimization_frequency = optimization_frequency

        # 预测网络
        self.predictor = nn.LSTM(15, 32, batch_first=True)

        # 优化决策网络
        self.decision_maker = nn.Sequential(
            nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 16), nn.ReLU(), nn.Linear(16, 8)
        )

        # 历史状态存储
        self.state_history = deque(maxlen=50)

        # 优化计数器
        self.step_counter = 0

    def should_optimize(self, current_state: SystemState) -> bool:
        """
        判断是否应该进行优化

        Args:
            current_state: 当前系统状态

        Returns:
            是否需要优化
        """
        self.step_counter += 1

        # 定期优化
        if self.step_counter % self.optimization_frequency == 0:
            return True

        # 基于状态变化检测
        if len(self.state_history) > 5:
            recent_states = list(self.state_history)[-5:]
            performance_trend = self._calculate_performance_trend(recent_states)

            # 如果性能下降趋势明显，触发优化
            if performance_trend < -0.1:  # 性能下降超过10%
                return True

        return False

    def predict_and_optimize(self, current_state: SystemState) -> OptimizationAction:
        """
        预测并优化

        Args:
            current_state: 当前系统状态

        Returns:
            优化动作
        """
        # 存储当前状态
        self.state_history.append(current_state)

        if len(self.state_history) < 5:
            # 数据不足，返回默认动作
            return self._default_action()

        # 准备预测输入
        historical_data = self._prepare_prediction_input()

        # 预测未来状态
        with torch.no_grad():
            predictions, _ = self.predictor(historical_data)
            future_states = predictions[:, -self.prediction_horizon :, :]

            # 基于预测做出决策
            decision_input = future_states.mean(dim=1)  # 平均未来状态
            decisions = self.decision_maker(decision_input)

        # 转换为优化动作
        return self._decisions_to_action(decisions.squeeze(), current_state)

    def _calculate_performance_trend(self, states: List[SystemState]) -> float:
        """计算性能趋势"""
        if len(states) < 2:
            return 0.0

        # 计算性能分数的趋势
        performance_scores = []
        for state in states:
            score = sum(state.performance_metrics.values()) / len(
                state.performance_metrics
            )
            performance_scores.append(score)

        # 线性拟合斜率
        if len(performance_scores) >= 3:
            slope = np.polyfit(range(len(performance_scores)), performance_scores, 1)[0]
            return slope
        else:
            return 0.0

    def _prepare_prediction_input(self) -> torch.Tensor:
        """准备预测输入"""
        # 将历史状态转换为张量
        historical_features = []

        for state in self.state_history:
            features = []

            # 性能指标
            for metric in ["throughput", "latency", "error_rate", "efficiency"]:
                features.append(state.performance_metrics.get(metric, 0.5))

            # 环境条件
            for condition in ["temperature", "humidity", "vibration"]:
                features.append(
                    state.environmental_conditions.get(condition, 25.0) / 100.0
                )

            # 资源利用率
            for resource in ["cpu", "memory", "power"]:
                features.append(state.resource_utilization.get(resource, 0.5))

            historical_features.append(features)

        # 填充到固定长度
        while len(historical_features) < 10:
            historical_features.insert(0, [0.5] * 10)

        return torch.tensor(historical_features[-10:], dtype=torch.float32).unsqueeze(0)

    def _decisions_to_action(
        self, decisions: torch.Tensor, current_state: SystemState
    ) -> OptimizationAction:
        """将决策转换为优化动作"""
        # 简化的决策转换（实际应基于具体参数）
        parameter_changes = {
            "power_level": decisions[0].item() * 0.1,
            "frequency": decisions[1].item() * 0.05,
            "threshold": decisions[2].item() * 0.02,
        }

        # 估算改进效果
        expected_improvement = abs(sum(decisions[:3].tolist())) * 0.1

        return OptimizationAction(
            parameter_changes=parameter_changes,
            expected_improvement=expected_improvement,
            confidence=0.8,  # 固定置信度
            execution_time=time.time(),
            rollback_plan={
                param: -change for param, change in parameter_changes.items()
            },
        )

    def _default_action(self) -> OptimizationAction:
        """默认优化动作"""
        return OptimizationAction(
            parameter_changes={},
            expected_improvement=0.0,
            confidence=0.5,
            execution_time=time.time(),
            rollback_plan={},
        )


class ComprehensiveAdaptiveSystem:
    """
    综合自适应系统

    集成监控、控制和预测功能
    """

    def __init__(self, parameter_bounds: Dict[str, Tuple[float, float]]):
        self.monitor = RealTimeMonitor()
        self.controller = AdaptiveController(parameter_bounds)
        self.predictor = PredictiveOptimizer()

        # 系统历史
        self.action_history = []
        self.performance_history = []

        # 自适应参数
        self.adaptation_enabled = True
        self.last_optimization_time = 0
        self.optimization_cooldown = 5.0  # 秒

    def process_system_state(self, system_state: SystemState) -> Dict[str, Any]:
        """
        处理系统状态并返回优化建议

        Args:
            system_state: 当前系统状态

        Returns:
            处理结果和优化建议
        """
        # 实时监控
        monitoring_result = self.monitor.update_metrics(
            system_state.performance_metrics, system_state.timestamp
        )

        result = {
            "monitoring": monitoring_result,
            "optimization_action": None,
            "system_health": monitoring_result["health_status"],
        }

        # 检查是否需要优化
        current_time = time.time()
        if (
            self.adaptation_enabled
            and self.predictor.should_optimize(system_state)
            and current_time - self.last_optimization_time > self.optimization_cooldown
        ):

            # 执行预测性优化
            optimization_action = self.predictor.predict_and_optimize(system_state)

            # 如果预测优化效果显著，执行自适应控制
            if optimization_action.expected_improvement > 0.05:
                adaptive_action = self.controller.adapt_parameters(system_state)

                # 合并两个优化动作
                combined_changes = {}
                combined_changes.update(optimization_action.parameter_changes)
                combined_changes.update(adaptive_action.parameter_changes)

                final_action = OptimizationAction(
                    parameter_changes=combined_changes,
                    expected_improvement=max(
                        optimization_action.expected_improvement,
                        adaptive_action.expected_improvement,
                    ),
                    confidence=min(
                        optimization_action.confidence, adaptive_action.confidence
                    ),
                    execution_time=current_time,
                    rollback_plan={
                        **optimization_action.rollback_plan,
                        **adaptive_action.rollback_plan,
                    },
                )

                result["optimization_action"] = final_action
                self.action_history.append(final_action)
                self.last_optimization_time = current_time

        # 记录性能历史
        self.performance_history.append(
            {
                "timestamp": system_state.timestamp,
                "metrics": system_state.performance_metrics.copy(),
                "health": monitoring_result["health_status"],
            }
        )

        return result

    def get_system_summary(self) -> Dict[str, Any]:
        """获取系统运行摘要"""
        if not self.performance_history:
            return {"status": "no_data"}

        recent_performance = self.performance_history[-10:]  # 最近10个状态

        # 计算平均性能
        avg_metrics = {}
        all_metrics = [p["metrics"] for p in recent_performance]
        for metric in all_metrics[0].keys():
            values = [m.get(metric, 0) for m in all_metrics]
            avg_metrics[metric] = np.mean(values)

        # 计算健康状态分布
        health_states = [p["health"] for p in recent_performance]
        health_distribution = {
            "HEALTHY": health_states.count("HEALTHY"),
            "WARNING": health_states.count("WARNING"),
            "CRITICAL": health_states.count("CRITICAL"),
        }

        # 计算优化效果
        optimization_effectiveness = 0.0
        if self.action_history:
            improvements = [
                action.expected_improvement for action in self.action_history[-10:]
            ]
            optimization_effectiveness = np.mean(improvements)

        return {
            "average_performance": avg_metrics,
            "health_distribution": health_distribution,
            "optimization_effectiveness": optimization_effectiveness,
            "total_optimizations": len(self.action_history),
            "monitoring_period": len(self.performance_history),
        }
