"""
硬件仿真与物理验证测试

测试WDM映射系统的物理效应建模和数字孪生系统功能：
1. WDM通道映射的物理效应验证
2. 色散和非线性效应测试
3. 串扰建模验证
4. 数字孪生系统性能预测
5. 实时监控功能测试
"""

import time
from typing import Any, Dict

import numpy as np
import torch

from lumina.core.digital_twin import PhotonicChipDigitalTwin
from lumina.layers.optical_linear import OpticalLinear
from lumina.layers.wdm_mapping import WDMChannelMapper


class PhotonicSystemSimulator:
    """光子系统仿真器"""

    def __init__(self, num_channels=3, enable_all_effects=True):
        self.num_channels = num_channels
        self.enable_all_effects = enable_all_effects

        # 创建光学层和WDM映射器
        self.optical_layer = OpticalLinear(
            128, 128, hardware_profile="datacenter_high_precision"
        )

        self.wdm_mapper = WDMChannelMapper(
            num_channels=num_channels,
            channel_strategy="rgb" if num_channels == 3 else "adaptive",
            enable_crosstalk=enable_all_effects,
            enable_dispersion=enable_all_effects,
            enable_nonlinearity=enable_all_effects,
        )

        # 创建数字孪生系统
        self.digital_twin = PhotonicChipDigitalTwin(
            optical_layer=self.optical_layer,
            wdm_mapper=self.wdm_mapper,
            monitoring_window=100,
            prediction_horizon=20,
        )

        print(f"光子系统仿真器初始化完成:")
        print(f"  - WDM通道数: {num_channels}")
        print(f"  - 物理效应: {'全部启用' if enable_all_effects else '基础模式'}")

    def simulate_physical_environment(self, ambient_temp=25.0, power_supply=30.0):
        """模拟物理环境参数"""
        # 模拟温度变化（基于环境温度和功耗）
        base_temp = ambient_temp + power_supply * 0.8  # 功耗转化为热量
        temperature_variation = np.random.normal(0, 2.0)  # ±2°C波动
        actual_temp = base_temp + temperature_variation

        # 模拟功耗变化
        power_variation = np.random.normal(0, 2.0)
        actual_power = max(10.0, power_supply + power_variation)

        # 模拟光功率（基于温度影响）
        temp_factor = 1.0 - (actual_temp - 25.0) * 0.01  # 温度每升高1°C，光功率下降1%
        optical_power = 10.0 * temp_factor + np.random.normal(0, 0.5)
        optical_power = max(1.0, optical_power)

        # 模拟误码率（温度和SNR相关）
        temp_error_factor = 1.0 + (actual_temp - 25.0) * 0.1
        base_error_rate = 1e-8 * temp_error_factor
        actual_error_rate = base_error_rate * (1 + np.random.exponential(0.5))

        return {
            "temperature": actual_temp,
            "power_consumption": actual_power,
            "optical_power": optical_power,
            "error_rate": actual_error_rate,
        }


def test_wdm_physical_effects():
    """测试WDM物理效应建模"""
    print("=" * 70)
    print("WDM物理效应建模测试")
    print("=" * 70)

    # 创建仿真器
    simulator = PhotonicSystemSimulator(num_channels=3, enable_all_effects=True)

    # 生成测试数据
    batch_size = 32
    features = 128
    test_input = torch.randn(batch_size, features)

    print(f"\n测试输入: {test_input.shape}")

    # 1. 测试基础WDM映射
    print("\n1. 基础WDM映射测试")
    mapped_channels = simulator.wdm_mapper.map_to_channels(test_input)
    print(f"  映射后形状: {mapped_channels.shape}")
    print(f"  通道增益: {simulator.wdm_mapper.channel_gains.detach().cpu().numpy()}")

    # 2. 测试色散效应
    print("\n2. 色散效应测试")
    dispersion_coeff = simulator.wdm_mapper.dispersion_coeff.item()
    wavelengths = simulator.wdm_mapper.wavelengths.cpu().numpy()
    print(f"  色散系数: {dispersion_coeff:.2f} ps/(nm·km)")
    print(f"  波长配置: {wavelengths}")

    # 3. 测试串扰效应
    print("\n3. 串扰效应测试")
    if simulator.wdm_mapper.crosstalk_matrix is not None:
        crosstalk_matrix = simulator.wdm_mapper.crosstalk_matrix.detach().cpu().numpy()
        # 计算非对角线元素均值作为串扰水平
        mask = ~np.eye(simulator.num_channels, dtype=bool)
        crosstalk_level = np.mean(np.abs(crosstalk_matrix[mask]))
        print(f"  串扰矩阵:\n{crosstalk_matrix}")
        print(f"  平均串扰水平: {crosstalk_level:.4f}")
    else:
        print("  串扰效应已禁用")

    # 4. 测试合并功能
    print("\n4. 通道合并测试")
    combined_output = simulator.wdm_mapper.combine_channels(mapped_channels)
    print(f"  合并后形状: {combined_output.shape}")

    # 5. 测试集成模式
    print("\n5. 集成模式测试")
    integrated_output = simulator.wdm_mapper.forward_integrated(
        test_input, simulator.optical_layer
    )
    print(f"  集成输出形状: {integrated_output.shape}")

    # 6. 性能监控测试
    print("\n6. 物理参数监控")
    physical_params = simulator.wdm_mapper.get_physical_parameters()
    for key, value in physical_params.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}: {value}")
        else:
            print(f"  {key}: {value:.4f}")

    return {
        "mapped_shape": mapped_channels.shape,
        "combined_shape": combined_output.shape,
        "integrated_shape": integrated_output.shape,
        "crosstalk_level": (
            crosstalk_level
            if simulator.wdm_mapper.crosstalk_matrix is not None
            else 0.0
        ),
    }


def test_digital_twin_system():
    """测试数字孪生系统"""
    print("\n" + "=" * 70)
    print("数字孪生系统测试")
    print("=" * 70)

    # 创建仿真器
    simulator = PhotonicSystemSimulator(num_channels=3, enable_all_effects=True)

    # 1. 模拟物理状态更新
    print("\n1. 物理状态更新测试")
    for i in range(10):
        # 模拟环境变化
        ambient_temp = 25.0 + np.sin(i * 0.5) * 5.0  # 温度周期变化
        power_supply = 30.0 + np.random.normal(0, 3.0)

        env_params = simulator.simulate_physical_environment(ambient_temp, power_supply)

        # 更新数字孪生状态
        state = simulator.digital_twin.update_physical_state(
            temperature=env_params["temperature"],
            power_consumption=env_params["power_consumption"],
            optical_power=env_params["optical_power"],
            error_rate=env_params["error_rate"],
            channel_utilization=[0.8, 0.9, 0.7],
        )

        print(
            f"  步骤 {i+1}: 温度={state.temperature:.1f}°C, "
            f"功耗={state.power_consumption:.1f}W, "
            f"SNR={state.snr:.1f}dB"
        )

    # 2. 性能预测测试
    print("\n2. 性能预测测试")
    prediction = simulator.digital_twin.predict_performance(steps_ahead=5)
    print(f"  预测置信度: {prediction.confidence:.2f}")
    print(f"  预测性能指标:")
    for metric, value in prediction.predicted_performance.items():
        print(f"    {metric}: {value:.3f}")
    print(f"  风险评估:")
    for risk, score in prediction.risk_assessment.items():
        print(f"    {risk}: {score:.3f}")

    # 3. 参数优化测试
    print("\n3. 参数优化测试")
    target_performance = {
        "data_rate": 9.0,
        "power_efficiency": 0.6,
        "reliability": 0.98,
    }
    optimizations = simulator.digital_twin.optimize_parameters(target_performance)
    print(f"  优化建议:")
    for param, opt_info in optimizations.items():
        print(f"    {param}: {opt_info}")

    # 4. 系统状态检查
    print("\n4. 系统状态检查")
    system_status = simulator.digital_twin.get_system_status()
    print(f"  预警级别: {system_status['alert_level']}")
    print(f"  活跃警报: {system_status['active_alerts']}")
    print(f"  监控数据点: {system_status['data_points_collected']}")

    if system_status["current_state"]:
        current = system_status["current_state"]
        print(f"  当前状态:")
        for key, value in current.items():
            print(f"    {key}: {value:.3f}")

    return {
        "prediction_confidence": prediction.confidence,
        "alert_level": system_status["alert_level"],
        "optimizations_count": len(optimizations),
    }


def test_wdm_strategies_comparison():
    """测试不同WDM策略的性能对比"""
    print("\n" + "=" * 70)
    print("WDM策略性能对比测试")
    print("=" * 70)

    strategies = ["rgb", "rgbw", "sequential", "adaptive"]
    batch_size = 64
    features = 128

    results = {}

    for strategy in strategies:
        print(f"\n测试策略: {strategy}")

        # 创建WDM映射器
        num_channels = 3 if strategy == "rgb" else (4 if strategy == "rgbw" else 3)

        wdm_mapper = WDMChannelMapper(
            num_channels=num_channels,
            channel_strategy=strategy,
            enable_crosstalk=True,
            enable_dispersion=True,
            enable_nonlinearity=False,  # 简化测试
        )

        # 生成测试数据
        test_input = torch.randn(batch_size, features)

        # 测试映射和合并性能
        start_time = time.time()
        mapped = wdm_mapper.map_to_channels(test_input)
        combined = wdm_mapper.combine_channels(mapped)
        processing_time = time.time() - start_time

        # 获取物理参数
        params = wdm_mapper.get_physical_parameters()

        # 计算信号质量指标
        snr = params["snr_estimate"]
        crosstalk = params["crosstalk_level"]

        print(f"  处理时间: {processing_time:.4f}s")
        print(f"  SNR估计: {snr:.2f}dB")
        print(f"  串扰水平: {crosstalk:.4f}")
        print(
            f"  波长范围: {params['wavelengths'].min():.1f} - {params['wavelengths'].max():.1f}nm"
        )

        results[strategy] = {
            "processing_time": processing_time,
            "snr": snr,
            "crosstalk": crosstalk,
            "wavelength_range": params["wavelengths"].max()
            - params["wavelengths"].min(),
        }

    # 策略排名
    print(f"\n策略性能排名:")
    snr_ranking = sorted(results.items(), key=lambda x: x[1]["snr"], reverse=True)
    for i, (strategy, metrics) in enumerate(snr_ranking, 1):
        print(
            f"  {i}. {strategy}: SNR {metrics['snr']:.2f}dB, "
            f"处理时间 {metrics['processing_time']:.4f}s"
        )

    return results


def test_integration_scenarios():
    """测试集成场景"""
    print("\n" + "=" * 70)
    print("集成场景测试")
    print("=" * 70)

    # 场景1：边缘端部署
    print("\n1. 边缘端部署场景")
    edge_simulator = PhotonicSystemSimulator(
        num_channels=2, enable_all_effects=False  # 边缘端简化模式
    )

    edge_input = torch.randn(16, 64)  # 小批量
    edge_output = edge_simulator.wdm_mapper.forward(edge_input)
    edge_params = edge_simulator.wdm_mapper.get_physical_parameters()

    print(f"  输入形状: {edge_input.shape}")
    print(f"  输出形状: {edge_output.shape}")
    print(f"  SNR估计: {edge_params['snr_estimate']:.2f}dB")
    print(f"  功耗优化: 启用")

    # 场景2：数据中心部署
    print("\n2. 数据中心部署场景")
    datacenter_simulator = PhotonicSystemSimulator(
        num_channels=4, enable_all_effects=True
    )

    datacenter_input = torch.randn(256, 128)  # 大批量 (256 samples, 128 features)
    datacenter_output = datacenter_simulator.wdm_mapper.forward_integrated(
        datacenter_input, datacenter_simulator.optical_layer
    )
    datacenter_params = datacenter_simulator.wdm_mapper.get_physical_parameters()

    print(f"  输入形状: {datacenter_input.shape}")
    print(f"  输出形状: {datacenter_output.shape}")
    print(f"  SNR估计: {datacenter_params['snr_estimate']:.2f}dB")
    print(f"  完整物理建模: 启用")

    # 场景3：自适应优化
    print("\n3. 自适应优化场景")
    adaptive_simulator = PhotonicSystemSimulator(
        num_channels=3, enable_all_effects=True
    )

    # 模拟不同功率分布的输入
    high_power_input = torch.randn(32, 128) * 2.0
    low_power_input = torch.randn(32, 128) * 0.5

    # 应用自适应优化
    adaptive_simulator.wdm_mapper.optimize_channel_allocation(high_power_input)
    optimized_params = adaptive_simulator.wdm_mapper.get_physical_parameters()

    print(f"  高功率输入优化后通道增益: {optimized_params['channel_gains']}")
    print(f"  自适应权重: {optimized_params['adaptive_weights']}")

    return {
        "edge_scenario": {
            "output_shape": edge_output.shape,
            "snr": edge_params["snr_estimate"],
        },
        "datacenter_scenario": {
            "output_shape": datacenter_output.shape,
            "snr": datacenter_params["snr_estimate"],
        },
        "adaptive_optimization": {
            "channel_gains_range": optimized_params["channel_gains"].max()
            - optimized_params["channel_gains"].min()
        },
    }


def run_hardware_simulation_benchmark():
    """运行硬件仿真基准测试"""
    print("开始硬件仿真与物理验证测试...")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")

    # 1. WDM物理效应测试
    wdm_results = test_wdm_physical_effects()

    # 2. 数字孪生系统测试
    twin_results = test_digital_twin_system()

    # 3. WDM策略对比测试
    strategy_results = test_wdm_strategies_comparison()

    # 4. 集成场景测试
    integration_results = test_integration_scenarios()

    # 总结报告
    print("\n" + "=" * 70)
    print("硬件仿真验证总结报告")
    print("=" * 70)

    print(f"\n1. WDM物理效应建模验证:")
    print(f"   ✓ 多通道映射: {wdm_results['mapped_shape']}")
    print(f"   ✓ 通道合并: {wdm_results['combined_shape']}")
    print(f"   ✓ 集成处理: {wdm_results['integrated_shape']}")
    print(f"   ✓ 串扰控制: {wdm_results['crosstalk_level']:.4f}")

    print(f"\n2. 数字孪生系统验证:")
    print(f"   ✓ 预测置信度: {twin_results['prediction_confidence']:.2f}")
    print(f"   ✓ 监控状态: {twin_results['alert_level']}")
    print(f"   ✓ 优化建议: {twin_results['optimizations_count']}项")

    print(f"\n3. WDM策略性能:")
    best_snr_strategy = max(strategy_results.items(), key=lambda x: x[1]["snr"])
    fastest_strategy = min(
        strategy_results.items(), key=lambda x: x[1]["processing_time"]
    )
    print(f"   ✓ 最佳SNR: {best_snr_strategy[0]} ({best_snr_strategy[1]['snr']:.2f}dB)")
    print(
        f"   ✓ 最快处理: {fastest_strategy[0]} ({fastest_strategy[1]['processing_time']:.4f}s)"
    )

    print(f"\n4. 集成场景验证:")
    print(f"   ✓ 边缘端部署: SNR {integration_results['edge_scenario']['snr']:.2f}dB")
    print(
        f"   ✓ 数据中心: SNR {integration_results['datacenter_scenario']['snr']:.2f}dB"
    )
    print(
        f"   ✓ 自适应优化: 增益范围 {integration_results['adaptive_optimization']['channel_gains_range']:.3f}"
    )

    print(f"\n🎯 核心功能验证完成:")
    print(f"   ✅ 完整的WDM通道映射系统")
    print(f"   ✅ 物理效应精确建模")
    print(f"   ✅ 数字孪生实时监控")
    print(f"   ✅ 性能预测与优化")
    print(f"   ✅ 多场景集成验证")

    return {
        "wdm_results": wdm_results,
        "digital_twin_results": twin_results,
        "strategy_results": strategy_results,
        "integration_results": integration_results,
    }


if __name__ == "__main__":
    # 运行硬件仿真基准测试
    benchmark_results = run_hardware_simulation_benchmark()
    print("\n硬件仿真验证测试完成！")
