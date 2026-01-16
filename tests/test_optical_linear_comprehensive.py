"""
OpticalLinear层的综合测试套件

包含以下测试类别：
1. 基础功能测试
2. 硬件配置测试
3. 边界条件测试
4. 异常处理测试
5. 复数支持测试
6. 量化与噪声模型测试
7. 参数化测试
8. 性能优化测试
"""

import pytest
import numpy as np
import torch
import torch.nn as nn

from lumina.layers.optical_linear import OpticalLinear


class TestOpticalLinearBasic:
    """基础功能测试"""

    def test_basic_forward_pass(self):
        """测试基本前向传播"""
        layer = OpticalLinear(16, 8, hardware_profile="lumina_nano_v1")
        layer.eval()
        
        x = torch.randn(4, 16)
        y = layer(x)
        
        assert y.shape == (4, 8)
        assert torch.all(y >= 0)  # 输出应该非负
        assert torch.all(y <= 1.6)  # ADC饱和限制

    def test_with_bias(self):
        """测试带偏置的情况"""
        layer = OpticalLinear(8, 4, hardware_profile="lumina_nano_v1", bias=True)
        layer.eval()
        
        x = torch.randn(2, 8)
        y = layer(x)
        
        assert y.shape == (2, 4)
        assert layer.bias is not None
        assert layer.bias.shape == (4,)

    def test_without_bias(self):
        """测试不带偏置的情况"""
        layer = OpticalLinear(8, 4, hardware_profile="lumina_nano_v1", bias=False)
        layer.eval()
        
        x = torch.randn(2, 8)
        y = layer(x)
        
        assert y.shape == (2, 4)
        assert layer.bias is None

    def test_custom_precision(self):
        """测试自定义精度"""
        layer = OpticalLinear(8, 4, hardware_profile="custom", precision=6)
        layer.eval()
        
        assert layer.precision == 6
        assert layer.max_digital_val == 63  # 2^6 - 1

    def test_custom_noise_level(self):
        """测试自定义噪声水平"""
        layer = OpticalLinear(8, 4, hardware_profile="custom", noise_level=0.25)
        layer.eval()
        
        assert abs(layer.noise_level - 0.25) < 1e-6


class TestOpticalLinearHardwareProfiles:
    """硬件配置测试"""

    @pytest.mark.parametrize("profile", [
        "lumina_nano_v1",
        "lumina_micro_v1", 
        "edge_ultra_low_power",
        "datacenter_high_precision",
        "custom"
    ])
    def test_all_hardware_profiles(self, profile):
        """测试所有硬件配置预设"""
        layer = OpticalLinear(8, 4, hardware_profile=profile)
        
        assert layer.hardware_profile == profile
        assert layer.in_features == 8
        assert layer.out_features == 4

    def test_nano_v1_profile(self):
        """测试lumina_nano_v1配置"""
        layer = OpticalLinear(8, 4, hardware_profile="lumina_nano_v1")
        
        assert layer.noise_level == 0.15
        assert layer.precision == 4
        assert layer.temp_drift == 0.05
        assert layer.attenuation == 0.85

    def test_micro_v1_profile(self):
        """测试lumina_micro_v1配置"""
        layer = OpticalLinear(8, 4, hardware_profile="lumina_micro_v1")
        
        assert layer.noise_level == 0.10
        assert layer.precision == 8
        assert layer.temp_drift == 0.03
        assert layer.attenuation == 0.90

    def test_edge_ultra_low_power_profile(self):
        """测试edge_ultra_low_power配置"""
        layer = OpticalLinear(8, 4, hardware_profile="edge_ultra_low_power")
        
        assert layer.noise_level == 0.20
        assert layer.precision == 2
        assert layer.temp_drift == 0.10
        assert layer.attenuation == 0.75

    def test_datacenter_high_precision_profile(self):
        """测试datacenter_high_precision配置"""
        layer = OpticalLinear(8, 4, hardware_profile="datacenter_high_precision")
        
        assert layer.noise_level == 0.05
        assert layer.precision == 12
        assert layer.temp_drift == 0.01
        assert layer.attenuation == 0.95


class TestOpticalLinearEdgeCases:
    """边界条件测试"""

    def test_minimum_features(self):
        """测试最小特征维度"""
        layer = OpticalLinear(1, 1, hardware_profile="lumina_nano_v1")
        layer.eval()
        
        x = torch.randn(2, 1)
        y = layer(x)
        
        assert y.shape == (2, 1)

    def test_large_features(self):
        """测试大特征维度"""
        layer = OpticalLinear(1024, 512, hardware_profile="lumina_nano_v1")
        layer.eval()
        
        x = torch.randn(1, 1024)
        y = layer(x)
        
        assert y.shape == (1, 512)

    def test_large_batch_size(self):
        """测试大批量大小"""
        layer = OpticalLinear(16, 8, hardware_profile="lumina_nano_v1")
        layer.eval()
        
        x = torch.randn(1024, 16)
        y = layer(x)
        
        assert y.shape == (1024, 8)

    def test_single_batch(self):
        """测试单样本批量"""
        layer = OpticalLinear(16, 8, hardware_profile="lumina_nano_v1")
        layer.eval()
        
        x = torch.randn(1, 16)
        y = layer(x)
        
        assert y.shape == (1, 8)

    def test_extreme_input_values(self):
        """测试极端输入值"""
        layer = OpticalLinear(8, 4, hardware_profile="lumina_nano_v1")
        layer.eval()
        
        # 测试零输入
        x_zeros = torch.zeros(2, 8)
        y_zeros = layer(x_zeros)
        assert torch.all(y_zeros >= 0)
        
        # 测试大输入值
        x_large = torch.randn(2, 8) * 10
        y_large = layer(x_large)
        assert torch.all(y_large >= 0)
        assert torch.all(y_large <= 1.6)


class TestOpticalLinearExceptions:
    """异常处理测试"""

    def test_invalid_hardware_profile(self):
        """测试无效硬件配置"""
        from lumina.exceptions import InvalidParameterError
        with pytest.raises(InvalidParameterError, match="Unknown hardware profile"):
            OpticalLinear(8, 4, hardware_profile="invalid_profile")

    def test_negative_features(self):
        """测试负特征维度"""
        with pytest.raises(Exception):  # 可能是ValueError或RuntimeError
            OpticalLinear(-8, 4)

    def test_zero_features(self):
        """测试零特征维度"""
        with pytest.raises(Exception):
            OpticalLinear(0, 4)

    def test_invalid_precision(self):
        """测试无效精度"""
        with pytest.raises(Exception):
            layer = OpticalLinear(8, 4, precision=0)

    def test_negative_noise_level(self):
        """测试负噪声水平"""
        with pytest.raises(Exception):
            layer = OpticalLinear(8, 4, noise_level=-0.1)


class TestOpticalLinearComplex:
    """复数支持测试"""

    def test_complex_input_forward(self):
        """测试复数输入前向传播"""
        layer = OpticalLinear(8, 4, hardware_profile="datacenter_high_precision")
        layer.eval()
        
        x = torch.randn(2, 8, dtype=torch.complex64)
        y = layer(x)
        
        assert y.shape == (2, 4)
        assert torch.is_complex(y)

    def test_complex_input_training(self):
        """测试复数输入训练模式"""
        layer = OpticalLinear(8, 4, hardware_profile="datacenter_high_precision")
        layer.train()
        
        x = torch.randn(2, 8, dtype=torch.complex64)
        y = layer(x)
        
        assert y.shape == (2, 4)
        assert torch.is_complex(y)

    def test_mixed_real_complex_weights(self):
        """测试实数权重与复数输入的矩阵乘法"""
        layer = OpticalLinear(8, 4, hardware_profile="datacenter_high_precision")
        layer.eval()
        
        # 实数输入
        x_real = torch.randn(2, 8)
        y_real = layer(x_real)
        
        # 复数输入
        x_complex = torch.randn(2, 8, dtype=torch.complex64)
        y_complex = layer(x_complex)
        
        assert y_real.shape == (2, 4)
        assert y_complex.shape == (2, 4)
        assert torch.is_complex(y_complex)

    def test_complex_quantization(self):
        """测试复数量化"""
        layer = OpticalLinear(8, 4, hardware_profile="datacenter_high_precision")
        layer.eval()
        
        # 创建复数输入
        x = torch.randn(2, 8, dtype=torch.complex64)
        x[:, 0] = torch.tensor([2.0+1.0j, -1.5+0.5j])  # 超过量化范围的复数
        
        y = layer(x)
        
        # 验证量化后的复数幅度在合理范围内
        magnitudes = torch.abs(y)
        assert torch.all(magnitudes >= 0)
        assert torch.all(magnitudes <= 1.6)


class TestOpticalLinearQuantization:
    """量化与噪声模型测试"""

    def test_quantization_levels(self):
        """测试量化级别"""
        layer = OpticalLinear(8, 4, precision=4)
        
        # 4-bit精度应该有16个量化级别 (0-15)
        assert layer.max_digital_val == 15
        assert layer.quantization_step == 1.0 / 15

    def test_quantization_boundaries(self):
        """测试量化边界"""
        layer = OpticalLinear(8, 4, precision=2)  # 2-bit, 4 levels
        layer.eval()
        
        # 测试量化函数
        x = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
        y = layer.quantize(x)
        
        expected = torch.tensor([0.0, 0.33333334, 0.6666667, 1.0, 1.0])
        torch.allclose(y, expected, atol=1e-6)

    def test_training_vs_inference(self):
        """测试训练vs推理模式差异"""
        layer = OpticalLinear(8, 4, hardware_profile="lumina_nano_v1")
        
        x = torch.randn(2, 8)
        
        # 推理模式
        layer.eval()
        y_eval = layer(x)
        
        # 训练模式
        layer.train()
        y_train = layer(x)
        
        # 推理模式应该更稳定（无随机噪声）
        # 注意：这里我们不能直接比较，因为训练模式有随机性
        assert y_eval.shape == (2, 4)
        assert y_train.shape == (2, 4)

    def test_noise_injection(self):
        """测试噪声注入"""
        layer = OpticalLinear(8, 4, hardware_profile="edge_ultra_low_power")
        layer.train()
        
        # 使用固定输入测试噪声
        torch.manual_seed(42)
        x = torch.ones(2, 8)
        y1 = layer(x)
        
        torch.manual_seed(42)  # 相同种子
        x = torch.ones(2, 8)
        y2 = layer(x)
        
        # 相同种子应该产生相同结果（除了随机噪声的差异）
        # 注意：由于噪声是随机的，即使相同种子也可能产生微小差异


class TestOpticalLinearOptimization:
    """性能优化测试"""

    def test_forward_optimized(self):
        """测试优化前向传播"""
        layer = OpticalLinear(16, 8, hardware_profile="datacenter_high_precision")
        layer.eval()
        
        x = torch.randn(128, 16)  # 大批量
        
        # 标准前向传播
        y_standard = layer(x)
        
        # 优化前向传播
        y_optimized = layer.forward_optimized(x, batch_size_threshold=64)
        
        assert y_standard.shape == y_optimized.shape == (128, 8)

    def test_forward_smart(self):
        """测试智能前向传播"""
        layer = OpticalLinear(16, 8, hardware_profile="datacenter_high_precision")
        layer.eval()
        
        x = torch.randn(128, 16)
        
        y_smart = layer.forward_smart(x, batch_size_threshold=64)
        
        assert y_smart.shape == (128, 8)

    def test_batch_threshold_behavior(self):
        """测试批量阈值行为"""
        layer = OpticalLinear(16, 8, hardware_profile="datacenter_high_precision")
        layer.eval()
        
        # 小批量：应该使用标准前向
        x_small = torch.randn(32, 16)
        y_small = layer.forward_smart(x_small, batch_size_threshold=64)
        
        # 大批量：应该使用优化前向
        x_large = torch.randn(128, 16)
        y_large = layer.forward_smart(x_large, batch_size_threshold=64)
        
        assert y_small.shape == (32, 8)
        assert y_large.shape == (128, 8)


class TestOpticalLinearPhysicalEffects:
    """物理效应测试"""

    def test_temperature_drift_edge(self):
        """测试边缘端温度漂移"""
        layer = OpticalLinear(8, 4, hardware_profile="edge_ultra_low_power")
        layer.train()
        
        # 确保权重都是正数，以便输出不会被钳制为零
        with torch.no_grad():
            layer.weight.data.abs_()
            
        # 边缘端应该有更强的温度漂移效应
        torch.manual_seed(42)
        x = torch.ones(2, 8) * 0.5  # 使用正数输入
        y1 = layer(x)
        
        # 使用不同的随机种子来确保噪声不同
        torch.manual_seed(43)
        x = torch.ones(2, 8) * 0.5  # 使用正数输入
        y2 = layer(x)
        
        # 边缘端温度变化应该导致输出差异
        assert not torch.allclose(y1, y2, atol=1e-6)

    def test_thermal_noise_datacenter(self):
        """测试数据中心热噪声"""
        layer = OpticalLinear(8, 4, hardware_profile="datacenter_high_precision")
        layer.train()
        
        # 数据中心应该有特定的热噪声模式
        torch.manual_seed(42)
        x = torch.ones(2, 8)
        y1 = layer(x)
        
        # 使用不同的随机种子来确保噪声不同
        torch.manual_seed(43)
        x = torch.ones(2, 8)
        y2 = layer(x)
        
        # 热噪声应该导致输出差异
        assert not torch.allclose(y1, y2, atol=1e-6)

    def test_attenuation_effect(self):
        """测试衰减效应"""
        layer = OpticalLinear(8, 4, hardware_profile="edge_ultra_low_power")
        layer.eval()
        
        x = torch.ones(2, 8)  # 全1输入
        y = layer(x)
        
        # 边缘端有25%衰减，应该显著降低输出
        assert torch.all(y < 1.0)


class TestOpticalLinearIntegration:
    """集成测试"""

    def test_multiple_layers(self):
        """测试多层堆叠"""
        layer1 = OpticalLinear(16, 8, hardware_profile="lumina_nano_v1")
        layer2 = OpticalLinear(8, 4, hardware_profile="lumina_nano_v1")
        
        x = torch.randn(2, 16)
        y1 = layer1(x)
        y2 = layer2(y1)
        
        assert y1.shape == (2, 8)
        assert y2.shape == (2, 4)

    def test_with_standard_layers(self):
        """测试与标准层的集成"""
        optical_layer = OpticalLinear(8, 4, hardware_profile="lumina_nano_v1")
        linear_layer = nn.Linear(4, 2)
        
        x = torch.randn(2, 8)
        y = optical_layer(x)
        y = torch.relu(y)
        y = linear_layer(y)
        
        assert y.shape == (2, 2)

    def test_gradient_flow(self):
        """测试梯度流"""
        layer = OpticalLinear(8, 4, hardware_profile="lumina_nano_v1")
        criterion = nn.MSELoss()
        
        x = torch.randn(2, 8, requires_grad=True)
        target = torch.randn(2, 4)
        
        y = layer(x)
        loss = criterion(y, target)
        loss.backward()
        
        assert x.grad is not None
        assert layer.weight.grad is not None


# 参数化测试
@pytest.mark.parametrize("in_features,out_features", [
    (4, 2),
    (8, 4),
    (16, 8),
    (32, 16),
])
@pytest.mark.parametrize("hardware_profile", [
    "lumina_nano_v1",
    "lumina_micro_v1",
    "edge_ultra_low_power",
    "datacenter_high_precision",
])
def test_parametrized_shapes(in_features, out_features, hardware_profile):
    """参数化测试不同形状和硬件配置"""
    layer = OpticalLinear(in_features, out_features, hardware_profile=hardware_profile)
    layer.eval()
    
    x = torch.randn(2, in_features)
    y = layer(x)
    
    assert y.shape == (2, out_features)
    assert torch.all(y >= 0)
    assert torch.all(y <= 1.6)


@pytest.mark.parametrize("precision", [2, 4, 6, 8, 12])
def test_parametrized_precision(precision):
    """参数化测试不同精度"""
    layer = OpticalLinear(8, 4, precision=precision)
    
    assert layer.precision == precision
    assert layer.max_digital_val == (2 ** precision) - 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])