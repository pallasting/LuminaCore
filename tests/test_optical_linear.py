import pytest
import torch

from lumina.nn import OpticalLinear
from lumina.exceptions import InvalidParameterError, ValidationError, BoundaryError


def test_optical_linear_forward_shape_and_range():
    torch.manual_seed(0)
    layer = OpticalLinear(16, 8, hardware_profile="lumina_nano_v1")
    layer.eval()  # 推理模式：无训练噪声，只剩衰减+量化

    x = torch.rand(4, 16)
    y = layer(x)

    assert y.shape == (4, 8)
    assert torch.all(y >= 0)
    # ADC 限幅为 1.5，留少量浮动裕度
    assert torch.max(y) <= 1.6


def test_optical_linear_hardware_profile_defaults():
    layer = OpticalLinear(4, 2, hardware_profile="lumina_nano_v1")
    assert layer.precision == 4
    # 15% 光路噪声
    assert abs(layer.noise_level - 0.15) < 1e-6
    # 5% 温漂
    assert abs(layer.temp_drift - 0.05) < 1e-6


def test_edge_temperature_drift_model():
    """测试边缘端增强温度漂移模型"""
    torch.manual_seed(42)
    layer = OpticalLinear(16, 8, hardware_profile="edge_ultra_low_power")
    layer.train()  # 训练模式：启用噪声

    x = torch.ones(2, 16)  # 恒定输入
    y1 = layer(x)
    y2 = layer(x)

    # 由于温度波动，输出应该有随机性
    assert not torch.allclose(y1, y2, atol=1e-6)

    # 验证输出范围合理
    assert torch.all(y1 >= 0) and torch.all(y1 <= 1.6)
    assert torch.all(y2 >= 0) and torch.all(y2 <= 1.6)


def test_datacenter_thermal_noise_model():
    """测试数据中心细化热噪声模型"""
    torch.manual_seed(42)
    layer = OpticalLinear(16, 8, hardware_profile="datacenter_high_precision")
    layer.train()  # 训练模式：启用噪声

    x = torch.ones(2, 16)  # 恒定输入
    y1 = layer(x)
    y2 = layer(x)

    # 由于热噪声，输出应该有随机性
    assert not torch.allclose(y1, y2, atol=1e-6)

    # 验证输出范围合理
    assert torch.all(y1 >= 0) and torch.all(y1 <= 1.6)
    assert torch.all(y2 >= 0) and torch.all(y2 <= 1.6)


def test_noise_model_consistency():
    """测试不同硬件配置的噪声模型行为一致性"""
    torch.manual_seed(42)

    # 标准配置
    layer_standard = OpticalLinear(8, 4, hardware_profile="lumina_nano_v1")
    layer_standard.train()

    # 边缘配置
    layer_edge = OpticalLinear(8, 4, hardware_profile="edge_ultra_low_power")
    layer_edge.train()

    # 数据中心配置
    layer_datacenter = OpticalLinear(8, 4, hardware_profile="datacenter_high_precision")
    layer_datacenter.train()

    x = torch.rand(1, 8)

    y_standard = layer_standard(x)
    y_edge = layer_edge(x)
    y_datacenter = layer_datacenter(x)

    # 所有输出都应该在合理范围内
    for y in [y_standard, y_edge, y_datacenter]:
        assert torch.all(y >= 0) and torch.all(y <= 1.6)


def test_optical_linear_invalid_parameters():
    """测试无效参数的错误处理"""

    # 测试无效的 in_features
    with pytest.raises(InvalidParameterError, match="in_features must be a positive integer"):
        OpticalLinear(-1, 8)

    with pytest.raises(InvalidParameterError, match="in_features must be a positive integer"):
        OpticalLinear(0, 8)

    with pytest.raises(InvalidParameterError, match="in_features must be a positive integer"):
        OpticalLinear(3.5, 8)

    # 测试无效的 out_features
    with pytest.raises(InvalidParameterError, match="out_features must be a positive integer"):
        OpticalLinear(8, -1)

    with pytest.raises(InvalidParameterError, match="out_features must be a positive integer"):
        OpticalLinear(8, 0)

    # 测试无效的 precision
    with pytest.raises(InvalidParameterError, match="precision must be a positive integer"):
        OpticalLinear(8, 4, precision=-1)

    with pytest.raises(BoundaryError, match="precision too high"):
        OpticalLinear(8, 4, precision=64)

    # 测试无效的 noise_level
    with pytest.raises(InvalidParameterError, match="noise_level must be a float between 0.0 and 1.0"):
        OpticalLinear(8, 4, noise_level=-0.1)

    with pytest.raises(InvalidParameterError, match="noise_level must be a float between 0.0 and 1.0"):
        OpticalLinear(8, 4, noise_level=1.5)

    # 测试无效的 temp_drift
    with pytest.raises(InvalidParameterError, match="temp_drift must be a float between 0.0 and 1.0"):
        OpticalLinear(8, 4, temp_drift=-0.1)

    with pytest.raises(InvalidParameterError, match="temp_drift must be a float between 0.0 and 1.0"):
        OpticalLinear(8, 4, temp_drift=1.5)

    # 测试无效的硬件配置
    with pytest.raises(InvalidParameterError, match="Unknown hardware profile"):
        OpticalLinear(8, 4, hardware_profile="invalid_profile")


def test_optical_linear_forward_validation():
    """测试前向传播输入验证"""
    layer = OpticalLinear(8, 4)

    # 测试非张量输入
    with pytest.raises(ValidationError, match="Input must be a torch.Tensor"):
        layer("not a tensor")

    # 测试维度错误
    with pytest.raises(ValidationError, match="Input tensor must be 2-dimensional"):
        layer(torch.rand(8))  # 1D

    with pytest.raises(ValidationError, match="Input tensor must be 2-dimensional"):
        layer(torch.rand(2, 4, 8))  # 3D

    # 测试特征维度不匹配
    with pytest.raises(ValidationError, match="Input feature dimension .* does not match expected"):
        layer(torch.rand(2, 6))  # 期望 8，但输入 6

    # 测试 NaN 值
    with pytest.raises(ValidationError, match="Input tensor contains NaN values"):
        layer(torch.tensor([[1.0, float('nan'), 0.5, 0.0, 0.0, 0.0, 0.0, 0.0]]))

    # 测试无穷大值
    with pytest.raises(ValidationError, match="Input tensor contains infinite values"):
        layer(torch.tensor([[1.0, float('inf'), 0.5, 0.0, 0.0, 0.0, 0.0, 0.0]]))


def test_quantizer_validation():
    """测试量化器输入验证"""
    from lumina.layers.optical_components import HardwareConfig, Quantizer

    config = HardwareConfig(noise_level=0.1, precision=8, temp_drift=0.05, attenuation=0.9)
    quantizer = Quantizer(config)

    # 测试非张量输入
    with pytest.raises(ValidationError, match="Input must be a torch.Tensor"):
        quantizer.quantize("not a tensor")

    # 测试 NaN 值
    with pytest.raises(ValidationError, match="Input tensor contains NaN values"):
        quantizer.quantize(torch.tensor([1.0, float('nan'), 0.5]))

    # 测试无穷大值
    with pytest.raises(ValidationError, match="Input tensor contains infinite values"):
        quantizer.quantize(torch.tensor([1.0, float('inf'), 0.5]))
