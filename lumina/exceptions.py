"""
Lumina 自定义异常类

定义了 Lumina 框架中使用的所有自定义异常类。
"""

class LuminaError(Exception):
    """Lumina 框架基础异常类"""
    pass

class InvalidParameterError(LuminaError):
    """参数无效异常"""
    pass

class OpticalComponentError(LuminaError):
    """光学组件相关异常"""
    pass

class ValidationError(LuminaError):
    """输入验证异常"""
    pass

class BoundaryError(LuminaError):
    """边界检查异常"""
    pass

class ConfigurationError(LuminaError):
    """配置相关异常"""
    pass

class SimulationError(LuminaError):
    """仿真相关异常"""
    pass