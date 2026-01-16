import json
import yaml
import os
import torch.nn as nn
from typing import Any, Dict, Optional, Tuple
from .quantizer import WeightQuantizer
from .planner import WDMPlanner

class LuminaExporter:
    """
    LuminaExporter - 高级模型导出工具
    
    支持将 PyTorch 模型导出为静态光子执行图 (Photonic Execution Graph - PEG)
    """
    
    def __init__(self, output_dir: str = "exports/v0.3"):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def export_execution_graph(self, model: nn.Module, input_shape: Tuple[int, ...]):
        """
        导出模型的执行拓扑结构和所有关联元数据。
        
        PEG 包含：
        - 算子序列与依赖
        - 每个算子的量化 scale/offset
        - WDM 通道映射
        """
        graph = {
            "version": "0.3.0",
            "metadata": {
                "input_shape": input_shape,
                "exported_at": "2025-01-16T12:00:00Z"
            },
            "nodes": []
        }
        
        # 遍历模型层 (简化版实现，导出 OpticalLinear, ComplexOpticalLinear 和 OpticalAttention 层)
        from ..layers.optical_linear import OpticalLinear
        from ..layers.complex_linear import ComplexOpticalLinear
        from ..layers.attention import OpticalAttention
        for name, module in model.named_modules():
            if isinstance(module, (OpticalLinear, ComplexOpticalLinear, OpticalAttention)):
                node = {
                    "id": name,
                    "type": module.__class__.__name__,
                    "params": {
                        "in_features": getattr(module, "in_features", getattr(module, "embed_dim", None)),
                        "out_features": getattr(module, "out_features", getattr(module, "embed_dim", None)),
                        "hardware_profile": getattr(module, "hardware_profile", None),
                        "is_complex": isinstance(module, ComplexOpticalLinear)
                    }
                }
                graph["nodes"].append(node)
                
        file_path = os.path.join(self.output_dir, "execution_graph.lmn.json")
        with open(file_path, 'w') as f:
            json.dump(graph, f, indent=4)
        
        return file_path

class ConfigExporter:
    """
    ConfigExporter - 编译器配置导出器
    
    将编译后的权重 LUT、WDM 映射和硬件参数导出为 JSON/YAML 格式
    """
    
    def __init__(self, output_dir: str = "exports"):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
    def export_layer_config(self, layer_id: str, lut: Dict[str, Any]):
        """
        导出层权重 LUT 为 JSON
        """
        file_path = os.path.join(self.output_dir, f"layer_{layer_id}_lut.json")
        with open(file_path, 'w') as f:
            json.dump(lut, f, indent=4)
        return file_path
        
    def export_wdm_config(self, mapping: Dict[str, Any]):
        """
        导出 WDM 映射表为 JSON
        """
        file_path = os.path.join(self.output_dir, "wdm_mapping.json")
        with open(file_path, 'w') as f:
            json.dump(mapping, f, indent=4)
        return file_path
        
    def export_hardware_profile(self, profile_name: str, config_data: Dict[str, Any]):
        """
        导出硬件配置参数为 YAML
        """
        file_path = os.path.join(self.output_dir, f"{profile_name}_config.yaml")
        with open(file_path, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False)
        return file_path

    def export_all(self, compilation_artifact: Dict[str, Any]):
        """
        一键导出所有编译产物
        """
        results = {}
        if "layers" in compilation_artifact:
            for lid, lut in compilation_artifact["layers"].items():
                results[f"layer_{lid}"] = self.export_layer_config(lid, lut)
        
        if "wdm" in compilation_artifact:
            results["wdm"] = self.export_wdm_config(compilation_artifact["wdm"])
            
        if "hardware" in compilation_artifact:
            results["hardware"] = self.export_hardware_profile(
                compilation_artifact.get("profile_name", "device"),
                compilation_artifact["hardware"]
            )
            
        return results
