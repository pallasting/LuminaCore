import json
import yaml
import os
from typing import Any, Dict, Optional
from .quantizer import WeightQuantizer
from .planner import WDMPlanner

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
