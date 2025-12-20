import torch
import torch.nn as nn
import json
import os
from typing import Dict, Any, Optional
from lumina.layers.optical_linear import OpticalLinear
from lumina.models.gpt import OpticalGPT

class LuminaCompiler:
    """
    Lumina Compiler: Translates PyTorch Optical models into hardware-specific instructions.
    
    This compiler handles:
    1. Weight quantization to hardware-supported bit-depths.
    2. Mapping weights to physical control signals (e.g., laser intensity, phase shifts).
    3. Exporting configuration files for the LuminaCore hardware or digital twin.
    """
    
    def __init__(self, hardware_profile: str = "edge"):
        self.hardware_profile = hardware_profile
        # Hardware constraints based on profile
        self.profiles = {
            "nano": {"bits": 4, "max_dim": 64},
            "micro": {"bits": 6, "max_dim": 256},
            "edge": {"bits": 8, "max_dim": 1024},
            "datacenter": {"bits": 12, "max_dim": 4096}
        }
        self.config = self.profiles.get(hardware_profile, self.profiles["edge"])

    def compile(self, model: nn.Module, output_path: str):
        """
        Compiles the model and saves the hardware instructions.
        """
        print(f"Compiling model for {self.hardware_profile} profile...")
        
        instructions = {
            "metadata": {
                "hardware_profile": self.hardware_profile,
                "quantization_bits": self.config["bits"],
                "version": "0.3.0"
            },
            "layers": []
        }

        for name, module in model.named_modules():
            if isinstance(module, OpticalLinear):
                layer_data = self._compile_optical_linear(name, module)
                instructions["layers"].append(layer_data)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(instructions, f, indent=4)
        
        print(f"Compilation complete. Instructions saved to {output_path}")
        return instructions

    def _compile_optical_linear(self, name: str, module: OpticalLinear) -> Dict[str, Any]:
        """
        Maps OpticalLinear weights to hardware control signals.
        """
        weights = module.weight.data
        
        # 1. Quantization
        q_min = 0
        q_max = (2 ** self.config["bits"]) - 1
        
        # Normalize weights to [0, 1] for optical intensity mapping
        w_min, w_max = weights.min(), weights.max()
        normalized_weights = (weights - w_min) / (w_max - w_min + 1e-8)
        
        # Scale to bit depth
        quantized_weights = (normalized_weights * q_max).round().clamp(q_min, q_max).to(torch.int32)
        
        return {
            "layer_name": name,
            "type": "OpticalLinear",
            "in_features": module.in_features,
            "out_features": module.out_features,
            "weight_mapping": {
                "min_val": float(w_min),
                "max_val": float(w_max),
                "data": quantized_weights.tolist()
            },
            "hardware_params": {
                "wavelength_spacing": "0.8nm", # Example WDM param
                "integration_time": "10ns"
            }
        }

def export_model(model: nn.Module, path: str, profile: str = "edge"):
    compiler = LuminaCompiler(hardware_profile=profile)
    return compiler.compile(model, path)
