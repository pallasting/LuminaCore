import torch
import torch.nn as nn
import unittest
import os
import json
from lumina.layers import ComplexOpticalLinear
from lumina.compiler import LuminaExporter, WDMPlanner
from lumina.compiler.instruction_set import MicroCodeCompiler
from lumina.core import PhotonicChipDigitalTwin, LuminaCalibrationPipeline

class TestV03E2ECompiler(unittest.TestCase):
    def setUp(self):
        # 1. 创建复杂的复数模型
        self.model = nn.Sequential(
            ComplexOpticalLinear(1024, 512, hardware_profile="datacenter_high_precision"),
            ComplexOpticalLinear(512, 10, hardware_profile="lumina_micro_v1")
        )
        self.output_dir = "tests/e2e_v03"
        self.exporter = LuminaExporter(output_dir=self.output_dir)
        self.mcc = MicroCodeCompiler()
        
    def test_full_compilation_and_calibration_loop(self):
        print("\nStarting v0.3 E2E Compiler Test...")
        
        # Step 1: 导出 PEG
        peg_path = self.exporter.export_execution_graph(self.model, input_shape=(1, 1024))
        self.assertTrue(os.path.exists(peg_path))
        print(f"✅ Step 1: PEG exported to {peg_path}")
        
        # Step 2: 编译微码
        bin_path = self.mcc.compile(peg_path)
        self.assertTrue(os.path.exists(bin_path))
        print(f"✅ Step 2: Micro-code generated at {bin_path}")
        
        # Step 3: 数字孪生监控与校准
        # 模拟 OpticalLinear 用于数字孪生 (DigitalTwin 构造函数需要)
        from lumina.layers import OpticalLinear
        mock_layer = OpticalLinear(1024, 10)
        dtwin = PhotonicChipDigitalTwin(mock_layer)
        
        # 执行校准流水线
        pipeline = LuminaCalibrationPipeline(self.model, dtwin, exporter=self.exporter)
        calib_results = pipeline.run_calibration()
        
        self.assertIn("calibrated_layers", calib_results)
        self.assertEqual(len(calib_results["calibrated_layers"]), 2)
        print(f"✅ Step 3: Calibration loop finished. Initial SNR: {calib_results['initial_snr']}dB")

    def tearDown(self):
        # Cleanup
        import shutil
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)

if __name__ == "__main__":
    unittest.main()
