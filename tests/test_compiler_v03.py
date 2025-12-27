import unittest
import torch
import os
import shutil
from lumina.layers.optical_linear import OpticalLinear
from lumina.compiler.exporter import ConfigExporter
from lumina.compiler.planner import WDMPlanner
from lumina.layers.wdm_mapping import WDMChannelMapper

class TestCompilerv03(unittest.TestCase):
    def setUp(self):
        self.test_dir = "test_exports"
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        self.exporter = ConfigExporter(output_dir=self.test_dir)

    def test_optical_linear_compilation(self):
        layer = OpticalLinear(in_features=4, out_features=2, hardware_profile="lumina_nano_v1")
        compilation_data = layer.compile_to_hardware()
        
        self.assertEqual(compilation_data["type"], "OpticalLinear")
        self.assertIn("lut", compilation_data)
        
        path = self.exporter.export_layer_config("test_layer", compilation_data["lut"])
        self.assertTrue(os.path.exists(path))

    def test_wdm_planner_export(self):
        mapper = WDMChannelMapper(num_channels=4, channel_strategy="adaptive")
        planner = WDMPlanner(mapper.num_channels)
        
        mapping_table = planner.export_mapping_table(mapper)
        self.assertEqual(len(mapping_table["wavelengths"]), 4)
        
        path = self.exporter.export_wdm_config(mapping_table)
        self.assertTrue(os.path.exists(path))

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

if __name__ == "__main__":
    unittest.main()
