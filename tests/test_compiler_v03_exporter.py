import torch
import torch.nn as nn
import unittest
import os
import json
from lumina.compiler.exporter import LuminaExporter
from lumina.layers.optical_linear import OpticalLinear

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = OpticalLinear(784, 128, hardware_profile="lumina_nano_v1")
        self.layer2 = OpticalLinear(128, 10, hardware_profile="lumina_micro_v1")

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

class TestLuminaExporterV03(unittest.TestCase):
    def setUp(self):
        self.exporter = LuminaExporter(output_dir="tests/exports")
        self.model = SimpleModel()

    def test_export_execution_graph(self):
        file_path = self.exporter.export_execution_graph(self.model, input_shape=(1, 784))
        
        self.assertTrue(os.path.exists(file_path))
        
        with open(file_path, 'r') as f:
            graph = json.load(f)
            
        self.assertEqual(graph["version"], "0.3.0")
        self.assertEqual(len(graph["nodes"]), 2)
        
        # Verify layer1
        node1 = graph["nodes"][0]
        self.assertEqual(node1["id"], "layer1")
        self.assertEqual(node1["params"]["in_features"], 784)
        self.assertEqual(node1["params"]["hardware_profile"], "lumina_nano_v1")

    def tearDown(self):
        # Cleanup
        if os.path.exists("tests/exports/execution_graph.lmn.json"):
            os.remove("tests/exports/execution_graph.lmn.json")

if __name__ == "__main__":
    unittest.main()
