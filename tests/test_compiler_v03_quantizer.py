import torch
import unittest
from lumina.compiler.quantizer import WeightQuantizer
from lumina.layers.optical_components import HardwareConfig

class TestWeightQuantizerV03(unittest.TestCase):
    def setUp(self):
        self.config = HardwareConfig.from_profile("lumina_nano_v1")
        self.quantizer = WeightQuantizer(self.config)

    def test_hwaq_calibration_max_abs(self):
        weights = torch.tensor([-0.5, 0.2, 0.8, -1.2])
        params = self.quantizer.calibrate(weights, method="max_abs")
        
        # Max abs is 1.2. Attenuation for nano_v1 is 0.85. 
        # Scale = (1.2 / 0.85) * 1.1 = 1.4117 * 1.1 = 1.5529
        self.assertAlmostEqual(params["scale"], (1.2 / 0.85) * 1.1, places=4)
        self.assertTrue(self.quantizer.is_calibrated)

    def test_quantize_to_states(self):
        weights = torch.tensor([0.0, 0.5, 1.0])
        self.quantizer.scale = 1.0
        self.quantizer.is_calibrated = True
        
        states = self.quantizer.quantize_to_states(weights)
        # 4-bit precision -> max_state = 15
        expected = torch.tensor([0, 8, 15], dtype=torch.int32)
        torch.testing.assert_close(states, expected)

    def test_dequantize(self):
        states = torch.tensor([0, 8, 15], dtype=torch.int32)
        self.quantizer.scale = 1.0
        
        weights = self.quantizer.dequantize(states)
        self.assertAlmostEqual(weights[0].item(), 0.0)
        self.assertAlmostEqual(weights[1].item(), 8/15, places=4)
        self.assertAlmostEqual(weights[2].item(), 1.0)

    def test_lut_generation(self):
        weights = torch.randn(4, 4)
        lut = self.quantizer.generate_lut(weights)
        
        self.assertIn("scale", lut)
        self.assertIn("zero_point", lut)
        self.assertTrue(lut["is_calibrated"])
        self.assertEqual(lut["precision"], 4)

if __name__ == "__main__":
    unittest.main()
