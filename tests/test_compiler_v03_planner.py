import torch
import unittest
from lumina.compiler.planner import WDMPlanner

class TestWDMPlannerV03(unittest.TestCase):
    def setUp(self):
        self.planner = WDMPlanner(num_channels=4)

    def test_plan_wavelengths_sequential(self):
        wl = self.planner.plan_wavelengths(strategy="sequential")
        self.assertEqual(len(wl), 4)
        self.assertAlmostEqual(wl[0].item(), 450.0)
        self.assertAlmostEqual(wl[-1].item(), 650.0)

    def test_optimize_allocation(self):
        # Create a simple crosstalk model: high crosstalk between adjacent grid points
        grid_size = 128
        model = torch.zeros((grid_size, grid_size))
        for i in range(grid_size):
            for j in range(grid_size):
                dist = abs(i - j)
                if dist == 0:
                    model[i, j] = 1.0
                else:
                    model[i, j] = 1.0 / (dist ** 2)
        
        optimized_wl = self.planner.optimize_allocation(model)
        self.assertEqual(len(optimized_wl), 4)
        
        # Check that they are spread out (greedy should pick far apart wavelengths)
        diffs = torch.diff(optimized_wl.sort().values)
        self.assertTrue((diffs > 20).all())
        print(f"Optimized Wavelengths: {optimized_wl}")

    def test_crosstalk_compensation(self):
        # 2x2 crosstalk matrix
        matrix = torch.tensor([[1.0, 0.2], [0.2, 1.0]])
        comp = self.planner.generate_crosstalk_compensation(matrix)
        
        # Test: matrix * comp should be Identity
        identity = torch.matmul(matrix, comp)
        torch.testing.assert_close(identity, torch.eye(2), atol=1e-5, rtol=1e-5)

if __name__ == "__main__":
    unittest.main()
