import unittest
import json
import lumina_kernel

class TestHalIntegration(unittest.TestCase):
    def test_device_management(self):
        print("\n[HAL] Testing Device Management...")
        
        # 1. Create a custom mock device
        device_name = "test_gpu_0"
        lumina_kernel.create_mock_device(device_name, 2 * 1024 * 1024 * 1024) # 2GB
        
        # 2. List devices
        devices = lumina_kernel.list_devices()
        print(f"Available devices: {devices}")
        
        self.assertIn(device_name, devices)
        
        # Trigger default device creation
        lumina_kernel.run_microcode("[]")
        devices_after = lumina_kernel.list_devices()
        self.assertIn("default_mock", devices_after)
        
    def test_run_on_specific_device(self):
        print("\n[HAL] Testing Microcode Execution on Specific Device...")
        
        device_name = "inference_accel_1"
        lumina_kernel.create_mock_device(device_name, 1024 * 1024 * 1024)
        
        # Simple instruction set
        instructions = [
            {"op": "INIT_SYS", "args": {"version": "0.4.0"}},
            {"op": "LOAD_WEIGHT", "args": {"layer": "layer1", "shape": [128, 128]}},
            {"op": "EXEC_VMM", "args": {"target": "layer1"}}
        ]
        
        # Run on specific device
        success = lumina_kernel.run_microcode(json.dumps(instructions), device_name)
        self.assertTrue(success)
        print(f"✅ Successfully executed on {device_name}")

if __name__ == "__main__":
    unittest.main()
