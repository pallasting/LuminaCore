import numpy as np
import matplotlib.pyplot as plt
import os

class LuminaCoreSimulatorV2:
    def __init__(self, noise_level=0.01, temp_drift=0.0, precision_bits=8):
        """
        LuminaCore 行为级仿真器 v2.0
        
        :param noise_level: 光路噪声水平 (标准差, relative to signal)
        :param temp_drift: 温度漂移导致的波长失配 (0.0 - 1.0). 
                           0.0 = 完美对准
                           0.1 = 10% 的光能因为波长漂移损失掉了，且增加了串扰
        :param precision_bits: DAC/ADC 精度
        """
        self.noise_level = noise_level
        self.temp_drift = temp_drift
        self.precision_bits = precision_bits
        self.max_digital_val = 2**precision_bits - 1
        
        # 能耗统计 (单位: pJ)
        self.energy_stats = {
            "dac": 0.0,
            "adc": 0.0,
            "optical": 0.0 # 主要是热调谐功耗，计算本身忽略不计
        }

    def quantize(self, value):
        """模拟 DAC/ADC 的量化过程"""
        step = 1.0 / self.max_digital_val
        return np.round(value / step) * step

    def dac_convert(self, digital_matrix):
        """
        Step 1: 电子层 -> 光源层 (DAC)
        """
        # 1. 归一化 (假设输入已经在 0-1 之间，或者我们在这里做 scaling)
        # 简单起见，假设输入是浮点，我们将其量化模拟 DAC 行为
        analog_signal = np.clip(digital_matrix, 0, 1)
        quantized_signal = self.quantize(analog_signal)
        
        # 能耗估算: 假设 1GS/s DAC, 8-bit ~ 1pJ/conv (非常激进的估算)
        # 这里按次计算
        self.energy_stats["dac"] += digital_matrix.size * (0.1 * self.precision_bits) 
        
        return quantized_signal

    def optical_matrix_mult(self, input_vec, weight_matrix):
        """
        Step 2 & 3: 光栅路由与并行计算 (包含热漂移模拟)
        """
        # 理想计算
        ideal_result = np.dot(weight_matrix, input_vec)
        
        # --- 模拟物理缺陷 ---
        
        # 1. 温度漂移 (Thermal Drift)
        # 漂移会导致：
        # a) 信号衰减 (Alignment Loss): 光没完全对准光栅端口
        # b) 串扰 (Crosstalk): 光漏到了相邻通道 (这里简化为增加底噪)
        
        drift_loss = 1.0 - (self.temp_drift * 0.8) # 漂移越大，信号越弱
        drift_crosstalk = np.mean(ideal_result) * self.temp_drift * 0.5 # 漏光造成的背景噪声
        
        signal_after_drift = ideal_result * drift_loss
        
        # 2. 光源/散粒噪声 (Shot Noise) - 与信号强度相关
        shot_noise = np.random.normal(0, self.noise_level, ideal_result.shape) * np.sqrt(np.abs(signal_after_drift) + 1e-6)
        
        # 3. 探测器热噪声 (Thermal Noise) - 固定底噪
        detector_noise = np.random.normal(0, 0.005, ideal_result.shape) # 假设 0.5% 的固定底噪
        
        # 4. 综合结果
        noisy_result = signal_after_drift + shot_noise + detector_noise + drift_crosstalk
        
        return noisy_result

    def adc_convert(self, analog_result):
        """
        Step 4: 探测器 -> 电子层 (ADC)
        """
        # 模拟饱和 (Saturation)
        clipped = np.clip(analog_result, 0, 1.5) # 允许一定的过曝
        
        # 量化
        digitized = self.quantize(clipped)
        
        # 能耗估算: ADC 通常比 DAC 费电
        self.energy_stats["adc"] += analog_result.size * (0.2 * self.precision_bits)
        
        return digitized

    def reset_stats(self):
        self.energy_stats = {"dac": 0, "adc": 0, "optical": 0}

def run_comprehensive_simulation():
    print("启动 LuminaCore 综合仿真 (v2.0)...")
    
    # 1. 准备数据
    np.random.seed(42)
    input_dim = 64
    output_dim = 10
    
    input_vector = np.random.rand(input_dim)
    weights = np.random.rand(output_dim, input_dim)
    
    # 归一化权重以防止输出爆炸
    weights = weights / np.sum(weights, axis=1, keepdims=True)
    
    # 理想结果
    cpu_result = np.dot(weights, input_vector)
    
    # 2. 实验变量
    noise_levels = [0.01, 0.05, 0.10, 0.20] # 1% to 20%
    temp_drifts = [0.0, 0.1, 0.3] # 0%, 10%, 30% drift
    
    results = {}
    
    plt.figure(figsize=(15, 10))
    plot_idx = 1
    
    for drift in temp_drifts:
        errors = []
        for noise in noise_levels:
            sim = LuminaCoreSimulatorV2(noise_level=noise, temp_drift=drift, precision_bits=8)
            
            # A. DAC
            opt_w = sim.dac_convert(weights)
            opt_in = sim.dac_convert(input_vector)
            
            # B. Optical Compute
            opt_out = sim.optical_matrix_mult(opt_in, opt_w)
            
            # C. ADC
            final_res = sim.adc_convert(opt_out)
            
            # 计算误差 (MSE)
            mse = np.mean((final_result_normalized(final_res) - final_result_normalized(cpu_result))**2)
            errors.append(mse)
            
            # 记录最后一次的波形用于绘图
            if noise == 0.10: # 绘制典型噪声下的波形
                plt.subplot(3, 3, plot_idx)
                plt.plot(cpu_result, 'k--', label='Ideal CPU')
                plt.plot(final_res, 'r-', alpha=0.7, label=f'Lumina (Drift={drift})')
                plt.title(f"Drift: {drift*100}%, Noise: 10%")
                plt.legend()
                plt.grid(True, alpha=0.3)
                plot_idx += 1
        
        results[drift] = errors

    # 绘制误差趋势图
    plt.subplot(3, 3, (7, 9))
    for drift, errs in results.items():
        plt.plot(noise_levels, errs, marker='o', linewidth=2, label=f'Temp Drift = {drift*100}%')
    
    plt.xlabel('Optical Noise Level')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title('Robustness Analysis: Noise vs. Thermal Drift')
    plt.legend()
    plt.grid(True)
    
    output_path = 'simulation_results_v2.png'
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"仿真完成。结果已保存至 {output_path}")
    
    # 打印能耗示例 (基于最后一次运行)
    print("\n--- 能耗估算 (单次矩阵乘法) ---")
    print(f"DAC Energy: {sim.energy_stats['dac']:.2f} pJ")
    print(f"ADC Energy: {sim.energy_stats['adc']:.2f} pJ")
    print(f"Total Digital Overhead: {sim.energy_stats['dac'] + sim.energy_stats['adc']:.2f} pJ")

def final_result_normalized(vec):
    """辅助函数：归一化以便比较形状"""
    return vec / (np.max(vec) + 1e-9)

if __name__ == "__main__":
    run_comprehensive_simulation()
