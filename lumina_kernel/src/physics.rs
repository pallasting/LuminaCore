use ndarray::{Array2, ArrayView2};
use rand::SeedableRng;
use rand_xoshiro::rand_core::RngCore;
use rand_xoshiro::Xoshiro256PlusPlus;

/// 物理仿真配置
#[derive(Clone, Copy, Debug)]
pub struct PhysicsConfig {
    /// 热串扰系数 (0.0 - 1.0)
    /// 表示相邻波导之间的能量泄漏比例
    pub thermal_crosstalk: f32,

    /// 光损耗 (dB/cm)
    /// 模拟光信号在传输过程中的衰减
    pub optical_loss_db: f32,

    /// 芯片温度 (°C)
    /// 温度影响相位稳定性
    pub temperature: f32,
}

impl Default for PhysicsConfig {
    fn default() -> Self {
        Self {
            thermal_crosstalk: 0.01, // 1% 串扰
            optical_loss_db: 0.5,    // 0.5 dB 损耗
            temperature: 25.0,       // 室温
        }
    }
}

/// 应用热串扰效应
///
/// 模拟相邻通道之间的能量泄漏
/// output[i] = (1-c)*input[i] + c/2*input[i-1] + c/2*input[i+1]
pub fn apply_crosstalk(input: &ArrayView2<f32>, crosstalk: f32) -> Array2<f32> {
    let (rows, cols) = input.dim();
    let mut output = Array2::zeros((rows, cols));

    // 如果串扰可忽略，直接返回副本
    if crosstalk < 1e-5 {
        output.assign(input);
        return output;
    }

    let keep_ratio = 1.0 - crosstalk;
    let leak_ratio = crosstalk / 2.0;

    for i in 0..rows {
        for j in 0..cols {
            let val = input[[i, j]];
            let mut new_val = val * keep_ratio;

            // 接收来自左侧邻居的串扰
            if j > 0 {
                new_val += input[[i, j - 1]] * leak_ratio;
            }

            // 接收来自右侧邻居的串扰
            if j < cols - 1 {
                new_val += input[[i, j + 1]] * leak_ratio;
            }

            output[[i, j]] = new_val;
        }
    }

    output
}

/// 应用光损耗
///
/// output = input * 10^(-loss_db/10)
pub fn apply_optical_loss(input: &mut Array2<f32>, loss_db: f32) {
    if loss_db.abs() < 1e-5 {
        return;
    }

    let attenuation = 10.0f32.powf(-loss_db / 10.0);
    input.mapv_inplace(|x| x * attenuation);
}

/// 应用温度引起的相位噪声
pub fn apply_thermal_noise(input: &mut Array2<f32>, temperature: f32, seed: u64) {
    // 假设温度越高，噪声越大
    // Base noise at 25°C is small, increases linearly
    let noise_scale = 0.001 * (temperature / 25.0);

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);

    input.mapv_inplace(|x| {
        // 简单的加性高斯白噪声模拟相位漂移对幅度的影响
        let noise = (rng.next_u32() as f32 / u32::MAX as f32 - 0.5) * 2.0 * noise_scale;
        x + x * noise
    });
}
