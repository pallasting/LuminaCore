use ndarray::{Array2, ArrayView2, Axis};
use rayon::prelude::*;

use crate::noise::{FastRng, RngPool};
use crate::quantization::Quantizer;

/// 融合算子：矩阵乘法 + 噪声注入 + 量化
/// 
/// 这是 LuminaKernel 的核心创新：
/// 将三个操作融合为一个算子，数据在 CPU 寄存器中完成所有计算，
/// 减少内存访问次数，大幅提升性能
/// 
/// # Arguments
/// * `input` - 输入矩阵 [batch_size, in_features]
/// * `weight` - 权重矩阵 [out_features, in_features]
/// * `bias` - 可选偏置 [out_features]
/// * `noise_std` - 噪声标准差（信号依赖）
/// * `bits` - 量化位数
/// * `seed` - 随机种子
/// 
/// # Returns
/// 输出矩阵 [batch_size, out_features]
pub fn optical_linear_forward(
    input: ArrayView2<f32>,
    weight: ArrayView2<f32>,
    bias: Option<&[f32]>,
    noise_std: f32,
    temperature_k: f32,
    crosstalk_coeff: f32,
    bits: u8,
    seed: u64,
) -> Array2<f32> {
    let batch_size = input.nrows();
    let out_features = weight.nrows();
    let in_features = input.ncols();
    
    assert_eq!(
        weight.ncols(),
        in_features,
        "Matrix dimension mismatch: input cols {} != weight cols {}",
        in_features,
        weight.ncols()
    );
    
    if let Some(b) = bias {
        assert_eq!(
            b.len(),
            out_features,
            "Bias length {} != out_features {}",
            b.len(),
            out_features
        );
    }
    
    // 创建输出矩阵
    let mut output = Array2::<f32>::zeros((batch_size, out_features));
    
    // 创建量化器（所有线程共享）
    let quantizer = Quantizer::new(bits, -10.0, 10.0);
    
    // 创建 RNG 池
    let rng_pool = RngPool::new(seed);
    
    // 并行计算每一行（融合操作）
    output
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .zip(input.axis_iter(Axis(0)).into_par_iter())
        .for_each(|(mut output_row, input_row)| {
            // 每个线程获取独立的 RNG
            let mut rng = rng_pool.get_thread_rng();
            
            // 对每个输出特征
            for (j, weight_row) in weight.axis_iter(Axis(0)).enumerate() {
                // 步骤 1: 矩阵乘法（点积）
                let mut dot_product: f32 = input_row
                    .iter()
                    .zip(weight_row.iter())
                    .map(|(a, b)| a * b)
                    .sum();
                
                // 添加偏置（如果有）
                if let Some(b) = bias {
                    dot_product += b[j];
                }
                
                // 步骤 2: 噪声注入（多物理场融合）
                
                // 2.1 散粒噪声 (Shot Noise): noise ∝ √signal
                let shot_noise_std = noise_std * dot_product.abs().sqrt();
                let shot_noise = rng.normal(0.0, shot_noise_std);
                
                // 2.2 热噪声 (Thermal Noise)
                let thermal_noise = rng.thermal_noise(temperature_k);
                
                // 2.3 串扰噪声 (Crosstalk)
                let crosstalk = rng.crosstalk_noise(dot_product, crosstalk_coeff);
                
                let noisy_signal = dot_product + shot_noise + thermal_noise + crosstalk;
                
                // 步骤 3: 量化（模拟 ADC）
                let quantized = quantizer.quantize(noisy_signal);
                
                // 写入输出（只有一次内存写入！）
                output_row[j] = quantized;
            }
        });
    
    output
}

/// 简化版融合算子（仅矩阵乘法 + 量化，无噪声）
/// 
/// 用于推理场景，不需要噪声注入
pub fn optical_linear_inference(
    input: ArrayView2<f32>,
    weight: ArrayView2<f32>,
    bias: Option<&[f32]>,
    bits: u8,
) -> Array2<f32> {
    let batch_size = input.nrows();
    let out_features = weight.nrows();
    let in_features = input.ncols();
    
    assert_eq!(weight.ncols(), in_features);
    
    if let Some(b) = bias {
        assert_eq!(b.len(), out_features);
    }
    
    let mut output = Array2::<f32>::zeros((batch_size, out_features));
    let quantizer = Quantizer::new(bits, -10.0, 10.0);
    
    output
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .zip(input.axis_iter(Axis(0)).into_par_iter())
        .for_each(|(mut output_row, input_row)| {
            for (j, weight_row) in weight.axis_iter(Axis(0)).enumerate() {
                let mut dot_product: f32 = input_row
                    .iter()
                    .zip(weight_row.iter())
                    .map(|(a, b)| a * b)
                    .sum();
                
                if let Some(b) = bias {
                    dot_product += b[j];
                }
                
                output_row[j] = quantizer.quantize(dot_product);
            }
        });
    
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn test_optical_linear_forward() {
        let input = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let weight = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
        let bias = vec![0.5, 0.5];
        
        let output = optical_linear_forward(
            input.view(),
            weight.view(),
            Some(&bias),
            0.1,
            8,
            42,
        );
        
        assert_eq!(output.shape(), &[2, 2]);
        
        // 由于有噪声，只能检查大致范围
        assert!(output[[0, 0]] > 1.0 && output[[0, 0]] < 2.0);
        assert!(output[[0, 1]] > 2.0 && output[[0, 1]] < 3.0);
    }
    
    #[test]
    fn test_optical_linear_inference() {
        let input = arr2(&[[1.0, 2.0]]);
        let weight = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
        
        let output = optical_linear_inference(
            input.view(),
            weight.view(),
            None,
            8,
        );
        
        assert_eq!(output.shape(), &[1, 2]);
        
        // 无噪声，应该接近精确值（但有量化误差）
        assert!((output[[0, 0]] - 1.0).abs() < 0.1);
        assert!((output[[0, 1]] - 2.0).abs() < 0.1);
    }
    
    #[test]
    fn test_with_zero_noise() {
        let input = arr2(&[[1.0, 2.0]]);
        let weight = arr2(&[[1.0, 1.0]]);
        
        let output = optical_linear_forward(
            input.view(),
            weight.view(),
            None,
            0.0, // 零噪声
            8,
            42,
        );
        
        // 零噪声时应该接近确定性结果
        assert!((output[[0, 0]] - 3.0).abs() < 0.2);
    }
}
