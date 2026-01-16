use ndarray::{Array2, ArrayView2, Axis};
use rayon::prelude::*;

use crate::noise::{RngPool};
use crate::quantization::Quantizer;
use crate::compute::dot_product;

/// 融合算子：矩阵乘法 + 噪声注入 + 量化
pub fn optical_linear_forward(
    input: ArrayView2<f32>,
    weight: ArrayView2<f32>,
    bias: Option<&[f32]>,
    noise_std: f32,
    bits: u8,
    seed: u64,
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
    let rng_pool = RngPool::new(seed);
    
    output
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .zip(input.axis_iter(Axis(0)).into_par_iter())
        .for_each(|(mut output_row, input_row)| {
            let mut rng = rng_pool.get_thread_rng();
            let input_slice = input_row.as_slice().unwrap();
            
            for (j, weight_row) in weight.axis_iter(Axis(0)).enumerate() {
                let weight_slice = weight_row.as_slice().unwrap();
                
                // 步骤 1: 矩阵乘法（使用优化后的点积）
                let mut val = dot_product(input_slice, weight_slice);
                
                if let Some(b) = bias {
                    val += b[j];
                }
                
                // 步骤 2: 噪声注入 + WDM 串扰模拟
                let signal_dependent_noise = noise_std * val.abs().sqrt();
                
                // 模拟波分复用串扰 (Crosstalk)
                // 假设相邻通道 j-1, j+1 对通道 j 有 1% 的能量泄露
                let mut crosstalk = 0.0;
                if j > 0 {
                    crosstalk += 0.01 * dot_product(input_slice, weight.axis_iter(Axis(0)).nth(j-1).unwrap().as_slice().unwrap());
                }
                if j < out_features - 1 {
                    crosstalk += 0.01 * dot_product(input_slice, weight.axis_iter(Axis(0)).nth(j+1).unwrap().as_slice().unwrap());
                }
                
                let noise = rng.normal(0.0, signal_dependent_noise);
                let noisy_signal = val + noise + crosstalk;
                
                // 步骤 3: 量化
                output_row[j] = quantizer.quantize(noisy_signal);
            }
        });
    
    output
}

/// 简化版融合算子（仅矩阵乘法 + 量化，无噪声）
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
            let input_slice = input_row.as_slice().unwrap();
            
            for (j, weight_row) in weight.axis_iter(Axis(0)).enumerate() {
                let weight_slice = weight_row.as_slice().unwrap();
                let mut val = dot_product(input_slice, weight_slice);
                
                if let Some(b) = bias {
                    val += b[j];
                }
                
                output_row[j] = val;
            }
            
            // 步骤 3: 批量量化（利用 SIMD 加速）
            quantizer.quantize_batch(output_row.as_slice_mut().unwrap());
        });
    
    output
}

/// 反向传播算子（Straight-Through Estimator）
/// 
/// 计算针对输入和权重的梯度
pub fn optical_linear_backward(
    grad_output: ArrayView2<f32>,
    input: ArrayView2<f32>,
    weight: ArrayView2<f32>,
) -> (Array2<f32>, Array2<f32>) {
    // grad_input = grad_output @ weight
    // ndarray 不直接提供矩阵乘法的高性能实现，我们使用之前定义的 parallel_matmul
    // 或者直接实现一个针对转置矩阵优化的版本
    
    let batch_size = grad_output.nrows();
    let out_features = grad_output.ncols();
    let in_features = input.ncols();
    
    let mut grad_input = Array2::<f32>::zeros((batch_size, in_features));
    let mut grad_weight = Array2::<f32>::zeros((out_features, in_features));
    
    // 计算 grad_input: [batch, out] @ [out, in] -> [batch, in]
    grad_input
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .zip(grad_output.axis_iter(Axis(0)).into_par_iter())
        .for_each(|(mut gi_row, go_row)| {
            let go_slice = go_row.as_slice().unwrap();
            for k in 0..in_features {
                let mut sum = 0.0;
                for j in 0..out_features {
                    sum += go_slice[j] * weight[[j, k]];
                }
                gi_row[k] = sum;
            }
        });
        
    // 计算 grad_weight: [out, batch]^T @ [batch, in] -> [out, in]
    // 等价于: sum_over_batch( grad_output[b, i] * input[b, j] )
    grad_weight
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut gw_row)| {
            for b in 0..batch_size {
                let go_val = grad_output[[b, i]];
                for j in 0..in_features {
                    gw_row[j] += go_val * input[[b, j]];
                }
            }
        });
        
    (grad_input, grad_weight)
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
        assert!(output[[0, 0]] > 1.0 && output[[0, 0]] < 2.0);
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
        assert!((output[[0, 0]] - 1.0).abs() < 0.1);
    }
}
