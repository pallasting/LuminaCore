use ndarray::{Array2, ArrayView2, Axis};
use rayon::prelude::*;

/// 计算两个向量的点积（SIMD 优化）
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { dot_product_avx2(a, b) };
        }
    }
    
    // 回退到自动向量化友好的循环
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_product_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let len = a.len();
    let mut sum = _mm256_setzero_ps();
    
    let mut i = 0;
    while i + 8 <= len {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        sum = _mm256_fmadd_ps(va, vb, sum);
        i += 8;
    }
    
    // 归约 256 位向量到单个 f32
    let mut res = [0.0f32; 8];
    _mm256_storeu_ps(res.as_mut_ptr(), sum);
    let mut final_sum: f32 = res.iter().sum();
    
    // 处理剩余元素
    while i < len {
        final_sum += a[i] * b[i];
        i += 1;
    }
    
    final_sum
}

/// 并行矩阵乘法
pub fn parallel_matmul(input: ArrayView2<f32>, weight: ArrayView2<f32>) -> Array2<f32> {
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
    
    // 创建输出矩阵
    let mut output = Array2::<f32>::zeros((batch_size, out_features));
    
    // 并行计算每一行
    output
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .zip(input.axis_iter(Axis(0)).into_par_iter())
        .for_each(|(mut output_row, input_row)| {
            let input_slice = input_row.as_slice().unwrap();
            // 对每个输出特征
            for (j, weight_row) in weight.axis_iter(Axis(0)).enumerate() {
                let weight_slice = weight_row.as_slice().unwrap();
                output_row[j] = dot_product(input_slice, weight_slice);
            }
        });
    
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn test_parallel_matmul() {
        let input = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let weight = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
        
        let output = parallel_matmul(input.view(), weight.view());
        
        assert_eq!(output.shape(), &[2, 2]);
        assert!((output[[0, 0]] - 1.0).abs() < 1e-6);
        assert!((output[[0, 1]] - 2.0).abs() < 1e-6);
        assert!((output[[1, 0]] - 3.0).abs() < 1e-6);
        assert!((output[[1, 1]] - 4.0).abs() < 1e-6);
    }
    
    #[test]
    fn test_matmul_dimensions() {
        let input = arr2(&[[1.0, 2.0, 3.0]]);
        let weight = arr2(&[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]);
        
        let output = parallel_matmul(input.view(), weight.view());
        
        assert_eq!(output.shape(), &[1, 2]);
        assert!((output[[0, 0]] - 1.0).abs() < 1e-6);
        assert!((output[[0, 1]] - 2.0).abs() < 1e-6);
    }
}
