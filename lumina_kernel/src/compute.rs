use ndarray::{Array2, ArrayView2, Axis};
use rayon::prelude::*;

/// 并行矩阵乘法
/// 
/// 使用 Rayon 按行并行计算，避免不必要的内存分配
/// 
/// # Arguments
/// * `input` - 输入矩阵 [batch_size, in_features]
/// * `weight` - 权重矩阵 [out_features, in_features]
/// 
/// # Returns
/// 输出矩阵 [batch_size, out_features]
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
            // 对每个输出特征
            // 优化：使用内部循环展开或 BLAS 风格的向量化
            // 对于较大的 in_features，这可以显著提高缓存命中率
            for (j, weight_row) in weight.axis_iter(Axis(0)).enumerate() {
                let mut sum = 0.0;
                let chunks_a = input_row.as_slice().unwrap().chunks_exact(8);
                let chunks_b = weight_row.as_slice().unwrap().chunks_exact(8);
                let rem_a = chunks_a.remainder();
                let rem_b = chunks_b.remainder();

                for (ca, cb) in chunks_a.zip(chunks_b) {
                    sum += ca[0] * cb[0] + ca[1] * cb[1] + ca[2] * cb[2] + ca[3] * cb[3] +
                           ca[4] * cb[4] + ca[5] * cb[5] + ca[6] * cb[6] + ca[7] * cb[7];
                }
                
                for (ra, rb) in rem_a.iter().zip(rem_b.iter()) {
                    sum += ra * rb;
                }
                
                output_row[j] = sum;
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
