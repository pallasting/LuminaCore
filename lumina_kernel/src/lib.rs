use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use numpy::{PyArray2, PyReadonlyArray2, PyReadonlyArray1};
use ndarray::ArrayView2;

mod compute;
mod noise;
mod quantization;
mod fused_ops;

use fused_ops::{optical_linear_forward, optical_linear_inference};

/// Hello World 测试函数 - 验证 Python-Rust FFI 工作正常
#[pyfunction]
fn hello_lumina() -> PyResult<String> {
    Ok("Hello from LuminaKernel (Rust Backend)! 🚀".to_string())
}

/// 获取版本信息
#[pyfunction]
fn version() -> PyResult<String> {
    Ok(env!("CARGO_PKG_VERSION").to_string())
}

/// 光子线性层前向传播（融合算子）
/// 
/// 一次性完成：矩阵乘法 + 噪声注入 + 量化
/// 
/// # Arguments
/// * `input` - 输入矩阵 [batch_size, in_features]
/// * `weight` - 权重矩阵 [out_features, in_features]
/// * `bias` - 可选偏置 [out_features]
/// * `noise_std` - 噪声标准差
/// * `bits` - 量化位数
/// * `seed` - 随机种子
#[pyfunction]
fn optical_linear_fused<'py>(
    py: Python<'py>,
    input: PyReadonlyArray2<f32>,
    weight: PyReadonlyArray2<f32>,
    bias: Option<PyReadonlyArray1<f32>>,
    noise_std: f32,
    temperature_k: f32,
    crosstalk_coeff: f32,
    bits: u8,
    seed: u64,
) -> PyResult<&'py PyArray2<f32>> {
    let input_view = input.as_array();
    let weight_view = weight.as_array();
    let bias_slice = bias.as_ref().map(|b| b.as_slice().unwrap());
    
    let output = optical_linear_forward(
        input_view,
        weight_view,
        bias_slice,
        noise_std,
        temperature_k,
        crosstalk_coeff,
        bits,
        seed,
    );
    
    Ok(PyArray2::from_owned_array(py, output))
}

/// 光子线性层推理（无噪声）
/// 
/// 用于推理场景，只包含矩阵乘法 + 量化
#[pyfunction]
fn optical_linear_infer<'py>(
    py: Python<'py>,
    input: PyReadonlyArray2<f32>,
    weight: PyReadonlyArray2<f32>,
    bias: Option<PyReadonlyArray1<f32>>,
    bits: u8,
) -> PyResult<&'py PyArray2<f32>> {
    let input_view = input.as_array();
    let weight_view = weight.as_array();
    let bias_slice = bias.as_ref().map(|b| b.as_slice().unwrap());
    
    let output = optical_linear_inference(
        input_view,
        weight_view,
        bias_slice,
        bits,
    );
    
    Ok(PyArray2::from_owned_array(py, output))
}

/// Python 模块入口点
#[pymodule]
fn lumina_kernel(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hello_lumina, m)?)?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(optical_linear_fused, m)?)?;
    m.add_function(wrap_pyfunction!(optical_linear_infer, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hello() {
        let result = hello_lumina();
        assert!(result.is_ok());
        assert!(result.unwrap().contains("LuminaKernel"));
    }
}
