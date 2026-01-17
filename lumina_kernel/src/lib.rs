use pyo3::prelude::*;
use numpy::{PyArray2, PyReadonlyArray2, PyReadonlyArray1};
use num_complex::Complex32;

mod compute;
mod noise;
mod quantization;
mod fused_ops;
mod runtime;
mod hal;
mod physics;

use fused_ops::{optical_linear_forward, optical_linear_inference};
use physics::PhysicsConfig;
use compute::parallel_complex_matmul;
use runtime::LuminaRuntime;

    /// 执行微码指令集
#[pyfunction]
#[pyo3(signature = (json_instructions, device_name=None))]
fn run_microcode(json_instructions: String, device_name: Option<String>) -> PyResult<bool> {
    let instructions: Vec<serde_json::Value> = serde_json::from_str(&json_instructions)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        
    let mut rt = LuminaRuntime::new(device_name)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;
        
    rt.execute_instructions(instructions)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;
        
    Ok(true)
}

/// 在指定设备上执行单层光子计算
///
/// Args:
///     input: 输入张量 [batch, in_features]
///     weight: 权重矩阵 [out_features, in_features]
///     bias: 可选偏置 [out_features]
///     noise_std: 噪声标准差
///     bits: 量化位数
///     device_name: 可选设备名称
///
/// Returns:
///     输出张量 [batch, out_features], 执行时间(秒)
#[pyfunction]
#[pyo3(signature = (input, weight, bias, noise_std, bits, device_name=None))]
fn execute_layer<'py>(
    py: Python<'py>,
    input: PyReadonlyArray2<'py, f32>,
    weight: PyReadonlyArray2<'py, f32>,
    bias: Option<PyReadonlyArray1<'py, f32>>,
    noise_std: f32,
    bits: u8,
    device_name: Option<String>,
) -> PyResult<(Bound<'py, PyArray2<f32>>, f32)> {
    use std::time::Instant;
    
    let input_view = input.as_array();
    let weight_view = weight.as_array();
    let bias_slice = bias.as_ref().map(|b| b.as_slice().unwrap());
    
    // 创建设备运行时
    let start = Instant::now();
    let mut rt = LuminaRuntime::new(device_name)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    
    // 执行融合光子计算
    let output = optical_linear_forward(
        input_view,
        weight_view,
        bias_slice,
        noise_std,
        bits,
        42, // 固定种子用于确定性
    );
    
    let execution_time = start.elapsed().as_secs_f32();
    
    Ok((PyArray2::from_owned_array_bound(py, output), execution_time))
}

/// 执行层推理（无噪声，适合生产环境）
#[pyfunction]
#[pyo3(signature = (input, weight, bias, bits, device_name=None))]
fn execute_layer_inference<'py>(
    py: Python<'py>,
    input: PyReadonlyArray2<'py, f32>,
    weight: PyReadonlyArray2<'py, f32>,
    bias: Option<PyReadonlyArray1<'py, f32>>,
    bits: u8,
    device_name: Option<String>,
) -> PyResult<(Bound<'py, PyArray2<f32>>, f32)> {
    use std::time::Instant;
    
    let input_view = input.as_array();
    let weight_view = weight.as_array();
    let bias_slice = bias.as_ref().map(|b| b.as_slice().unwrap());
    
    let start = Instant::now();
    let _rt = LuminaRuntime::new(device_name)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    
    // 执行无噪声推理
    let output = optical_linear_inference(
        input_view,
        weight_view,
        bias_slice,
        bits,
    );
    
    let execution_time = start.elapsed().as_secs_f32();
    
    Ok((PyArray2::from_owned_array_bound(py, output), execution_time))
}


/// 列出可用设备
#[pyfunction]
fn list_devices() -> PyResult<Vec<String>> {
    let manager = crate::hal::manager::DeviceManager::instance();
    Ok(manager.list_devices())
}

/// 创建模拟设备
#[pyfunction]
fn create_mock_device(name: String, memory_limit: usize) -> PyResult<()> {
    let mut manager = crate::hal::manager::DeviceManager::instance();
    manager.create_mock_device(&name, memory_limit)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    Ok(())
}

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

/// 并行复数矩阵乘法
#[pyfunction]
fn complex_matmul<'py>(
    py: Python<'py>,
    input: PyReadonlyArray2<'py, Complex32>,
    weight: PyReadonlyArray2<'py, Complex32>,
) -> PyResult<Bound<'py, PyArray2<Complex32>>> {
    let input_view = input.as_array();
    let weight_view = weight.as_array();
    
    let output = parallel_complex_matmul(input_view, weight_view);
    
    Ok(PyArray2::from_owned_array_bound(py, output))
}

/// 光子线性层前向传播（融合算子）
#[pyfunction]
#[pyo3(signature = (input, weight, bias, noise_std, bits, seed))]
fn optical_linear_fused<'py>(
    py: Python<'py>,
    input: PyReadonlyArray2<'py, f32>,
    weight: PyReadonlyArray2<'py, f32>,
    bias: Option<PyReadonlyArray1<'py, f32>>,
    noise_std: f32,
    bits: u8,
    seed: u64,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let input_view = input.as_array();
    let weight_view = weight.as_array();
    let bias_slice = bias.as_ref().map(|b| b.as_slice().unwrap());
    
    let output = optical_linear_forward(
        input_view,
        weight_view,
        bias_slice,
        noise_std,
        bits,
        seed,
    );
    
    Ok(PyArray2::from_owned_array_bound(py, output))
}

/// 光子线性层推理（无噪声）
#[pyfunction]
#[pyo3(signature = (input, weight, bias, bits))]
fn optical_linear_infer<'py>(
    py: Python<'py>,
    input: PyReadonlyArray2<'py, f32>,
    weight: PyReadonlyArray2<'py, f32>,
    bias: Option<PyReadonlyArray1<'py, f32>>,
    bits: u8,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let input_view = input.as_array();
    let weight_view = weight.as_array();
    let bias_slice = bias.as_ref().map(|b| b.as_slice().unwrap());
    
    let output = optical_linear_inference(
        input_view,
        weight_view,
        bias_slice,
        bits,
    );
    
    Ok(PyArray2::from_owned_array_bound(py, output))
}

/// 反向传播算子（Straight-Through Estimator）
#[pyfunction]
#[pyo3(signature = (grad_output, input, weight))]
fn optical_linear_backward_kernel<'py>(
    py: Python<'py>,
    grad_output: PyReadonlyArray2<'py, f32>,
    input: PyReadonlyArray2<'py, f32>,
    weight: PyReadonlyArray2<'py, f32>,
) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<f32>>)> {
    let grad_output_view = grad_output.as_array();
    let input_view = input.as_array();
    let weight_view = weight.as_array();
    
    let (grad_input, grad_weight) = fused_ops::optical_linear_backward(
        grad_output_view,
        input_view,
        weight_view,
    );
    
    Ok((
        PyArray2::from_owned_array_bound(py, grad_input),
        PyArray2::from_owned_array_bound(py, grad_weight),
    ))
}

/// 执行物理增强的光子线性层计算
///
/// 包含:
/// - 矩阵乘法
/// - 热串扰效应
/// - 光损耗模拟
/// - 相位噪声
/// - 探测器量化
#[pyfunction]
#[pyo3(signature = (input, weight, bias, physics_params, bits, seed))]
fn optical_linear_physics<'py>(
    py: Python<'py>,
    input: PyReadonlyArray2<'py, f32>,
    weight: PyReadonlyArray2<'py, f32>,
    bias: Option<PyReadonlyArray1<'py, f32>>,
    physics_params: std::collections::HashMap<String, f32>,
    bits: u8,
    seed: u64,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let input_view = input.as_array();
    let weight_view = weight.as_array();
    let bias_slice = bias.as_ref().map(|b| b.as_slice().unwrap());
    
    // 解析物理参数
    let config = PhysicsConfig {
        thermal_crosstalk: *physics_params.get("thermal_crosstalk").unwrap_or(&0.01),
        optical_loss_db: *physics_params.get("optical_loss_db").unwrap_or(&0.5),
        temperature: *physics_params.get("temperature").unwrap_or(&25.0),
    };
    
    // 1. 应用热串扰到输入 (模拟输入波导耦合)
    let mut input_phys = physics::apply_crosstalk(&input_view, config.thermal_crosstalk);
    
    // 2. 应用光损耗
    physics::apply_optical_loss(&mut input_phys, config.optical_loss_db);
    
    // 3. 执行核心矩阵乘法 (使用修改后的输入)
    let output = optical_linear_forward(
        input_phys.view(),
        weight_view,
        bias_slice,
        0.0, // 暂时禁用额外的随机噪声，使用物理噪声替代
        bits,
        seed,
    );
    
    // 4. 应用温度相关的相位噪声到输出 (模拟探测器前的相位漂移)
    let mut output_phys = output;
    physics::apply_thermal_noise(&mut output_phys, config.temperature, seed);
    
    Ok(PyArray2::from_owned_array_bound(py, output_phys))
}

/// Python 模块入口点
#[pymodule]
fn lumina_kernel(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hello_lumina, m)?)?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(optical_linear_fused, m)?)?;
    m.add_function(wrap_pyfunction!(optical_linear_infer, m)?)?;
    m.add_function(wrap_pyfunction!(complex_matmul, m)?)?;
    m.add_function(wrap_pyfunction!(run_microcode, m)?)?;
    m.add_function(wrap_pyfunction!(list_devices, m)?)?;
    m.add_function(wrap_pyfunction!(create_mock_device, m)?)?;
    m.add_function(wrap_pyfunction!(optical_linear_backward_kernel, m)?)?;
    m.add_function(wrap_pyfunction!(execute_layer, m)?)?;
    m.add_function(wrap_pyfunction!(execute_layer_inference, m)?)?;
    m.add_function(wrap_pyfunction!(optical_linear_physics, m)?)?;
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
