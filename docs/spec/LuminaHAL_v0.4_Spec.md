# Lumina Hardware Abstraction Layer (HAL) Specification v0.4.0

## 1. 概述
Lumina HAL (Hardware Abstraction Layer) 旨在为上层 Runtime 提供统一的硬件访问接口，屏蔽底层不同硬件后端（如 Mock Simulator, FPGA Accelerator, Silicon Photonic Chip）的差异。

设计理念借鉴 OpenCL 和 Vulkan，强调异步指令队列和显式资源管理。

## 2. 核心架构

```rust
pub trait Device {
    fn create_buffer(&self, size: usize) -> Result<Box<dyn Buffer>, Error>;
    fn create_command_queue(&self) -> Result<Box<dyn CommandQueue>, Error>;
    fn name(&self) -> String;
}

pub trait Buffer {
    fn copy_from_host(&mut self, data: &[u8]) -> Result<(), Error>;
    fn copy_to_host(&self, data: &mut [u8]) -> Result<(), Error>;
}

pub trait CommandQueue {
    fn enqueue_write_buffer(&mut self, buffer: &dyn Buffer, data: &[u8]) -> Result<(), Error>;
    fn enqueue_read_buffer(&mut self, buffer: &dyn Buffer, data: &mut [u8]) -> Result<(), Error>;
    fn enqueue_kernel(&mut self, kernel: &str, args: &[&dyn Any]) -> Result<(), Error>;
    fn finish(&self) -> Result<(), Error>;
}
```

## 3. 指令集扩展 (Micro-code++)

HAL 层将负责解析微码并将高级操作（如 `EXEC_VMM`）拆解为设备特定的原子指令。

### 3.1 内存操作
- `MEM_ALLOC`: 分配设备内存
- `MEM_WRITE`: 主机 -> 设备
- `MEM_READ`: 设备 -> 主机

### 3.2 计算操作
- `VMM_CONFIG`: 配置向量矩阵乘法器（加载权重）
- `VMM_EXEC`: 执行计算
- `WDM_SET`: 设置波长路由

## 4. 后端实现计划

| Backend | 描述 | 状态 |
| :--- | :--- | :--- |
| **CPU (Reference)** | 基于 ndarray/rayon 的参考实现 | 已有 (v0.3) |
| **Mock Driver** | 模拟延迟、带宽、噪声的虚拟设备 | v0.4 目标 |
| **FPGA (PCIe)** | 通过 ioctl/mmap 与 FPGA 通信 | 未来规划 |

## 5. 错误处理
统一使用 `LuminaError` 枚举，涵盖 `DeviceNotFound`, `OutOfMemory`, `Timeout`, `HardwareFault` 等情况。
