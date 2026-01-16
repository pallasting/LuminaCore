use std::any::Any;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum HalError {
    #[error("Device not found")]
    DeviceNotFound,
    #[error("Out of memory")]
    OutOfMemory,
    #[error("Command execution failed: {0}")]
    ExecutionError(String),
    #[error("Invalid argument: {0}")]
    InvalidArgument(String),
    #[error("Timeout")]
    Timeout,
}

pub type Result<T> = std::result::Result<T, HalError>;

/// 抽象设备接口
pub trait Device: Send + Sync {
    fn name(&self) -> String;
    fn create_buffer(&self, size: usize) -> Result<Box<dyn Buffer>>;
    fn create_command_queue(&self) -> Result<Box<dyn CommandQueue>>;
}

/// 抽象缓冲区接口
pub trait Buffer: Send + Sync {
    fn size(&self) -> usize;
    fn as_ptr(&self) -> *const u8;
    fn as_mut_ptr(&mut self) -> *mut u8;
}

/// 抽象命令队列接口
pub trait CommandQueue: Send + Sync {
    /// 提交写入命令
    fn enqueue_write(&mut self, buffer: &mut dyn Buffer, data: &[u8]) -> Result<()>;
    
    /// 提交读取命令
    fn enqueue_read(&mut self, buffer: &dyn Buffer, data: &mut [u8]) -> Result<()>;
    
    /// 提交计算核心命令
    fn enqueue_kernel(&mut self, kernel_name: &str, args: Vec<Box<dyn Any>>) -> Result<()>;
    
    /// 等待队列执行完成
    fn finish(&self) -> Result<()>;
}

pub mod mock;
pub mod manager;
