use std::any::Any;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;
use super::{Device, Buffer, CommandQueue, Result, HalError};

/// 模拟设备：具有人为注入的延迟和带宽限制
pub struct MockDevice {
    name: String,
    memory_limit: usize,
    allocated_memory: Arc<Mutex<usize>>,
}

impl MockDevice {
    pub fn new(name: &str, memory_limit: usize) -> Self {
        Self {
            name: name.to_string(),
            memory_limit,
            allocated_memory: Arc::new(Mutex::new(0)),
        }
    }
}

impl Device for MockDevice {
    fn name(&self) -> String {
        self.name.clone()
    }

    fn create_buffer(&self, size: usize) -> Result<Box<dyn Buffer>> {
        let mut allocated = self.allocated_memory.lock().unwrap();
        if *allocated + size > self.memory_limit {
            return Err(HalError::OutOfMemory);
        }
        *allocated += size;
        
        Ok(Box::new(MockBuffer {
            data: vec![0; size],
            size,
        }))
    }

    fn create_command_queue(&self) -> Result<Box<dyn CommandQueue>> {
        Ok(Box::new(MockCommandQueue {
            latency_ms: 10, // 模拟 10ms 延迟
        }))
    }
}

pub struct MockBuffer {
    data: Vec<u8>,
    size: usize,
}

impl Buffer for MockBuffer {
    fn size(&self) -> usize {
        self.size
    }

    fn as_ptr(&self) -> *const u8 {
        self.data.as_ptr()
    }

    fn as_mut_ptr(&mut self) -> *mut u8 {
        self.data.as_mut_ptr()
    }
}

pub struct MockCommandQueue {
    latency_ms: u64,
}

impl CommandQueue for MockCommandQueue {
    fn enqueue_write(&mut self, buffer: &mut dyn Buffer, data: &[u8]) -> Result<()> {
        // 模拟 PCIe 传输延迟
        thread::sleep(Duration::from_millis(self.latency_ms));
        
        if data.len() > buffer.size() {
            return Err(HalError::InvalidArgument("Data too large for buffer".into()));
        }
        
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr(),
                buffer.as_mut_ptr(),
                data.len()
            );
        }
        Ok(())
    }

    fn enqueue_read(&mut self, buffer: &dyn Buffer, data: &mut [u8]) -> Result<()> {
        thread::sleep(Duration::from_millis(self.latency_ms));
        
        if data.len() > buffer.size() {
            return Err(HalError::InvalidArgument("Output buffer too small".into()));
        }
        
        unsafe {
            std::ptr::copy_nonoverlapping(
                buffer.as_ptr(),
                data.as_mut_ptr(),
                data.len()
            );
        }
        Ok(())
    }

    fn enqueue_kernel(&mut self, kernel_name: &str, _args: Vec<Box<dyn Any>>) -> Result<()> {
        println!("[MockHAL] Executing kernel: {}", kernel_name);
        // 模拟计算时间
        thread::sleep(Duration::from_millis(50));
        Ok(())
    }

    fn finish(&self) -> Result<()> {
        // 模拟同步等待
        Ok(())
    }
}
