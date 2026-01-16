use std::collections::HashMap;
use std::sync::Arc;
use serde_json::Value;
use crate::hal::{Device, Buffer, CommandQueue, manager::DeviceManager};

pub struct LuminaRuntime {
    device: Arc<dyn Device>,
    queue: Box<dyn CommandQueue>,
    pub weights: HashMap<String, Box<dyn Buffer>>,
}

impl LuminaRuntime {
    pub fn new(device_name: Option<String>) -> Result<Self, String> {
        let mut manager = DeviceManager::instance();
        
        let device = if let Some(name) = device_name {
            manager.get_device(&name).map_err(|e| e.to_string())?
        } else {
            // Default: Create/Get a mock device
            if let Ok(dev) = manager.get_device("default_mock") {
                dev
            } else {
                manager.create_mock_device("default_mock", 1024 * 1024 * 1024)
                    .map_err(|e| e.to_string())?
            }
        };

        let queue = device.create_command_queue().map_err(|e| e.to_string())?;
        
        Ok(Self {
            device,
            queue,
            weights: HashMap::new(),
        })
    }


    pub fn execute_instructions(&mut self, instructions: Vec<Value>) -> Result<(), String> {
        for inst in instructions {
            let op = inst["op"].as_str().ok_or("Missing opcode")?;
            let args = &inst["args"];

            match op {
                "LOAD_WEIGHT" => {
                    let layer_id = args["layer"].as_str().ok_or("Missing layer id")?;
                    let shape = args["shape"].as_array().ok_or("Missing shape")?;
                    let rows = shape[0].as_u64().unwrap() as usize;
                    let cols = shape[1].as_u64().unwrap() as usize;
                    let size_bytes = rows * cols * 4; // f32
                    
                    // 分配设备内存
                    let mut buffer = self.device.create_buffer(size_bytes)
                        .map_err(|e| e.to_string())?;
                    
                    // 初始化权重数据 (模拟)
                    let dummy_data = vec![0u8; size_bytes];
                    self.queue.enqueue_write(buffer.as_mut(), &dummy_data)
                        .map_err(|e| e.to_string())?;
                        
                    self.weights.insert(layer_id.to_string(), buffer);
                    println!("Runtime: Loaded weights for layer {} ({} bytes)", layer_id, size_bytes);
                }
                "EXEC_VMM" => {
                    let target = args["target"].as_str().ok_or("Missing target")?;
                    if let Some(_buffer) = self.weights.get(target) {
                        self.queue.enqueue_kernel("VMM_EXEC", vec![])
                            .map_err(|e| e.to_string())?;
                        println!("Runtime: Executed VMM for layer {}", target);
                    } else {
                        return Err(format!("Layer {} not found", target));
                    }
                }
                "EXEC_ATTN_MASK" => {
                    self.queue.enqueue_kernel("ATTN_MASK", vec![])
                        .map_err(|e| e.to_string())?;
                    println!("Runtime: Executed Fused Attention with Mask");
                }
                "INIT_SYS" => {
                    println!("Runtime: Initialized Lumina System v{}", args["version"]);
                }
                _ => {
                    println!("Runtime: Skipping unknown opcode {}", op);
                }
            }
        }
        
        // 等待所有指令执行完成
        self.queue.finish().map_err(|e| e.to_string())?;
        
        Ok(())
    }
}
