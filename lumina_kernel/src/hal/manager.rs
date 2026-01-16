use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use once_cell::sync::Lazy;
use super::{Device, Result, HalError, mock::MockDevice};

static GLOBAL_MANAGER: Lazy<Mutex<DeviceManager>> = Lazy::new(|| Mutex::new(DeviceManager::new()));

pub struct DeviceManager {
    devices: HashMap<String, Arc<dyn Device>>,
}

impl DeviceManager {
    pub fn new() -> Self {
        Self {
            devices: HashMap::new(),
        }
    }

    pub fn instance() -> std::sync::MutexGuard<'static, DeviceManager> {
        GLOBAL_MANAGER.lock().unwrap()
    }

    pub fn create_mock_device(&mut self, name: &str, memory_limit: usize) -> Result<Arc<dyn Device>> {
        let device = Arc::new(MockDevice::new(name, memory_limit));
        self.devices.insert(name.to_string(), device.clone());
        Ok(device)
    }

    pub fn get_device(&self, name: &str) -> Result<Arc<dyn Device>> {
        self.devices.get(name)
            .cloned()
            .ok_or(HalError::DeviceNotFound)
    }

    pub fn list_devices(&self) -> Vec<String> {
        self.devices.keys().cloned().collect()
    }
}
