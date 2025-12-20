use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use std::f32::consts::PI;

/// 快速随机数生成器（每线程独立）
/// 
/// 使用 Xoshiro256++ 算法，比标准库的随机数生成器快 10 倍以上
pub struct FastRng {
    rng: Xoshiro256PlusPlus,
}

impl FastRng {
    /// 创建新的随机数生成器
    pub fn new(seed: u64) -> Self {
        Self {
            rng: Xoshiro256PlusPlus::seed_from_u64(seed),
        }
    }
    
    /// 从系统熵创建
    pub fn from_entropy() -> Self {
        Self {
            rng: Xoshiro256PlusPlus::from_entropy(),
        }
    }
    
    /// 生成标准正态分布随机数（Box-Muller 变换）
    /// 
    /// 使用 Box-Muller 变换从均匀分布生成正态分布
    /// 比 ziggurat 算法简单，且对于我们的用例足够快
    #[inline(always)]
    pub fn randn(&mut self) -> f32 {
        use rand::Rng;
        
        let u1: f32 = self.rng.gen();
        let u2: f32 = self.rng.gen();
        
        // Box-Muller 变换
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * PI * u2;
        
        r * theta.cos()
    }
    
    /// 生成指定均值和标准差的正态分布随机数
    #[inline(always)]
    pub fn normal(&mut self, mean: f32, std: f32) -> f32 {
        mean + std * self.randn()
    }
    /// 生成热噪声 (Thermal Noise)
    ///
    /// 基于玻尔兹曼常数和温度的简化模型
    #[inline(always)]
    pub fn thermal_noise(&mut self, temperature_k: f32) -> f32 {
        // 噪声功率与温度成正比
        let noise_std = (temperature_k / 300.0).sqrt() * 0.01;
        self.normal(0.0, noise_std)
    }

    /// 生成串扰噪声 (Crosstalk)
    ///
    /// 模拟相邻波导或通道之间的信号泄漏
    #[inline(always)]
    pub fn crosstalk_noise(&mut self, signal_intensity: f32, coupling_coeff: f32) -> f32 {
        let mean_leakage = signal_intensity * coupling_coeff;
        self.normal(mean_leakage, mean_leakage * 0.1)
    }
}

/// 线程本地随机数生成器池
/// 
/// 每个线程维护独立的 RNG 状态，避免锁竞争
pub struct RngPool {
    seed: u64,
}

impl RngPool {
    pub fn new(seed: u64) -> Self {
        Self { seed }
    }
    
    /// 为当前线程获取 RNG
    /// 
    /// 使用线程 ID 作为种子偏移，确保每个线程的随机序列不同
    pub fn get_thread_rng(&self) -> FastRng {
        use std::thread;
        let thread_id = thread::current().id();
        let thread_seed = self.seed.wrapping_add(hash_thread_id(thread_id));
        FastRng::new(thread_seed)
    }
}

/// 简单的线程 ID 哈希函数
fn hash_thread_id(id: std::thread::ThreadId) -> u64 {
    // 使用 Debug 格式获取线程 ID 的数字表示
    let id_str = format!("{:?}", id);
    let digits: String = id_str.chars().filter(|c| c.is_numeric()).collect();
    digits.parse::<u64>().unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fast_rng() {
        let mut rng = FastRng::new(42);
        
        // 生成一些随机数
        let samples: Vec<f32> = (0..1000).map(|_| rng.randn()).collect();
        
        // 检查均值接近 0
        let mean: f32 = samples.iter().sum::<f32>() / samples.len() as f32;
        assert!(mean.abs() < 0.2, "Mean should be close to 0, got {}", mean);
        
        // 检查标准差接近 1
        let variance: f32 = samples.iter().map(|x| (x - mean).powi(2)).sum::<f32>() 
            / samples.len() as f32;
        let std = variance.sqrt();
        assert!((std - 1.0).abs() < 0.2, "Std should be close to 1, got {}", std);
    }
    
    #[test]
    fn test_normal_distribution() {
        let mut rng = FastRng::new(123);
        
        let mean = 5.0;
        let std = 2.0;
        let samples: Vec<f32> = (0..1000).map(|_| rng.normal(mean, std)).collect();
        
        let sample_mean: f32 = samples.iter().sum::<f32>() / samples.len() as f32;
        assert!((sample_mean - mean).abs() < 0.5, 
            "Sample mean should be close to {}, got {}", mean, sample_mean);
    }
    
    #[test]
    fn test_rng_pool() {
        let pool = RngPool::new(42);
        
        // 获取两个线程的 RNG
        let mut rng1 = pool.get_thread_rng();
        let mut rng2 = pool.get_thread_rng();
        
        // 它们应该生成相同的序列（因为在同一线程）
        let val1 = rng1.randn();
        let val2 = rng2.randn();
        
        // 但是值应该是有效的正态分布样本
        assert!(val1.abs() < 10.0);
        assert!(val2.abs() < 10.0);
    }
}
