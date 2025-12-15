/// 量化模拟器 - 模拟 DAC/ADC 精度限制
/// 
/// 使用高效的位操作实现 k-bit 量化

/// k-bit 量化器
#[derive(Debug, Clone)]
pub struct Quantizer {
    bits: u8,
    max_val: f32,
    min_val: f32,
    scale: f32,
    levels: u32,
}

impl Quantizer {
    /// 创建新的量化器
    /// 
    /// # Arguments
    /// * `bits` - 量化位数（2-8 bit）
    /// * `min_val` - 最小值
    /// * `max_val` - 最大值
    pub fn new(bits: u8, min_val: f32, max_val: f32) -> Self {
        assert!(bits >= 2 && bits <= 8, "Bits must be in range [2, 8]");
        assert!(max_val > min_val, "max_val must be greater than min_val");
        
        let levels = (1u32 << bits) - 1; // 2^bits - 1
        let scale = (levels as f32) / (max_val - min_val);
        
        Self {
            bits,
            max_val,
            min_val,
            scale,
            levels,
        }
    }
    
    /// 对单个值进行量化
    /// 
    /// 算法：
    /// 1. 归一化到 [0, 2^bits - 1]
    /// 2. 四舍五入到最近的整数
    /// 3. 反归一化回原始范围
    #[inline(always)]
    pub fn quantize(&self, val: f32) -> f32 {
        // 裁剪到有效范围
        let clamped = val.clamp(self.min_val, self.max_val);
        
        // 归一化并量化
        let normalized = (clamped - self.min_val) * self.scale;
        let quantized = normalized.round().min(self.levels as f32);
        
        // 反归一化
        self.min_val + quantized / self.scale
    }
    
    /// 批量量化（SIMD 优化潜力）
    pub fn quantize_batch(&self, values: &mut [f32]) {
        for val in values.iter_mut() {
            *val = self.quantize(*val);
        }
    }
}

/// DAC 转换器（数字 -> 模拟）
#[inline(always)]
pub fn dac_convert(digital_val: f32, bits: u8) -> f32 {
    let quantizer = Quantizer::new(bits, 0.0, 1.0);
    quantizer.quantize(digital_val.clamp(0.0, 1.0))
}

/// ADC 转换器（模拟 -> 数字）
#[inline(always)]
pub fn adc_convert(analog_val: f32, bits: u8) -> f32 {
    let quantizer = Quantizer::new(bits, -1.0, 1.0);
    quantizer.quantize(analog_val.clamp(-1.0, 1.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantizer_4bit() {
        let q = Quantizer::new(4, 0.0, 1.0);
        
        // 4-bit 有 16 个级别 (0-15)
        assert_eq!(q.levels, 15);
        
        // 测试边界值
        assert!((q.quantize(0.0) - 0.0).abs() < 1e-6);
        assert!((q.quantize(1.0) - 1.0).abs() < 1e-6);
        
        // 测试中间值
        let mid = q.quantize(0.5);
        assert!((mid - 0.5).abs() < 0.1);
    }
    
    #[test]
    fn test_quantizer_2bit() {
        let q = Quantizer::new(2, -1.0, 1.0);
        
        // 2-bit 只有 4 个级别 (0, 1, 2, 3)
        assert_eq!(q.levels, 3);
        
        // 量化应该产生粗糙的值
        let val = q.quantize(0.1);
        
        // 应该被量化到最近的级别
        let expected_levels = vec![-1.0, -1.0/3.0, 1.0/3.0, 1.0];
        assert!(expected_levels.iter().any(|&level| (val - level).abs() < 0.1));
    }
    
    #[test]
    fn test_quantizer_clipping() {
        let q = Quantizer::new(4, 0.0, 1.0);
        
        // 超出范围的值应该被裁剪
        assert!((q.quantize(-0.5) - 0.0).abs() < 1e-6);
        assert!((q.quantize(1.5) - 1.0).abs() < 1e-6);
    }
    
    #[test]
    fn test_quantize_batch() {
        let q = Quantizer::new(4, 0.0, 1.0);
        let mut values = vec![0.1, 0.5, 0.9, 1.2];
        
        q.quantize_batch(&mut values);
        
        // 所有值应该在 [0, 1] 范围内
        for val in values.iter() {
            assert!(*val >= 0.0 && *val <= 1.0);
        }
    }
    
    #[test]
    fn test_dac_convert() {
        let result = dac_convert(0.5, 4);
        assert!(result >= 0.0 && result <= 1.0);
    }
    
    #[test]
    fn test_adc_convert() {
        let result = adc_convert(0.5, 8);
        assert!(result >= -1.0 && result <= 1.0);
    }
}
