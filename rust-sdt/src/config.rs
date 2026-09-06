//! 模型结构配置：与 PyTorch 侧 `SpikeDrivenTransformer` 的超参数一一对应。
//!
//! 默认值对应 conf/cifar10/2_256_300E_t4.yml：
//! - dim=256, layer=2, num_heads=8, img_size=32, patch_size=16, T=4
//! - pooling_stat="0011", mlp_ratio=4, in_channels=3, num_classes=10

/// 模型结构配置
// 与 PyTorch 侧 conf/cifar10/2_256_300E_t4.yml 对齐的完整结构信息；部分字段
// （img_size_h/w、patch_size、pooling_stat）当前前向按硬编码约定展开，暂不读取，
// 保留以维持配置完整性（与 conf 文件一一对应）
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct SdtConfig {
    /// 输入图像高度
    pub img_size_h: usize,
    /// 输入图像宽度
    pub img_size_w: usize,
    /// patch 大小（SPS 4 级下采样后的空间缩放）
    pub patch_size: usize,
    /// 输入通道数
    pub in_channels: usize,
    /// 分类类别数
    pub num_classes: usize,
    /// 嵌入维度
    pub embed_dims: usize,
    /// 注意力头数
    pub num_heads: usize,
    /// MLP 隐层扩张比
    pub mlp_ratio: usize,
    /// Transformer block 数量
    pub depths: usize,
    /// 时间步 T
    pub time_steps: usize,
    /// SPS 每级是否做 maxpool（"0011" => 第 0、1 级不池化，第 2、3 级池化）
    pub pooling_stat: [bool; 4],
}

impl Default for SdtConfig {
    fn default() -> Self {
        Self {
            img_size_h: 32,
            img_size_w: 32,
            patch_size: 16,
            in_channels: 3,
            num_classes: 10,
            embed_dims: 256,
            num_heads: 8,
            mlp_ratio: 4,
            depths: 2,
            time_steps: 4,
            pooling_stat: [false, false, true, true],
        }
    }
}

// 辅助方法保留（配置完整性；部分暂未在前向中调用）
#[allow(dead_code)]
impl SdtConfig {
    /// SPS 输出的特征图高度（H_img / patch_size）
    pub fn feat_h(&self) -> usize {
        self.img_size_h / self.patch_size
    }

    /// SPS 输出的特征图宽度（W_img / patch_size）
    pub fn feat_w(&self) -> usize {
        self.img_size_w / self.patch_size
    }

    /// patch token 数量
    pub fn num_patches(&self) -> usize {
        self.feat_h() * self.feat_w()
    }

    /// MLP 隐层维度
    pub fn mlp_hidden(&self) -> usize {
        self.embed_dims * self.mlp_ratio
    }
}
