//! CIFAR-10 数据加载器（从 torchvision 导出的 npz 读取），供 Burn 训练循环使用。
//!
//! 数据来源：`scripts/export_cifar10.py` 生成的 rust-sdt/artifacts/cifar10_data.npz，
//! 包含 train_x/train_y/test_x/test_y（图像 [N,3,32,32] 归一化到 [0,1]，标签 int64）。
//!
//! 本模块提供：
//! - `Cifar10Npz`：一次性载入全部数据（扁平 CHW 连续存储）
//! - `get_batch`：按索引组装 [T,B,3,32,32] 静态帧批次（T 个时间步复制同一帧）与 Int 标签 [B]
//! - `shuffled_indices`：固定种子的数据混洗（保证训练可复现）

use burn::prelude::*;

/// 图像通道数（CIFAR-10 为 RGB 3 通道）
pub const C: usize = 3;
/// 图像高度
pub const H: usize = 32;
/// 图像宽度
pub const W: usize = 32;

/// 数据集划分
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Split {
    /// 训练集
    Train,
    /// 测试集
    Test,
}

/// CIFAR-10 数据集（从 npz 一次性载入内存）
///
/// 图像按 扁平 CHW 连续存储：sample i 的图像占据
/// `pixels[i*C*H*W .. (i+1)*C*H*W]`。
pub struct Cifar10Npz {
    /// 训练图像（扁平 CHW 连续），长度 = n_train * 3 * 32 * 32
    pub train_x: Vec<f32>,
    /// 训练标签，长度 = n_train
    pub train_y: Vec<i64>,
    /// 测试图像（扁平 CHW 连续），长度 = n_test * 3 * 32 * 32
    pub test_x: Vec<f32>,
    /// 测试标签，长度 = n_test
    pub test_y: Vec<i64>,
    /// 训练样本数
    pub n_train: usize,
    /// 测试样本数
    pub n_test: usize,
}

impl Cifar10Npz {
    /// 从 NPZ 文件加载数据集（复用 tensor_io::read_npz，统一 f32 中转）。
    ///
    /// 标签在 npy 中是 "<i8"（int64），经 f32 中转对 0..10 的小整数无损，
    /// 直接 `f32 as i64` 即可。
    pub fn load(path: &str) -> Self {
        // 读取全部数组（键名：train_x / train_y / test_x / test_y）
        let npz = crate::tensor_io::read_npz(path);

        // 取出各数组（shape 校验在下方按维度展开）
        let train_x = npz.get("train_x").expect("npz 缺少 train_x");
        let train_y = npz.get("train_y").expect("npz 缺少 train_y");
        let test_x = npz.get("test_x").expect("npz 缺少 test_x");
        let test_y = npz.get("test_y").expect("npz 缺少 test_y");

        // train_x: [N, 3, 32, 32]
        let sx = train_x.shape();
        assert!(
            sx.len() == 4 && sx[1] == C && sx[2] == H && sx[3] == W,
            "train_x 形状应为 [N,3,32,32]，实际 {:?}",
            sx
        );
        let n_train = sx[0];
        // test_x: [N, 3, 32, 32]
        let se = test_x.shape();
        assert!(
            se.len() == 4 && se[1] == C && se[2] == H && se[3] == W,
            "test_x 形状应为 [N,3,32,32]，实际 {:?}",
            se
        );
        let n_test = se[0];

        // 图像：ArrayD -> 扁平 Vec（已是 C 顺序连续内存）
        let train_x = train_x.clone().into_raw_vec();
        let test_x = test_x.clone().into_raw_vec();

        // 标签：f32 中转（0..10 无精度损失）转 i64
        let train_y: Vec<i64> = train_y.iter().map(|&v| v as i64).collect();
        let test_y: Vec<i64> = test_y.iter().map(|&v| v as i64).collect();
        assert_eq!(train_y.len(), n_train, "train_y 长度与 train_x 不匹配");
        assert_eq!(test_y.len(), n_test, "test_y 长度与 test_x 不匹配");

        Self {
            train_x,
            train_y,
            test_x,
            test_y,
            n_train,
            n_test,
        }
    }

    /// 组装一个批次张量。
    ///
    /// 返回值：
    /// - 图像 [T, B, 3, 32, 32]（Float）：T 个时间步均为同一静态帧（reshape 复制）
    /// - 标签 [B]（Int）：供 burn 0.18 的 `CrossEntropyLoss::forward` 直接使用
    ///
    /// `indices` 中的每个索引必须 < 对应划分的样本数。
    /// 泛型后端：训练时以 autodiff 后端调用（与 autodiff 权重匹配），
    /// 校准/静态统计时以 wgpu 后端调用；设备同型，张量构造路径一致。
    pub fn get_batch<B: Backend>(
        &self,
        split: Split,
        indices: &[usize],
        t: usize,
        device: &B::Device,
    ) -> (Tensor<B, 5>, Tensor<B, 1, Int>) {
        // 依据划分选择数据源
        let (pixels, labels, n) = match split {
            Split::Train => (&self.train_x, &self.train_y, self.n_train),
            Split::Test => (&self.test_x, &self.test_y, self.n_test),
        };
        let b = indices.len();
        assert!(b > 0, "indices 不能为空");
        assert!(t > 0, "时间步 T 必须大于 0");

        // 校验索引范围
        for &i in indices {
            assert!(i < n, "样本索引 {} 越界（该划分共 {} 个样本）", i, n);
        }

        // 组装 [T*B, 3*32*32] 的帧数据：T 个时间步复制同一帧
        let frame = C * H * W;
        let mut batch = vec![0f32; t * b * frame];
        for ti in 0..t {
            let base = ti * b * frame; // 该时间步在扁平 batch 中的起始偏移
            for (bi, &si) in indices.iter().enumerate() {
                let src = si * frame;
                let dst = base + bi * frame;
                batch[dst..dst + frame].copy_from_slice(&pixels[src..src + frame]);
            }
        }

        // 参照 model.rs 的 t4/t2 写法：先建 1 维 from_floats，再 reshape 成 5 维
        let images = Tensor::<B, 1>::from_floats(batch.as_slice(), device)
            .reshape([t, b, C, H, W]);

        // 标签构造 Int 张量 [B]（I64 数据，经 TensorData::new 直接构造）
        let y_i64: Vec<i64> = indices.iter().map(|&i| labels[i]).collect();
        let targets =
            Tensor::<B, 1, Int>::from_data(burn::tensor::TensorData::new(y_i64, [b]), device);

        (images, targets)
    }
}

/// 固定种子的数据混洗：返回 [0, n) 的随机排列。
///
/// 使用 rand 0.8 的 StdRng::seed_from_u64 + SliceRandom::shuffle，
/// 相同 seed 产生相同排列，保证训练顺序可复现。
pub fn shuffled_indices(n: usize, seed: u64) -> Vec<usize> {
    use rand::seq::SliceRandom;
    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut v: Vec<usize> = (0..n).collect();
    v.shuffle(&mut rng);
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    /// npz 数据文件路径（相对于仓库根目录运行 cargo test 时有效）
    const NPZ: &str = "artifacts/cifar10_data.npz";

    /// 验证 npz 加载后样本数与张量组装形状正确，且时间步复制语义成立
    #[test]
    fn test_cifar10_npz_load_and_get_batch() {
        // 跳过条件：npz 尚未导出时（CI/编译期检查）直接通过
        if !std::path::Path::new(NPZ).exists() {
            eprintln!("跳过：{} 不存在（未运行导出脚本）", NPZ);
            return;
        }
        let data = Cifar10Npz::load(NPZ);
        // 导出脚本约定：train 8000 / test 2000
        assert_eq!(data.n_train, 8000);
        assert_eq!(data.n_test, 2000);
        // 扁平数据长度校验
        assert_eq!(data.train_x.len(), 8000 * C * H * W);
        assert_eq!(data.test_x.len(), 2000 * C * H * W);
        // 标签取值范围 0..10
        assert!(data.train_y.iter().all(|&y| (0..10).contains(&y)));
        assert!(data.test_y.iter().all(|&y| (0..10).contains(&y)));

        // 组装批次：T=4, B=2（显式指定 wgpu 后端，泛型函数可被任意后端调用）
        let device = Default::default();
        let (images, targets) =
            data.get_batch::<crate::model::BackendAdapter>(Split::Train, &[0, 1], 4, &device);
        assert_eq!(images.dims(), [4, 2, 3, 32, 32]);
        assert_eq!(targets.dims(), [2]);

        // 时间步复制语义：t0 与 t1 的同一样本完全相同
        let imgs = images
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .expect("读取图像数据失败");
        let frame = C * H * W;
        // t0 样本 0 与 t1 样本 0（batch[0..frame] 与 batch[b*frame..b*frame+frame]）
        assert_eq!(imgs[0..frame], imgs[2 * frame..2 * frame + frame]);

        // 标签与原始数据一致
        let ys = targets
            .into_data()
            .convert::<i64>()
            .to_vec::<i64>()
            .expect("读取标签失败");
        assert_eq!(ys, vec![data.train_y[0], data.train_y[1]]);
    }

    /// 验证 shuffled_indices：确定性 + 排列完整性
    #[test]
    fn test_shuffled_indices() {
        let a = shuffled_indices(100, 42);
        let b = shuffled_indices(100, 42);
        // 相同种子 => 相同排列
        assert_eq!(a, b);
        // 是 0..100 的一个排列
        let mut sorted = a.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..100).collect::<Vec<usize>>());
        // 不同种子大概率不同（小概率碰撞，可接受）
        let c = shuffled_indices(100, 7);
        assert_ne!(a, c);
    }
}
