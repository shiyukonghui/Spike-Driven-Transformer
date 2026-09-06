//! NPZ/NPY 张量读写工具：读取 PyTorch 导出的权重与中间张量。
//!
//! 使用 zip + 手写 NPY 解析（避免额外依赖版本冲突），统一输出 f32。

use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use ndarray::{ArrayD, IxDyn};
use std::collections::HashMap;
use std::path::Path;

/// 单个 NPY 数组（键名 + 形状 + f32 数据），用于把张量写出为 NPZ。
pub struct NpyArray {
    /// 数组键名（不含 .npy 后缀）
    pub name: String,
    /// 数组形状
    pub shape: Vec<usize>,
    /// C 顺序 f32 数据
    pub data: Vec<f32>,
}

impl NpyArray {
    /// 从任意 burn 张量提取形状与数据（泛型后端，统一 f32 输出）。
    fn from_tensor<B: Backend, const D: usize>(t: &Tensor<B, D>, name: &str) -> Self {
        NpyArray {
            name: name.to_string(),
            shape: t.dims().to_vec(),
            data: t
                .clone()
                .into_data()
                .convert::<f32>()
                .to_vec::<f32>()
                .expect("读取张量数据失败"),
        }
    }

    /// 从 1 维张量构造
    pub fn from_tensor1<B: Backend>(t: &Tensor<B, 1>, name: &str) -> Self {
        Self::from_tensor(t, name)
    }

    /// 从 2 维张量构造
    pub fn from_tensor2<B: Backend>(t: &Tensor<B, 2>, name: &str) -> Self {
        Self::from_tensor(t, name)
    }

    /// 从 4 维张量构造
    pub fn from_tensor4<B: Backend>(t: &Tensor<B, 4>, name: &str) -> Self {
        Self::from_tensor(t, name)
    }
}

/// 把全部数组写出为 NPZ（zip 容器，每个条目为 NPY v1.0 格式，f32 小端）。
/// 条目命名（键名 + .npy 后缀）与 scripts/export_reference.py 的 np.savez 约定一致，
/// 可被本模块 read_npz 与 numpy 直接读取。
pub fn write_npz(path: &str, arrays: &[NpyArray]) {
    // 创建文件与 zip 容器
    let file = std::fs::File::create(Path::new(path)).expect("无法创建 NPZ 文件");
    let mut zip = zip::ZipWriter::new(file);
    let options = zip::write::FileOptions::default();
    for arr in arrays {
        // zip 条目名：键名 + .npy 后缀（numpy 读取约定）
        zip.start_file(format!("{}.npy", arr.name), options)
            .expect("写入 zip 条目失败");
        // 构造 shape 元组文本：1 维必须带尾逗号（如 "(n,)"）
        let shape_str = match arr.shape.len() {
            0 => String::new(),
            1 => format!("{},", arr.shape[0]),
            _ => arr
                .shape
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join(", "),
        };
        // NPY v1.0 头（与 numpy 写端格式一致）：{'descr': '<f4', 'fortran_order': False, 'shape': (...)}
        let mut header = format!(
            "{{'descr': '<f4', 'fortran_order': False, 'shape': ({})",
            shape_str
        );
        // 头部总长（前缀 10 字节 + 头文本 + 结尾 '}'）对齐到 64 字节，用空格填充（numpy 惯例）
        let total_without_end = 10 + header.len() + 1;
        let pad = (64 - total_without_end % 64) % 64;
        header.push_str(&" ".repeat(pad + 1));
        header.push('}');
        // 写 NPY 魔数、版本、头长度、头
        let mut buf: Vec<u8> = Vec::with_capacity(10 + header.len() + arr.data.len() * 4);
        buf.extend_from_slice(b"\x93NUMPY");
        buf.push(1); // 主版本 v1.0
        buf.push(0); // 次版本
        let header_len = header.len() as u16;
        buf.extend_from_slice(&header_len.to_le_bytes());
        buf.extend_from_slice(header.as_bytes());
        // f32 小端数据（C 顺序）
        for v in &arr.data {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        // ZipWriter 实现了 std::io::Write，写整个 NPY 缓冲
        std::io::Write::write_all(&mut zip, &buf).expect("写入 NPY 数据失败");
    }
    // 显式 finish 确保中央目录完整落盘
    zip.finish().expect("关闭 NPZ 文件失败");
}

/// 从 NPZ 文件读取全部数组（键名为去掉 .npy 后缀后的名字）。
pub fn read_npz(path: &str) -> HashMap<String, ArrayD<f32>> {
    let file = std::fs::File::open(Path::new(path)).expect("无法打开 NPZ 文件");
    let mut zip = zip::ZipArchive::new(file).expect("NPZ 不是有效的 zip 容器");
    let mut out = HashMap::new();
    for i in 0..zip.len() {
        let mut entry = zip.by_index(i).expect("读取 zip 条目失败");
        let name = entry.name().to_string();
        let key = name.trim_end_matches(".npy").to_string();
        let mut buf = Vec::with_capacity(entry.size() as usize);
        std::io::Read::read_to_end(&mut entry, &mut buf).expect("读取条目内容失败");
        let arr = parse_npy_f32(&buf).expect("解析 NPY 失败");
        out.insert(key, arr);
    }
    out
}

/// 解析 NPY v1.0 格式（f32 / f64 / i64 等常见类型，统一转 f32）。
pub fn parse_npy_f32(data: &[u8]) -> Result<ArrayD<f32>, String> {
    // 魔数 \x93NUMPY
    if data.len() < 10 || &data[0..6] != b"\x93NUMPY" {
        return Err("不是 NPY 文件".into());
    }
    let major = data[6];
    let (header_len, offset) = if major == 1 {
        (u16::from_le_bytes([data[8], data[9]]) as usize, 10)
    } else {
        (u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize, 12)
    };
    let header = std::str::from_utf8(&data[offset..offset + header_len])
        .map_err(|e| e.to_string())?;
    // 解析 dict：'descr': '<f4', 'fortran_order': False, 'shape': (...)
    let descr = extract_str(header, "'descr'").ok_or("缺少 descr")?;
    let shape = extract_shape(header).ok_or("缺少 shape")?;
    let fortran = header.contains("'fortran_order': True");
    let data_start = offset + header_len;
    let raw = &data[data_start..];

    let mut arr = match descr.trim_matches('\'') {
        "<f4" | "f4" | "|f4" => {
            let n: usize = shape.iter().product();
            let mut v = vec![0f32; n];
            for (i, chunk) in raw.chunks_exact(4).take(n).enumerate() {
                v[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            }
            ArrayD::from_shape_vec(IxDyn(&shape), v).map_err(|e| e.to_string())?
        }
        "<f8" | "f8" => {
            let n: usize = shape.iter().product();
            let mut v = vec![0f32; n];
            for (i, chunk) in raw.chunks_exact(8).take(n).enumerate() {
                v[i] = f64::from_le_bytes([
                    chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
                ]) as f32;
            }
            ArrayD::from_shape_vec(IxDyn(&shape), v).map_err(|e| e.to_string())?
        }
        // 整数类型（bool/i8/i16/i32/i64 等，统一转 f32）
        "|b1" | "b1" => {
            let n: usize = shape.iter().product();
            let v: Vec<f32> = raw.iter().take(n).map(|&b| (b != 0) as i32 as f32).collect();
            ArrayD::from_shape_vec(IxDyn(&shape), v).map_err(|e| e.to_string())?
        }
        "|i1" | "<i1" | "i1" => {
            let n: usize = shape.iter().product();
            let v: Vec<f32> = raw.iter().take(n).map(|&b| b as i8 as f32).collect();
            ArrayD::from_shape_vec(IxDyn(&shape), v).map_err(|e| e.to_string())?
        }
        "|i2" | "<i2" | "i2" => {
            let n: usize = shape.iter().product();
            let mut v = vec![0f32; n];
            for (i, chunk) in raw.chunks_exact(2).take(n).enumerate() {
                v[i] = i16::from_le_bytes([chunk[0], chunk[1]]) as f32;
            }
            ArrayD::from_shape_vec(IxDyn(&shape), v).map_err(|e| e.to_string())?
        }
        "|i4" | "<i4" | "i4" => {
            let n: usize = shape.iter().product();
            let mut v = vec![0f32; n];
            for (i, chunk) in raw.chunks_exact(4).take(n).enumerate() {
                v[i] = i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as f32;
            }
            ArrayD::from_shape_vec(IxDyn(&shape), v).map_err(|e| e.to_string())?
        }
        "|i8" | "<i8" | "i8" => {
            let n: usize = shape.iter().product();
            let mut v = vec![0f32; n];
            for (i, chunk) in raw.chunks_exact(8).take(n).enumerate() {
                v[i] = i64::from_le_bytes([
                    chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
                ]) as f32;
            }
            ArrayD::from_shape_vec(IxDyn(&shape), v).map_err(|e| e.to_string())?
        }
        other => return Err(format!("暂不支持的 NPY dtype: {}", other)),
    };
    if fortran {
        // Fortran 顺序重排为 C 顺序
        arr = permute_fortran_to_c(arr, &shape);
    }
    Ok(arr)
}

/// 从 header dict 中提取字符串值。
fn extract_str<'a>(header: &'a str, key: &str) -> Option<&'a str> {
    let pos = header.find(key)?;
    let rest = &header[pos + key.len()..];
    let start = rest.find('\'')? + 1;
    let end = rest[start..].find('\'')? + start;
    Some(&rest[start..end])
}

/// 从 header dict 中提取 shape 元组。
fn extract_shape(header: &str) -> Option<Vec<usize>> {
    let pos = header.find("'shape'")?;
    let rest = &header[pos..];
    let open = rest.find('(')?;
    let close = rest[open..].find(')')? + open;
    let inner = &rest[open + 1..close];
    Some(
        inner
            .split(',')
            .filter_map(|s| s.trim().parse::<usize>().ok())
            .collect(),
    )
}

/// 将 Fortran 顺序数据重排为 C 顺序（递归转置）。
fn permute_fortran_to_c(arr: ArrayD<f32>, shape: &[usize]) -> ArrayD<f32> {
    // 读取时按列主序解释：等效于先 reshape 再 transpose 全轴反转
    let n: usize = shape.iter().product();
    let flat = arr.into_raw_vec();
    let mut strides_f = vec![1usize; shape.len()];
    for i in (0..shape.len() - 1).rev() {
        strides_f[i] = strides_f[i + 1] * shape[i + 1];
    }
    let mut strides_c = vec![1usize; shape.len()];
    for i in (1..shape.len()).rev() {
        strides_c[i] = strides_c[i - 1] * shape[i];
    }
    let mut out = vec![0f32; n];
    // 枚举 C 顺序索引，映射到 Fortran 线性偏移
    let dims = shape.to_vec();
    let mut idx = vec![0usize; dims.len()];
    for (c_off, val) in out.iter_mut().enumerate() {
        let mut f_off = 0usize;
        let mut rem = c_off;
        for (d, &dim) in dims.iter().enumerate().rev() {
            idx[d] = rem % dim;
            rem /= dim;
        }
        for (d, &i) in idx.iter().enumerate() {
            f_off += i * strides_f[d];
        }
        *val = flat[f_off];
    }
    let _ = strides_c;
    ArrayD::from_shape_vec(IxDyn(&dims), out).expect("重排失败")
}
