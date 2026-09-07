use crate::{
    memory_management::{
        BytesFormat, ManagedMemoryHandle, MemoryLocation, MemoryUsage,
        memory_pool::{MemoryPage, MemoryPool, Slice},
    },
    server::IoError,
    storage::StorageId,
};
use alloc::vec::Vec;
use alloc::collections::BTreeMap;
use core::fmt::Display;
use core::sync::atomic::{AtomicU64, Ordering};

// PATCH (rust-sdt 显存排查)：池级分配统计（每池独立，非全局）。
// reuse_hits = try_reserve 复用空闲切片成功的次数；new_pages = 新开整页次数。
// 若 reuse_hits≈0 而 new_pages 巨大，说明 free 切片从未被复用（钉住/回收路径失效）。
// 环境变量 BURN_SDT_POOL_DIAG=1 时每 50 个新页打印一次统计。
#[derive(Default)]
struct PoolDiag {
    reuse_hits: AtomicU64,
    new_pages: AtomicU64,
}

fn pool_diag_enabled() -> bool {
    static ONCE: core::sync::atomic::AtomicU8 = core::sync::atomic::AtomicU8::new(0);
    match ONCE.load(Ordering::Relaxed) {
        0 => {
            let v = std::env::var("BURN_SDT_POOL_DIAG").map(|v| v == "1").unwrap_or(false);
            ONCE.store(if v { 2 } else { 1 }, Ordering::Relaxed);
            v
        }
        2 => true,
        _ => false,
    }
}

pub struct SlicedPool {
    pages: Vec<(MemoryPage, StorageId)>,
    pages_tmp: Vec<(MemoryPage, StorageId)>,
    page_size: u64,
    alignment: u64,
    max_alloc_size: u64,
    location_base: MemoryLocation,
    /// PATCH：每池独立的诊断计数器
    diag: PoolDiag,
}

impl SlicedPool {
    pub fn new(page_size: u64, max_slice_size: u64, alignment: u64, pool_pos: u8) -> Self {
        Self {
            pages: Vec::new(),
            pages_tmp: Vec::new(),
            page_size,
            alignment,
            max_alloc_size: max_slice_size,
            location_base: MemoryLocation::new(pool_pos, 0, 0),
            diag: PoolDiag::default(),
        }
    }
}

impl MemoryPool for SlicedPool {
    fn accept(&self, size: u64) -> bool {
        self.max_alloc_size >= size
            ||
            // If the size is close to the page size so it doesn't create much fragmentation with
            // unused space.
            match self.page_size.checked_sub(size) {
                Some(diff) => diff * 5 < self.page_size, // 20 % unused space is the max allowed.
                None => false,
            }
    }

    fn find(&self, binding: &super::ManagedMemoryBinding) -> Result<&Slice, IoError> {
        let (page, _) = &self.pages[binding.descriptor().page()];
        page.find(binding)
    }

    fn try_reserve(&mut self, size: u64) -> Option<super::ManagedMemoryHandle> {
        for (page, _) in self.pages.iter_mut() {
            page.coalesce();
            if let Some(handle) = page.try_reserve(size) {
                self.diag.reuse_hits.fetch_add(1, Ordering::Relaxed);
                return Some(handle);
            }
        }

        None
    }

    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(level = "trace", skip(self, storage))
    )]
    fn alloc<Storage: crate::storage::ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        size: u64,
    ) -> Result<super::ManagedMemoryHandle, crate::server::IoError> {
        let storage = storage.alloc(self.page_size)?;

        let storage_id = storage.id;
        let mut location_base = self.location_base;
        location_base.page = self.pages.len() as u16;

        let mut page = MemoryPage::new(storage, self.alignment, location_base);
        let returned = page.try_reserve(size);
        self.pages.push((page, storage_id));

        let new_pages = self.diag.new_pages.fetch_add(1, Ordering::Relaxed) + 1;
        if pool_diag_enabled() && new_pages % 100 == 0 {
            let reuse = self.diag.reuse_hits.load(Ordering::Relaxed);
            // PATCH：强引用计数直方图 + 被钉住切片的来源大小直方图（仅本池）
            let mut hist = [0u64; 4];
            let mut num_pages = 0u64;
            let mut num_slices = 0u64;
            let mut pinned_sizes: BTreeMap<u64, u64> = BTreeMap::new();
            let mut pinned_ids_sample: Vec<u64> = Vec::new();
            for (page, _) in self.pages.iter() {
                num_pages += 1;
                let h = page.strong_count_histogram();
                for i in 0..4 {
                    hist[i] += h[i];
                }
                num_slices += h.iter().sum::<u64>();
                for id in page.pinned_ids() {
                    if let Some(sz) = crate::alloc_sizes::lookup(id) {
                        *pinned_sizes.entry(sz).or_insert(0) += 1;
                    }
                    if pinned_ids_sample.len() < 400 {
                        pinned_ids_sample.push(id);
                    }
                }
            }
            // 只打印出现次数最多的 8 种钉住大小 + 前 24 个钉住 id（id=全局分配序号）
            let mut top: Vec<(u64, u64)> = pinned_sizes.into_iter().collect();
            top.sort_by(|a, b| b.1.cmp(&a.1));
            top.truncate(8);
            let mut ids: Vec<u64> = pinned_ids_sample;
            ids.sort_unstable();
            ids.truncate(24);
            std::eprintln!(
                "[pool-diag] page_size={} new_pages={} pages={} slices={} reuse_hits={} cur_alloc_size={} counts[free,c2,c3,c4+]=[{},{},{},{}] pinned_sizes={:?} pinned_ids={:?}",
                BytesFormat::new(self.page_size),
                new_pages,
                num_pages,
                num_slices,
                reuse,
                size,
                hist[0],
                hist[1],
                hist[2],
                hist[3],
                top,
                ids,
            );
        }

        Ok(returned.expect("effective_size to be smaller than page_size"))
    }

    fn get_memory_usage(&self) -> MemoryUsage {
        let mut usage = MemoryUsage {
            number_allocs: 0,
            bytes_in_use: 0,
            bytes_padding: 0,
            bytes_reserved: 0,
        };

        for (page, _) in self.pages.iter() {
            let current = page.memory_usage();
            usage = usage.combine(current);
        }

        usage
    }

    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(level = "trace", skip(self, storage))
    )]
    fn cleanup<Storage: crate::storage::ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        _alloc_nr: u64,
        explicit: bool,
    ) {
        if !explicit {
            return;
        }

        for (mut page, id) in self.pages.drain(..) {
            page.coalesce();
            let summary = page.summary(false);

            if summary.amount_free == summary.amount_total {
                storage.dealloc(id);
            } else {
                let page_pos = self.pages_tmp.len() as u16;
                page.update_page(page_pos);
                self.pages_tmp.push((page, id));
            }
        }

        core::mem::swap(&mut self.pages, &mut self.pages_tmp);
    }

    /// Binds a user defined [`ManagedMemoryHandle`] to a slice in this memory pool.
    fn bind(
        &mut self,
        reserved: ManagedMemoryHandle,
        assigned: ManagedMemoryHandle,
        cursor: u64,
    ) -> Result<(), IoError> {
        let (page, _) = &mut self.pages[reserved.descriptor().page()];

        page.bind(reserved, assigned, cursor)?;

        Ok(())
    }
}

impl Display for SlicedPool {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        if self.pages.is_empty() {
            return Ok(());
        }

        f.write_fmt(format_args!(
            " - Sliced Pool page_size={} max_alloc_size={}\n",
            BytesFormat::new(self.page_size),
            BytesFormat::new(self.max_alloc_size)
        ))?;

        for (page, id) in self.pages.iter() {
            let summary = page.summary(false);
            f.write_fmt(format_args!(
                "   - Page {id} num_slices={} =>",
                summary.num_total
            ))?;

            let size_free = BytesFormat::new(summary.amount_free);
            let size_full = BytesFormat::new(summary.amount_full);
            let size_total = BytesFormat::new(summary.amount_total);

            f.write_fmt(format_args!(
                " {size_free} free - {size_full} full - {size_total} total\n"
            ))?;
        }

        f.write_fmt(format_args!("\n{}\n", self.get_memory_usage()))?;

        Ok(())
    }
}
