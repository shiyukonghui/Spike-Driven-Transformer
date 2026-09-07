// PATCH (rust-sdt 显存排查)：分配 id -> 请求大小 注册表。
// 池诊断打印被钉住切片（descriptor 强引用 >1）时，反查其来源分配大小，
// 直接定位"泄漏的是哪些张量"。
// 仅在 BURN_SDT_POOL_DIAG=1 时记录（诊断模式），默认零开销。
use alloc::collections::BTreeMap;
use spin::Mutex;

static ALLOC_SIZES: Mutex<BTreeMap<u64, u64>> = Mutex::new(BTreeMap::new());

static mut ENABLED: bool = false;
static INIT: spin::Once<bool> = spin::Once::new();

fn enabled() -> bool {
    *INIT.call_once(|| {
        let on = std::env::var("BURN_SDT_POOL_DIAG")
            .map(|v| v == "1")
            .unwrap_or(false);
        unsafe {
            ENABLED = on;
        }
        on
    })
}

pub fn record(id: u64, size: u64) {
    if enabled() {
        ALLOC_SIZES.lock().insert(id, size);
    }
}

pub fn lookup(id: u64) -> Option<u64> {
    if !enabled() {
        return None;
    }
    ALLOC_SIZES.lock().get(&id).copied()
}
