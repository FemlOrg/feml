# Backend 接入指南

本文档定义 `feml-core` 的 backend trait 契约（M2 起冻结，变更走 semver）。
第三方后端（如 `feml-vulkan`、`feml-metal`、`feml-wgpu`）按此实现即可被
`Registry` 发现、被 `GraphBuilder::compile` 使用。

## 1. 实现三个 trait（均在 `feml_core::backend`）

| trait | 职责 | 必实现 |
|---|---|---|
| `BackendRegistrar` | 插件入口：设备枚举 + 评分 | 全部 |
| `BackendDevice` | 设备信息 + 能力查询 + 建流 | `info` / `init_backend` / `supports_op` |
| `Backend` | 执行流：缓冲生命周期 + 图执行 | 全部 |

### 硬性约束

- **`Send + Sync`**：所有三个 trait 均有此超trait。`graph_compute` 可能在工作线程上执行。
- **零 panic**：所有方法返回 `Result`。内核内部可使用 `unsafe`（带 `SAFETY:` 注释），
  但不得泄漏到公共 API 表面。
- **越界即错误**：`write/read/fill` 的字节范围必须在缓冲内，越界返回 `Err`，
  绝不写穿。
- **`graph_compute` 零分配**：plan 是编译产物（句柄 + offset 已固化），执行路径
  不得分配内存（如需临时缓冲，应预分配在 plan 中）。
- **handle 后端私有**：`BufferHandle` 只对创建它的后端有效，不得跨后端混用。

## 2. 最小骨架

```rust
use feml_core::backend::*;

pub struct MyBackend {
    // 你的设备句柄 + 缓冲表（内部用 Mutex/RwLock 保证 Send+Sync）
    inner: std::sync::Arc<std::sync::Mutex<Inner>>,
}

impl Backend for MyBackend {
    fn name(&self) -> &str { "MyBackend" }
    fn create_buffer(&self, size: usize, usage: BufferUsage) -> Result<BufferHandle> { /* ... */ }
    fn release_buffer(&self, handle: BufferHandle) -> Result<()> { /* ... */ }
    fn write(&self, handle: BufferHandle, offset: usize, data: &[u8]) -> Result<()> { /* ... */ }
    fn read(&self, handle: BufferHandle, offset: usize, out: &mut [u8]) -> Result<()> { /* ... */ }
    fn fill(&self, handle: BufferHandle, offset: usize, value: u8, len: usize) -> Result<()> { /* ... */ }
    fn synchronize(&self) -> Result<()> { Ok(()) }
    fn graph_compute(&self, plan: &GraphPlan) -> Result<()> {
        for t in plan.nodes() {
            match t.op {
                // 只实现你支持的算子；其余返回 Err(Error::msg(...))
                Op::Mul => { /* kernel */ }
                _ => return Err(Error::msg(format!("{}: op {} not implemented", self.name(), t.op.name()))),
            }
        }
        Ok(())
    }
    fn supports_op(&self, op: Op, _dtypes: &[DType]) -> bool { matches!(op, Op::Mul) }
}
```

`graph_compute` 中通过 `plan.nodes()` 拿到每个节点的 `PlanTensor`：
- 源：`plan.tensors[t.srcs[i]]`；
- 目标：`t` 自身（其 `buffer`/`offset`/`layout` 即输出位置）；
- 缓冲句柄：`plan.buffers[t.buffer].handle`。

## 3. 设备与注册

```rust
pub struct MyDevice { /* 设备信息 */ }
impl BackendDevice for MyDevice {
    fn info(&self) -> Result<DeviceInfo> { Ok(self.info.clone()) }
    fn init_backend(&self) -> Result<Box<dyn Backend>> { Ok(Box::new(MyBackend::new()?)) }
    fn supports_op(&self, op: Op, dt: &[DType]) -> bool { /* 同 Backend */ }
    fn buffer_from_host_ptr(&self, _p: &mut [u8], _s: usize, _m: usize) -> Result<BufferHandle> {
        Err(Error::msg("MyBackend: host pointers not supported"))
    }
}

pub struct MyRegistrar { devices: Vec<MyDevice> }
impl BackendRegistrar for MyRegistrar {
    fn name(&self) -> &str { "MyBackend" }
    fn device_count(&self) -> usize { self.devices.len() }
    fn device(&self, i: usize) -> Result<Box<dyn BackendDevice>> { /* clone + box */ }
    fn score(&self) -> u32 { if self.devices.is_empty() { 0 } else { 500 } }
}
```

## 4. 评分约定（`open_best`）

- `0` = 当前环境不可用（无设备）；
- CPU 参考值 `100`，通用 GPU 后端参考值 `1000`；
- 专用加速器（且该机器上比通用 GPU 快）可更高。

## 5. 算子分发的推荐结构

参考 `feml-cpu`：`graph_compute` 里 `match op`，每个算子一个内核模块
（`ops/<op>.rs`），内核通过 `feml_cpu::ops::elementwise::Operand` 这类
「指针 + 长度 + offset + layout」结构访问数据。未实现的算子返回
`Error::msg(format!("{backend}: op {op} not implemented"))`。

## 6. 测试要求

每个后端至少提供：
1. 缓冲生命周期测试（write/read/fill/越界拒绝）；
2. 一个算子的数值测试（与参照实现对比）；
3. 与 CPU 后端的跨后端一致性测试（同一 graph，逐元素比较）。
