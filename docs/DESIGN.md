# Feml 2.0 重新设计方案

> 状态：草案（v0.2）｜日期：2026-08-07
> 目标：在继承 ggml 核心优势（零分配执行、静态计算图、多后端、量化）的同时，用 Rust 的类型安全与生态解决 ggml 的固有缺陷，并超越它。
>
> **定位（v0.2 明确）**：本项目是 **LLM 推理专用高性能算子库**——不做训练、不做 autograd、不做优化器。所有设计围绕推理（prefill/decode、KV cache、权重量化、静态形状）展开。这既是简化（砍掉反向/梯度体系），也是超越 ggml 的机会（推理专属抽象：KV cache 一等公民、decode 模式、图融合、kernel autotune）。

---

## 1. 背景与目标

Feml 是 ggml 的 Rust 重写项目。ggml 支撑了 llama.cpp 生态（约 100 个算子、14 个后端、20+ 量化类型、GGUF 模型格式），但其 C 语言本质带来一系列无法在 C 层面解决的问题。本次重新设计的目标：

1. **继承 ggml 的优点**（§3）：零运行时分配、静态图 + 延迟执行、五层后端抽象、异步执行模型、量化体系、持久线程池。
2. **修复 ggml 的缺点**（§4）：内存安全、错误传播、类型安全、生态分发、后端代码复用、异步生态集成。
3. **超越 ggml**：编译期内存规划（Rust ownership 红利）、类型安全的图构建、可插拔算子注册、crates.io 分发、结构化设备信息、完善的测试与基准体系。

---

## 2. 现状审查结论（为什么要重写）

对当前代码库（42 个文件，约 4000 行）的全面审查发现：

| 严重度 | 问题 | 位置 |
|---|---|---|
| P0 | **main 分支无法编译**（默认 32 错误 / opencl 33 错误）：backend trait 重构后 CPU/OpenCL 未同步，三套不兼容 trait 并存 | `src/backend.rs` vs `src/cpu/*`、`src/opencl/*` |
| P0 | **stride/nbytes 内存计算错误**：stride[1] 公式忽略 shape[0]；data_type 表把整型误标 quantized、block_size 全等于 type_size；`[2,3]` F32 张量算成 10 字节（应为 24） | `src/context.rs:174-175`、`src/data_type.rs:31-40`、`src/layout.rs:36` |
| P0 | 无 CI、无编译门槛，合并即坏 | 仓库无 `.github/` |
| P1 | 全链路 `Rc<RefCell>`（Tensor/Context/Graph），**!Send + !Sync**，多线程/多设备执行不可能；context.rs 注释自称 Arc 线程安全，实现却是 Rc | `src/tensor.rs:84`、`src/context.rs:77` |
| P1 | `todo!()`/`unwrap()` 遍布公共路径（OpenCL `info()`、`reset()`、downcast unwrap、CUDA kernel expect） | `src/opencl/backend_device.rs:34` 等 |
| P1 | 设备支持靠字符串匹配（非 Intel/Qualcomm 全拒；Qualcomm 还错设成 Intel 族） | `src/opencl/backend_device.rs:83-92` |
| P1 | `ObjectPool` 无任何 `put()` 调用，形同虚设；README 宣称 "zero runtime memory allocation" 无任何机制支撑 | `src/object_pool.rs` |
| P1 | 图构建为递归 DFS（深图栈溢出风险）+ expand 模式下 use_count 不重置 | `src/compute_graph.rs:109-138` |
| P1 | OpenCL 注册器使用 `static mut` + 裸指针（edition 2024 编译错误级问题） | `src/opencl/backend_register.rs:11` |
| P2 | 全库仅实现 1 个算子（mul）；CPU `graph_compute` 直接返回 Unsupported | 各 backend |
| P2 | `Tensor::clone()` 共享底层（语义陷阱）；`Deref<Target=RefCell>` 泄露内部可变性 | `src/tensor.rs:264-270, 383-393` |
| P2 | 拼写错误（`cacacity`、`aysnc`、`Famlily`）；死代码（`TensorIdArray` 等） | 多处 |
| P2 | 测试全部为元数据测试，无任何数值正确性测试 | `tests/` |

**结论**：核心骨架概念（Context/Registry/ne-nb 布局思路）有价值，但实现路径（Rc<RefCell> 句柄 + Any downcast + 字符串匹配）是死胡同，需要重新设计。

---

## 3. ggml 必须继承的 8 个优点（已对照源码验证）

1. **零运行时分配**：`ggml_init` 竞技场 + tensor 扁平结构（`op_params` 定长数组，无 per-tensor 堆分配，`ggml.h:660`）+ `ggml-alloc.c` 图感知分配器（liveness + inplace 复用）。
2. **静态图 + 延迟执行**：`ggml_build_forward_expand` 建图与 `ggml_graph_compute` 执行分离；执行期零分配、零建结构。
3. **五层后端抽象**（`ggml-backend-impl.h`）：`reg(插件) → device(能力) → backend/stream(执行流) → buffer(数据访问) → buffer_type(内存策略)`。
4. **异步执行模型**：`set/get_tensor_async`、`cpy_tensor_async`、`synchronize`、`event_record/event_wait`（多流依赖同步）、`graph_plan`（同拓扑图复用 plan）。
5. **能力查询而非硬编码**：`supports_op` / `offload_op` / `supports_buft`，设备自声明能力。
6. **量化体系**：`GGML_TYPE_SIZE / GGML_BLCK_SIZE` 类型表 + block 内存布局 + 专用内核。
7. **持久线程池**（`ggml-cpu.c:472`）：cond-var 工作队列 + 每算子按 `(ith, n_threads)` 并行分片 + abort/pause/priority。
8. **图去重 + use_counts**（`ggml.c:229`）：hash set 去重 + 每节点引用计数，是 liveness 分配与 inplace 决策的基础。

## 4. ggml 必须修复的 7 个缺点

1. 无类型安全：裸指针 + `GGML_ASSERT` 直接 abort，错误不可恢复。
2. 内存所有权模糊：view/data/buffer 归属靠约定 → UAF/内存错乱。
3. 14 个后端重复实现内核，op × dtype × 后端组合爆炸，人力维护。
4. 无包生态：源码编译、无版本管理、无文档、几乎无测试。
5. 单线程建图，无法与 Rust 异步生态（tokio）集成。
6. 手动 vtable 样板 + 弱版本协商，插件开发门槛高。
7. 诊断手段原始：无结构化错误/日志/性能接口。

---

## 5. 五大核心设计决策

### 决策 1：两阶段 API —— Builder → Compile → Execute

```
GraphBuilder（建图期，可分配）──compile()──▶ GraphPlan（Send + Sync，不可变）──execute()──▶ 结果
```

- **建图期**：`GraphBuilder` 持有 bump arena；`Tensor` 是轻量 `TensorId(u32)` 句柄（Copy）。可放心使用 `HashMap`/`Vec`。所有 shape/dtype 校验在此返回 `Result`。
- **编译期**：一次完成拓扑排序、内核选择、**内存规划（liveness 分析）**，产出不可变、`Send` 的 `GraphPlan`。
- **执行期**：`plan.execute(&ctx)` 纯读写已规划内存，**零分配、零分支决策、无锁**。算子按预先分好的线程分片直接执行。

> 对比：candle 单阶段（方便但无法零分配）；ggml 运行时 alloc（内存规划与执行分离为两个手动步骤）。Rust 的 ownership 允许我们把内存规划**固化进编译产物**——这是相对 ggml 的架构级红利。

### 决策 2：张量 = `TensorId` + 内存句柄（废弃 Rc<RefCell>）

```
TensorId(u32)                            // Copy；进 plan、跨线程、跨设备
TensorView { id, offset, len, layout }   // 执行期只读描述，栈上可放
DeviceBuffer { base, size, device }      // 由 MemoryPlanner 统一分配，张量 = {buffer_id, offset}
```

- 元数据在 `GraphBuilder` 的 bump arena（仿 ggml 竞技场，safe Rust 表达）；
- 数据在 `DeviceBuffer`，张量只是 `{buffer_id, offset}`；
- 视图 = `{view_src: TensorId, offset, shape, stride}`，共享父内存（ggml `view_src/view_offs` 语义，`ggml.h:682-683`）；
- plan 天然 `Send`，线程池/GPU 流随意传递；无借用冲突、无 clone 语义陷阱。

### 决策 3：MemoryPlanner —— ggml-alloc 算法的 Rust 移植（核心价值点）

算法完全可移植且大部分可 safe 实现（`unsafe` 仅出现在最后写指针）：

1. **liveness**：编译期遍历图，计算每张量 `use_counts`；
2. **逆拓扑序分配**：张量最后一次使用后释放回 free-list；
3. **inplace 复用**：`op_can_inplace` 表 + `n_children == 1` + 同 layout（`ggml-alloc.c:610-653` 完整规则）；
4. **视图共享父块** + 多余空间回收（`ggml_gallocr_free_extra_space`）；
5. **对齐与 reuse_factor**：chunk 列表 + free-list + 可配置复用冗余（`ggml_dyn_tallocr_alloc`）。

输出 `AllocPlan: { buffer_id, offset }` 固化进 `GraphPlan`。执行期按 plan 直接寻址，分配器无需存在。

### 决策 4：后端抽象 —— 五层 trait（关联类型，零 Any downcast）

```rust
pub trait BackendBufferType {                       // = ggml buffer_type：内存策略
    fn alloc(&self, size: usize, usage: Usage) -> Result<Box<dyn BackendBuffer>>;
    fn alignment(&self) -> usize;
    fn is_host(&self) -> bool;                      // 可否 mmap/指针导入
}
pub trait BackendBuffer {                           // = ggml buffer：张量数据访问
    fn write(&self, tv: &TensorView, data: &[u8], offset: usize) -> Result<()>;
    fn read(&self, tv: &TensorView, out: &mut [u8], offset: usize) -> Result<()>;
    fn memset(&self, tv: &TensorView, value: u8, offset: usize, size: usize) -> Result<()>;
    fn copy(&self, src: &TensorView, dst: &TensorView) -> Result<()>;  // 可跨缓冲
}
pub trait Backend {                                 // = ggml backend/stream：执行流
    fn graph_compute(&self, plan: &GraphPlan) -> Result<()>;
    fn synchronize(&self) -> Result<()>;
    fn copy_async(&self, src: &TensorView, dst: &TensorView) -> Result<()>;
    // 多流依赖：event_record / event_wait
}
pub trait BackendDevice {                           // = ggml device：能力声明
    fn info(&self) -> DeviceInfo;                   // 结构化信息，禁止字符串匹配
    fn supports_op(&self, op: Op, src_dtypes: &[DType]) -> bool;
    fn offload_op(&self, op: Op) -> bool;
    fn buffer_type(&self) -> Box<dyn BackendBufferType>;
    fn buffer_from_host_ptr(&self, ptr: *mut u8, size: usize, max_tensor: usize)
        -> Result<Box<dyn BackendBuffer>>;          // mmap / GGUF 零拷贝
}
pub trait BackendRegistrar {                        // = ggml reg：插件入口
    fn name(&self) -> &str;
    fn devices(&self) -> Vec<Box<dyn BackendDevice>>;
    fn score(&self) -> u32;                         // ggml_backend_score：open_best 按分数排序
}
```

- **`as_any`/downcast 全部删除**：具体类型用 `Box<dyn>` + 关联类型，内核注册走函数表；
- 能力用结构化枚举/位标志，不用字符串；
- 错误走 `Result`，无 panic/assert；
- 插件：`cdylib` + 版本协商（仿 `GGML_BACKEND_API_VERSION`）+ `score()`。

### 决策 5：算子 = 注册表 + 定长参数

```rust
pub enum Op { Mul, Add, MulMat { alpha: f32 }, RmsNorm { eps: f32 }, Rope { .. }, .. }
// 参数编码为定长 [i32; N]（仿 ggml op_params），算子对象无堆分配

// 全局注册表：(op, src0_dtype, src1_dtype) → 内核函数
pub type KernelFn = fn(&[TensorView], &mut TensorView, &ThreadCtx) -> Result<()>;
pub struct OpRegistry { /* signature → KernelFn，编译期查表一次，执行期零分支 */ }
```

- 第三方 op = 注册 `(signature, kernel)`，即插即用——"丰富生态"的机制支点；
- **仅推理：无 autograd/backward/grads**。对比 ggml 的 grads/grad_accs（`ggml-impl.h:335-336`）直接不做，图结构更简单、tensor 无梯度字段、内核表只有 forward 一项。未来的自动微分可以以独立 crate（`feml-ad`）形式外挂，不污染核心。

---

## 6. 架构总览与 crate 结构

```
┌─────────────────────────────────────────────────────────┐
│  feml（门面 crate：prelude + 后端 feature 开关）          │
├────────────┬──────────────┬─────────────┬───────────────┤
│ feml-core  │ feml-cpu     │ feml-cuda   │ feml-opencl   │
│ 图/张量/    │ SIMD 内核     │ CUDA 内核    │ OpenCL 内核    │
│ plan/内存   │ + 线程池      │ (cudarc)    │ (ocl)         │
├────────────┴──────────────┴─────────────┴───────────────┤
│ feml-gguf（GGUF 读写 + mmap）                              │
│ feml-python（pyo3，M5）                                   │
└─────────────────────────────────────────────────────────┘
```

依赖底座：`half`、`bytemuck`、`rayon`（或自研轻量线程池）、`smallvec`、可选 `parking_lot`。

**发布到 crates.io 是生态起点**：`cargo add feml` 即可用——ggml 永远做不到。

---

## 7. 关键 API 草图

```rust
use feml::prelude::*;

// ── 建图期 ──────────────────────────────────────────────
let mut g = GraphBuilder::new();
let x = g.input("x", DType::F32, shape![1, 4096])?;          // Result：shape/dtype 校验
let w = g.weight("w", DType::Q8_0, shape![4096, 11008])?;
let h = g.mul_mat(&w, &x)?;                                   // 类型安全 + 形状检查
let o = g.rms_norm(&h, 1e-5)?;
g.mark_output(&o)?;

// ── 编译期（一次）────────────────────────────────────────
let ctx = Device::Cpu.into_context()?;                       // 或 Registry::open_best()
let plan = g.compile(&ctx)?;                                 // 拓扑排序 + 内核选择 + 内存规划

// ── 执行期（每次推理，零分配）────────────────────────────
ctx.write(&x, input_data)?;                                  // 输入
plan.execute(&ctx)?;                                         // 输出写入规划好的位置
let out = ctx.read(&o)?;
```

---

## 8. 执行引擎设计

- **CPU**：持久线程池（翻译 `ggml-cpu.c:472`：cond-var 队列 + `(ith, n_threads)` 分片 + abort 标志）；`mul_mat` 按输出行分片；后续可加节点级 DAG 并行。
- **GPU**：`graph_compute` 提交到流；跨设备依赖用 event；`graph_plan` 让同拓扑图（LLM 每 token 同一图）复用预编译 CUDA 图。
- **异构**：张量带 `device` 标记；plan 编译期自动插入跨设备 copy 节点（ggml 在运行时判断，我们在编译期完成）。
- **异步**：核心 API 同步返回 `Result`；上层 `feml-async` wrapper 接 tokio——修 ggml 无法异步集成的缺点。

---

## 9. 推理专属设计（超越 ggml 的发力点）

### 9.1 KV Cache 一等公民（ggml 最痛的点，我们做成原生抽象）

llama.cpp 的 KV cache 是靠手写"视图拼接"graph 实现的（每 token 位置用 view 指向 cache 缓冲的偏移），
复杂、易错、每轮重建。feml 把 KV cache 做成**原生类型**：

```rust
let kv = g.kv_cache("kv", DType::F16, n_layers, n_heads, head_dim, cache_len)?;
// 内部 = 一块静态缓冲 + 每层 (K,V) 的 (buffer_id, base_offset)
// prefill：一次性写入整段；decode：按 pos 写入单个位置

let out = g.flash_attn(&q, kv, pos)?;      // 或 g.attn(&q, &kv, pos)
let out = g.mul_mat(&w_out, out)?;
```

- cache 缓冲**编译期一次性分配**，prefill/decode 都只是对它的读写，graph 不再需要重建视图；
- 支持**分页 KV（paged attention，vLLM 风格）**作为后续扩展点：cache 换页与空闲页管理在算子层解决。

### 9.2 prefill / decode 双执行模式

推理两个阶段形态完全不同（prefill 计算密集、大 batch；decode 访存密集、batch=1、形状恒定），
编译时让用户声明模式，产出两种 plan（或同一 plan 带模式标志）：

- **decode 模式**：形状全静态 → kernel 可模板化特化；CUDA 后端可编译成 CUDA Graph（一次图捕获、每次 replay，省 launch 开销）；CPU 后端选单 token 最优分片。
- **prefill 模式**：大 batch 并行分片优先。
- `plan.execute(&ctx, Mode::Decode)` 零分配切换，graph 结构不变。

### 9.3 编译期图融合（peephole 优化器）

ggml 的融合靠人工（llama.cpp 手动 fuse rms_norm 前的 mul_mat、把 bias+act 揉进 kernel）。
feml 的 plan 是编译产物，可以在 `compile()` 里做规则化融合：

- `rms_norm + mul_mat` 融合（省一次中间张量全量读写，decode 场景收益显著）；
- `mul_mat + bias + gelu/silu` 融合；
- 常量折叠、公共子表达式消除（CSE）、死节点消除（liveness 天然支持）。
- 融合规则表可扩展（第三方可注册），与算子注册表同构。

### 9.4 Kernel 自动调优（autotune）

ggml 每个后端手写固定 kernel 与分片策略。feml 在 `compile()` 时按
`(op, dtype, shape, device)` 在候选 kernel 集合里做快速基准（trial），
选最优绑定进 plan（CUDA 侧顺带完成 graph capture）。

- 候选集合：分片粒度 × SIMD 变体 × 特化模板；
- 结果缓存（按签名），同形状图只 tune 一次——推理场景形状稳定，tune 成本摊薄；
- 提供 `AutotuneMode::{Fast, Full, Off}` 供用户权衡。

### 9.5 流水线与异步推理

- **层间流水**：GPU 上算 layer i+1 与拷贝 layer i 输出重叠（多流 + event，ggml 的 event 机制直接复用）；
- **跨 token 预取**：采样阶段后台预加载下一 token 输入（上层 `feml-async` 提供）；
- 执行 API 保持同步 `Result`，异步由 wrapper 层提供（与核心零分配不冲突）。

### 9.6 推理算子路线（LLM 覆盖集）

第一优先（llama 推理必需）：`mul_mat / mul_mat_id(MoE) / rms_norm / rope / softmax /
attention(flash) / add / mul / get_rows / concat / silu / view 族 / copy`。

后续：`flash_attn(FA2/FA3)`、`MLA`（DeepSeek）、`grouped-query attention`、
`FP8 (E4M3/E5M2)`、`BF16`、`quantized mul_mat`（Q4_0/Q8_0/Q4_K/Q6_K，SIMD 内核）、
`speculative decoding` 辅助算子（验证分支）。

---

## 10. 量化与 GGUF

- **类型表正确移植**（当前 feml 的 data_type.rs 错误，重写）：`Q4_0/Q8_0/Q4_K/Q6_K` 等，`size = sizeof(block)`、`blck = QK`；
- **内核生成**：`kernel!` 宏 + 每后端 trait 实现，避免 14 后端手抄（修复 ggml 的组合爆炸）；
- **GGUF**：独立 crate；mmap + `buffer_from_host_ptr` 零拷贝加载 llama 模型——与 llama.cpp 生态无缝对接的接口。

---

## 11. 对现有代码的处置

| 保留 | 废弃 |
|---|---|
| 项目名 / crate 名 | 所有 `Rc<RefCell>` 张量/上下文 |
| `Error`/`Result` 思路（精简 542 行） | `ObjectPool`、`TensorIdArray` 死代码 |
| `Registry` 概念（重写为分数排序） | `as_any` 下转型、三套不兼容 trait |
| ne/nb/view 概念（shape/layout 方向对） | 字符串设备匹配、`todo!()`/`unwrap()`、`static mut` |
| | stride/nbytes 错误算法（先测试后重写） |

后端代码无迁移价值（全是旧 trait 的 mul 桩），概念与命名空间可继承。

---

## 12. 里程碑

| 阶段 | 内容 | 验收标准 |
|---|---|---|
| **M0** 骨架 | workspace + `feml-core` 类型系统（正确 size/blck 表）+ CPU `mul` + 数值测试 + CI | `cargo test` 全绿；`[2,3]` F32 nbytes=24 锁定 |
| **M1** 引擎 | GraphBuilder（迭代式，非递归）+ compile→plan + MemoryPlanner + 线程池 + `mul_mat` | mul_mat 与 numpy 对齐；执行期零分配（bench 验证） |
| **M2** 后端 | trait 冻结为公共 API + OpenCL 移植 + 15 个 LLM 核心算子 | CPU/OpenCL 前向结果一致 |
| **M3** 模型 | 量化 + SIMD 内核 + `feml-gguf` | 端到端推理 llama 3.2 1B，输出与 llama.cpp 对齐 |
| **M4** 推理特性 | KV cache 原生抽象 + prefill/decode 双模式 + 图融合 + kernel autotune | 单 token decode 延迟对标 llama.cpp（同机型），KV cache 用法无需手写视图 |
| **M5** 生态 | 插件加载（cdylib + 版本协商）+ pyo3 + 发布 crates.io + docs + 基准页 + 推理算子扩展（flash_attn/MLA/MoE/FP8） | `cargo add feml` 可跑示例；算子覆盖主流开源模型（llama/qwen/deepseek 架构） |

---

## 13. 风险与开放问题

1. **两阶段 API 学习曲线**：与"零分配执行"强绑定，值得；用 builder 糖衣缓解。
2. **生态竞争**（candle/burn）：差异化锚定在 GGUF/量化兼容 + 零分配执行 + 多后端（candle 无深度 OpenCL/Metal 支持）。
3. **unsafe 边界**：仅允许三处——内核读写指针、后端 ffi、mmap。公共 API 100% safe。
4. **CUDA 绑定选择**：立项前 1 天实测 cudarc 当前状态（维护活跃度、文档、burn 集成案例），避免重蹈 cuda-oxide 实验性依赖的覆辙。
5. **线程模型取舍**：先按 ggml 的算子内分片（简单、可达标），节点级 DAG 并行作为后续优化，不阻塞主线。

---

## 附录 A：审查发现的完整问题清单

（§2 表格的展开版本，含 file:line 定位，供后续开发对照修复）

- 编译错误根因：`src/backend.rs` trait 重构（3a35565）未传播到 `src/cpu/*`（仍实现旧 trait 方法：`memcpy_async`/`set_tensor_async`/`props`/`init` 等）；`src/cpu/backend_buffers.rs` 使用裸 `Result<()>`（缺 `crate::error::Result` 导入）、未导入的 `TensorStorage`、不存在的 `tensor.borrow().length()`。
- 内存布局 bug 细节：`src/context.rs:174-175` `stride[1] = stride[0]*(stride[0]/block_size)`（正确：`stride[0]*shape[0]`）；`src/data_type.rs:31-40` U8/U32/I16/I32/I64 误标 `quantized: true`，所有类型 `block_size == type_size`（正确：普通类型 blck=1）；`src/layout.rs:36` nbytes 对 F32 返回元素个数而非字节数。
- 线程安全：`src/tensor.rs:84`、`src/context.rs:77`、`src/compute_graph.rs:44` 全部 `Rc<RefCell>`；context.rs:72-75 注释声称 Arc 线程安全，实现是 Rc。
- 恐慌路径：`src/opencl/backend_device.rs:34`（info `todo!()`）、`src/opencl/backend_buffer.rs:127`（reset `todo!()`）、`src/opencl/backend.rs:96`（downcast unwrap）、`src/context.rs:243`（new unwrap）、`src/cuda/kernels/mul.rs:46,55`（expect/unwrap）、`src/cpu/backend_buffers.rs:62`。
- 设备字符串匹配：`src/opencl/backend_device.rs:83-92`（非 "Intel"/"Qualcomm" 拒绝；Qualcomm 误设 Intel 族）。
- 注册器 unsound 模式：`src/opencl/backend_register.rs:11` `static mut` + 裸指针 + `Once`；与 `src/cpu/backend_register.rs:6` 的 `OnceLock` 不一致。
- 图构建：`src/compute_graph.rs:109-138` 递归 DFS + 每步 borrow_mut；`build_forward(expand=true)` 时 `node_use_count` 不重置会翻倍。
- API 陷阱：`src/tensor.rs:264-270` `Deref<Target=RefCell>`；`:383-393` clone 共享语义。
- 死代码/拼写：`TensorIdArray`（tensor.rs:32-66）、`backend_cpu_device_context.rs`（3 行空壳）、`graph_pool_cacacity`、`aysnc`、`OpenclGpuFamlily`。
- 功能差距：全库仅 `TensorOpMul` 1 个算子（ggml ~100 个）；无量化类型；CPU `graph_compute` 返回 Unsupported（`src/cpu/backend.rs:25-27`）；OpenCL 仅支持 mul（`src/opencl/backend_device.rs:41-46`）。
