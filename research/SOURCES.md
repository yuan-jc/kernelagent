# 调研来源索引

调研日期：2026-09-12。以下是设计调研快照，不是已经验证可兼容的运行时依赖锁。

README/源码属于上游资料，未作为本项目代码执行。各项目保留原有许可证；未完整审计每个仓库。

设计文档区分已核对源码、README 描述和未实测能力。本索引中的文件被保存不等于每行都经人工审计。

| 项目 | 固定 commit | 本地材料 |
|---|---|---|
| [BytedTsinghua-SIA/CUDA-Agent](https://github.com/BytedTsinghua-SIA/CUDA-Agent/tree/473025c8af7878e928525138b5cb327d6bcdf9dd) | `473025c8af7878e928525138b5cb327d6bcdf9dd` | 本地缓存 `sources/BytedTsinghua-SIA__CUDA-Agent`（不随公开仓库分发） |
| [flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer/tree/eea399b74fe3cb74f942f21eecfed6c0d46561c4) | `eea399b74fe3cb74f942f21eecfed6c0d46561c4` | 本地缓存 `sources/flashinfer-ai__flashinfer`（不随公开仓库分发） |
| [flashinfer-ai/flashinfer-bench](https://github.com/flashinfer-ai/flashinfer-bench/tree/40e6ca7844b514eb4b1c7edba6d6a7377df57870) | `40e6ca7844b514eb4b1c7edba6d6a7377df57870` | 本地缓存 `sources/flashinfer-ai__flashinfer-bench`（不随公开仓库分发） |
| [gpu-mode/reference-kernels](https://github.com/gpu-mode/reference-kernels/tree/51e22db671d36c1c76091c43c36a44546ba324a1) | `51e22db671d36c1c76091c43c36a44546ba324a1` | 本地缓存 `sources/gpu-mode__reference-kernels`（不随公开仓库分发）；仅 pmpp_v2+linalg 子集（12 题），逐文件哈希见 `configs/reference-kernels/`。License: June 9 Researcher Reciprocity License v1.0（Open RAIL-S 衍生）——本项目仅本地评测用，不训练、不做检索/微调材料、不再分发缓存字节；所有本地数字为"本地复刻协议，不与官方榜单互比"（获取/校验：`research/fetch_reference_kernels.py`） |
| [KernelTuner/kernel_tuner](https://github.com/KernelTuner/kernel_tuner/tree/5d0d9e98066c06a7da0097cac8806e6f95952ab9) | `5d0d9e98066c06a7da0097cac8806e6f95952ab9` | 本地缓存 `sources/KernelTuner__kernel_tuner`（不随公开仓库分发） |
| [meta-pytorch/KernelAgent](https://github.com/meta-pytorch/KernelAgent/tree/e0647170da36ef9b059ac0bd3d60103aa4ed378b) | `e0647170da36ef9b059ac0bd3d60103aa4ed378b` | 本地缓存 `sources/meta-pytorch__KernelAgent`（不随公开仓库分发） |
| [meta-pytorch/tritonbench](https://github.com/meta-pytorch/tritonbench/tree/cdadd2ea64f487c75c504e1de9d03fb88a06bd2d) | `cdadd2ea64f487c75c504e1de9d03fb88a06bd2d` | 本地缓存 `sources/meta-pytorch__tritonbench`（不随公开仓库分发） |
| [mit-han-lab/ncu-report-skill](https://github.com/mit-han-lab/ncu-report-skill/tree/74a12918e9f64d78036f14da5f8765e435b949a4) | `74a12918e9f64d78036f14da5f8765e435b949a4` | 本地缓存 `sources/mit-han-lab__ncu-report-skill`（不随公开仓库分发） |
| [NVIDIA/cutlass](https://github.com/NVIDIA/cutlass/tree/147295a3d4b75f3aeff247c25b8927cea9a7006a) | `147295a3d4b75f3aeff247c25b8927cea9a7006a` | 本地缓存 `sources/NVIDIA__cutlass`（不随公开仓库分发） |
| [NVIDIA/MatX](https://github.com/NVIDIA/MatX/tree/afb0b2e8c0a07532b9984508c1fdbc9738759736) | `afb0b2e8c0a07532b9984508c1fdbc9738759736` | 本地缓存 `sources/NVIDIA__MatX`（不随公开仓库分发） |
| [NVIDIA/nvbench](https://github.com/NVIDIA/nvbench/tree/410dcdd21c9b48191ecb3d3d77060b1bf4ac6244) | `410dcdd21c9b48191ecb3d3d77060b1bf4ac6244` | 本地缓存 `sources/NVIDIA__nvbench`（不随公开仓库分发） |
| [NVlabs/kda](https://github.com/NVlabs/kda/tree/4806866492d5c7cad05e07918790bd1e94e976a1) | `4806866492d5c7cad05e07918790bd1e94e976a1` | 本地缓存 `sources/NVlabs__kda`（不随公开仓库分发） |
| [ScalingIntelligence/caesar](https://github.com/ScalingIntelligence/caesar/tree/292f5d39452217ca8ce3b96334fe6840b44556da) | `292f5d39452217ca8ce3b96334fe6840b44556da` | 本地缓存 `sources/ScalingIntelligence__caesar`（不随公开仓库分发） |
| [ScalingIntelligence/KernelBench](https://github.com/ScalingIntelligence/KernelBench/tree/423217d9fda91e0c2d67e4a43bf62f96f6d104f1) | `423217d9fda91e0c2d67e4a43bf62f96f6d104f1` | 本地缓存 `sources/ScalingIntelligence__KernelBench`（不随公开仓库分发） |
| [thunlp/TritonBench](https://github.com/thunlp/TritonBench/tree/603e28a5050e8c268f6883a69709d477a272d49a) | `603e28a5050e8c268f6883a69709d477a272d49a` | 本地缓存 `sources/thunlp__TritonBench`（不随公开仓库分发） |
| [tile-ai/tilelang](https://github.com/tile-ai/tilelang/tree/66c003c3e75c336f22d928fcd5d3b284cf5a18a0) | `66c003c3e75c336f22d928fcd5d3b284cf5a18a0` | 本地缓存 `sources/tile-ai__tilelang`（不随公开仓库分发） |
| [triton-lang/triton](https://github.com/triton-lang/triton/tree/979c55829b152507bf2b44815bc7eea4ba7e8803) | `979c55829b152507bf2b44815bc7eea4ba7e8803` | 本地缓存 `sources/triton-lang__triton`（不随公开仓库分发） |
| [wzzll123/MultiKernelBench](https://github.com/wzzll123/MultiKernelBench/tree/460a972c9be7ce18035984321d38b910e27df95f) | `460a972c9be7ce18035984321d38b910e27df95f` | 本地缓存 `sources/wzzll123__MultiKernelBench`（不随公开仓库分发）；仅 NVIDIA 平台 `reference/` 300 题 + 上游协议 5 文件（MIT），逐文件哈希见 `configs/multikernelbench/`（获取/校验：`research/fetch_multikernelbench_problems.py`） |

## NVIDIA 官方文档

| 文档 | 本地快照 |
|---|---|
| [ncu_profiling_guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html) | 本地缓存 `sources/nvidia_docs/ncu_profiling_guide.html`（不随公开仓库分发） |
| [ncu_cli](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html) | 本地缓存 `sources/nvidia_docs/ncu_cli.html`（不随公开仓库分发） |
| [compute_sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html) | 本地缓存 `sources/nvidia_docs/compute_sanitizer.html`（不随公开仓库分发） |
| [nsight_systems](https://docs.nvidia.com/nsight-systems/UserGuide/index.html) | 本地缓存 `sources/nvidia_docs/nsight_systems.html`（不随公开仓库分发） |
| [cuda_binary_utilities](https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html) | 本地缓存 `sources/nvidia_docs/cuda_binary_utilities.html`（不随公开仓库分发） |

## 复核与获取

- `snapshot_manifest.json` 保留调研固定 commit、文件 URL 与 SHA-256；不是已经验证兼容的运行依赖锁。
- `fetch_kernelbench_problems.py` 仅恢复 `configs/kernelbench/` 已冻结的问题与协议引用，不改写清单、不执行上游代码；首次克隆可直接运行。
- `fetch_multikernelbench_problems.py` / `fetch_reference_kernels.py` 同理，分别恢复 `configs/multikernelbench/` 与 `configs/reference-kernels/` 已冻结的快照；两者都提供 `--generate-manifest`（钉 commit 生成清单，唯一允许产生清单的入口），默认模式只做带哈希校验的恢复。
- 其他调研原始文件可按 snapshot_manifest 中 URL 获取并校验，日常开发无需下载全部研究项目。
- 第三方资料及许可证属于原项目；本仓库不分发本地原始缓存。
- 分批抓取流水与一次性脚本已从当前目录移除，历史可在 Git 提交 `1b4a306` 查阅。
