# ADR-0002：T05 上游评测环境与协议引用扩展

- 状态：Accepted（2026-09-13）
- 影响包：T05（父包）；T06/T07/T14/T20 复用同一评测镜像
- 相关设计：设计文档 §8.1、§9.1；T04/ADR-0001 的容器边界

## 背景

T05 要求复用 pinned KernelBench 上游评测（`src/kernelbench/eval.py` 的 `eval_kernel_against_ref`），在目标 GPU 上对开发子集问题给出可信的正/反例判定。三个环境事实：

1. 上游 evaluator 需要 torch/triton/nvcc/pydantic/numpy 等依赖；控制端（Domain）不得安装这些（T00 边界），因此评测必须在 ADR-0001 容器边界内、使用锁定的 GPU 评测镜像执行。
2. 上游 `eval_kernel_against_ref` 在调用时（非导入时）`from .utils import get_gpu_vendor`，而冻结清单只恢复了 `dataset.py/eval.py/timing.py`；`utils.py` 缺失使 pinned evaluator 无法运行。`utils.py` 顶层又导入 `dotenv/tqdm/openai`（仅导入，评测路径不调用、无凭据、容器禁网使其惰性）。
3. Docker Hub 直连不可用（daemon 级），经 daocloud 镜像源拉取；pip 经清华镜像源安装。

## 决定

1. **评测镜像**：本地构建、按 image ID 锁定；构建源为提交到仓库的 `configs/eval-image/Dockerfile`：
   - 基础镜像（按 digest 锁定）：`docker.m.daocloud.io/pytorch/pytorch@sha256:14611869895df612b7b07227d5925f30ec3cd6673bad58ce3d84ed107950e014`（tag `2.5.1-cuda12.4-cudnn9-devel`，torch 2.5.1+cu124、triton 3.1.0、CUDA 12.4 toolkit 含 nvcc；宿主驱动 580.95.05 兼容 cu124）。
   - 追加固定版本 pip 依赖：`pydantic==2.9.2 requests==2.32.3 python-dotenv==1.0.1 tqdm==4.66.5 openai==1.54.4`（构建期经 `https://pypi.tuna.tsinghua.edu.cn/simple` 获取）。
   - 当前构建产物 image ID：`sha256:bb4ddb1e04d2662ef5c684d8ce12bda1ad8b87c87d9979dedb0642c749c16874`。重建一致性由"提交的 Dockerfile + 锁定的基础镜像 digest + 固定版本"保证；镜像内容变化必须换新 image ID 并重跑 T05 GPU 验收。
2. **协议引用扩展（不改题集/容差/计时）**：`configs/kernelbench/dev-manifest.json` 的 `protocols.upstream_compatible.references` 追加 `src/kernelbench/utils.py`（sha256 `c421bf34…`，取自同一 pinned commit `423217d9…`，raw.githubusercontent 经代理获取后核验）。dev manifest 的 `problems`、容差与 `files_manifest_sha256` 绑定一律不动；`bench verify` 扩展后仍 PASS。此扩展属"恢复 pinned evaluator 的可运行性"，不是评测规则变更。
3. **执行方式**：评测以 ADR-0001 容器执行器运行：快照根只读挂载于 `/task`，候选与问题源码由父端 hash 后下发；evaluator 驱动脚本在容器内调用 `eval_kernel_against_ref`（`measure_performance=False`，T05 只做正确性判定；正式计时属 T07），原始 `KernelExecResult` 序列化写入输出目录，由可信父端独立核对"适配器判定 = 上游结果"的一致性。
4. **GPU 分配**：沿用 ADR-0001——CDI 按 UUID 独占、串行。
5. **对照原则**：适配器不得改写上游结果字段；"通过/失败"必须与上游 `KernelExecResult.passed` 一致；扩展检查（若有）单独报告，不混入 upstream_compatible 赛道。

## 后果与已知限制

- 13.3GB 镜像依赖 daocloud 拉取通道；离线机器需预先导入。
- `openai` SDK 进入镜像但评测路径不实例化客户端；容器禁网 + 无凭据（ADR-0001）保证其惰性。
- `utils.py` 的 `load_dotenv` 若在模块级执行，读取的是容器内 `/task`（只读、无 .env）；无宿主泄漏通道。
