# ADR-0001：T04 可信 worker 的容器边界

- 状态：Accepted（2026-09-13）
- 影响包：T04（父包）；T05/T15 的镜像选择将另行记录
- 相关设计：设计文档 §7、§9.2、§12

## 背景

T04 要求候选代码在 OS 强制边界内执行：禁网、只读可信输入、独立输出、资源限额、任务结束无遗留后代。T04a 进程执行器（killpg）不能阻止 setsid 逃逸，也不提供网络/文件/资源边界。目标机已确认：Docker Engine 28.2.2（cgroup v2、systemd 驱动）、NVIDIA Container Toolkit 1.19.1（`nvidia` runtime 已注册、CDI 设备 `nvidia.com/gpu=0` 与 `nvidia.com/gpu=GPU-ea248ec5-…` 已发现）、驱动 580.95.05（CC 8.9）。宿主无 nvcc/ncu（另一会话安装中）；Docker daemon 直连 `registry-1.docker.io` 超时，且无密码 sudo 可为 daemon 配代理；`docker.m.daocloud.io` 直连可用。已在本机验证：`python:3.11-slim-bookworm` + `--device nvidia.com/gpu=<UUID>` 容器内 `ctypes.CDLL('libcuda.so.1')` + `cuInit(0)` 返回 `0x0`、可见 1 个设备。

## 决定

1. **运行时**：Docker Engine + NVIDIA Container Toolkit；执行器以 docker CLI 子进程驱动，容器名用独立随机标识（不使用 request_id）。回收 = `docker kill` → `docker rm -f` → `docker inspect` 确认不存在；容器打 `kernelagent.worker=1` 标签便于泄漏检测。容器内 PID namespace + cgroup 使 setsid 后代无法逃出 kill 范围（关闭 T04a 记录的缺口）。
2. **镜像锁定（按 digest）**，经 daocloud 镜像源拉取：
   - CPU 隔离：`docker.m.daocloud.io/library/ubuntu:24.04` @ `sha256:224a1869083a311ef3f13648a154ba79832fbef6364d31493642ca03082da254`
   - GPU smoke：`docker.m.daocloud.io/library/python:3.11-slim-bookworm` @ `sha256:528257d48c1da0dcecc2e725d1ae34498d60c965f1241e39cd6a85a8859bdf84`（仅 CUDA Driver API；libcuda 由 toolkit 从宿主驱动注入，容器 CUDA 能力 ≤ 宿主驱动 580.95.05）
3. **用户与权限**：容器以非 root `--user 20000:20000` 运行，`--cap-drop ALL`、`--security-opt no-new-privileges`、rootfs `--read-only`。可信输入（reference/候选源码，装载前由父端 hash）只读 bind mount；候选输出仅能写入 size 受限的 tmpfs（默认 64 MiB，挂载于 `/out`），运行结束后由父端 `docker cp` 提取，再删除容器。容器内用户无法写宿主父端目录。
4. **网络**：`--network none`；负例以容器内真实 connect/DNS 失败为证据。
5. **资源限额**：`--memory`（默认 512 MiB）、`--cpus`（默认 1.0）、`--pids-limit`（默认 64）由 cgroup 强制；stdout/stderr 由父端监视循环限额（默认 32 MiB，超限 kill 并记 `failed`）；输出磁盘由 tmpfs size 强制。超限结果为结构化 `failed`/`timeout`/`infra_error`，不得转为正确分数。
6. **GPU 分配**：单卡（RTX 4060 Laptop，无 MIG）；按 CDI UUID 独占分配（`--device nvidia.com/gpu=<UUID>`），请求由父端串行化；共享/并发策略不做声明。GPU 身份、镜像、驱动一并进入证据。
7. **时间限额**：复用 `WorkerRequest.timeout_seconds`；超时即 `docker kill`，状态 `timeout`；取消（cancel）同路径。
8. **协议映射**：容器执行器实现与 T04a 相同的 `WorkerRequest → WorkerOutcome` 契约（worker-v1），另加 `ContainerSpec`（镜像 ref+digest、uid、限额、GPU 设备）。候选退出码/stdout/自写 JSON 不构成正确性或晋升；可信父端只从 `/out` 提取物 + 自有规则判定结果。环境变量仅显式传入并复用 T04a 的秘密名过滤，容器 env 中出现控制端凭据为负例。
9. **执行环境**：docker 不可用时相关测试以明确原因 skip（skip 不计为通过），GPU 用例在无 GPU 环境同样 skip；T04 的 GPU 验收只能由目标机真实运行证据支持。

## 后果与已知限制

- 依赖宿主 Docker 服务与 daocloud 镜像源可达性；镜像升级必须换新 digest 并重跑 T04 回归。
- stdout/stderr 与输出目录的父端监视为轮询（0.5 s 间隔），不是内核级上限；两次轮询之间的超额由内核 memory/pids 限额兜底。
- 宿主侧 docker daemon 本身是可信基座；其自身漏洞不在本 ADR 威胁模型内。
- 备选方案否决记录：podman/systemd-nspawn 未安装且无 CDI GPU 接线；gVisor/Kata 超出需求。

## 修订（2026-09-13，实现期实证）

决定 3/5 中"输出=tmpfs size 强制"改为"输出=宿主 bind 目录 + 父端监视限额"。原因（本机实测，保留在测试历史中）：容器退出即销毁自身 mount namespace，tmpfs 内容随之消失——对已停止容器执行 `docker cp container:/out/.` 返回 0 但得到空目录，单文件 cp 直接报 "Could not find the file"，结果不可恢复且具欺骗性（静默空输出）。因此 `/out` 改为父端私有目录的读写 bind mount（目录 0777、容器 uid 20000 写入、父端持有目录所有权），大小限额由父端 0.5 s 监视循环强制，超限 kill 并记 `failed`；内核 memory/pids 限额在轮询间隙兜底。`/tmp` 仍为 size 受限 tmpfs（容器内、随容器销毁）。此修订不放松验收：输出超限仍是结构化失败，不转成正确分数。

## 修订二（2026-09-13，T05 实现期实证）

/tmp tmpfs 挂载显式加 `exec`。原因：triton/inductor JIT 需要从缓存目录 dlopen 编译出的 .so，Docker tmpfs 默认 noexec 使 torch.compile 候选在正确性评测中报 "failed to map segment from shared object"。对本威胁模型（候选本就是任意可执行代码，容器边界由 namespace/cgroup/能力剥离提供）noexec 不构成额外防线；移除后不放松任何验收约束。