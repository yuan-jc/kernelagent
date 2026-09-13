# T20 diagnostic image layer (CUDA tuning + CUTLASS template path).
# Base: the pinned ADR-0002 evaluation image. Kernel Tuner is GPLv3: it
# stays inside the container, is never redistributed, and its license is
# recorded separately in docs/work-packages/T20.md.
FROM kernelagent-eval:cuda124-pytorch251-t05

RUN pip install --no-cache-dir \
        -i https://pypi.tuna.tsinghua.edu.cn/simple \
        kernel-tuner==1.0.0b6
