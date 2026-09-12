# T18 diagnostic image (ADR-0002 addendum): the pinned evaluation image
# plus the tritonbench runner's runtime dependency (tabulate, psutil).
FROM kernelagent-eval:cuda124-pytorch251-t05

RUN pip install --no-cache-dir \
        -i https://pypi.tuna.tsinghua.edu.cn/simple \
        tabulate==0.9.0 \
        psutil==6.0.0 \
        pynvml==12.0.0
