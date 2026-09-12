# T05+ evaluation environment (ADR-0002).
#
# Base image pinned by manifest digest; pip dependencies pinned by exact
# version (tsinghua PyPI mirror is a build-time fetch channel, recorded in
# the ADR). Built locally and pinned by the resulting image ID; a rebuild
# from this committed Dockerfile reproduces the same environment. The host
# control plane never installs any of these dependencies.
FROM docker.m.daocloud.io/pytorch/pytorch@sha256:14611869895df612b7b07227d5925f30ec3cd6673bad58ce3d84ed107950e014

RUN pip install --no-cache-dir \
        -i https://pypi.tuna.tsinghua.edu.cn/simple \
        pydantic==2.9.2 \
        requests==2.32.3 \
        python-dotenv==1.0.1 \
        tqdm==4.66.5 \
        openai==1.54.4
