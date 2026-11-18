# 固定 x86_64 Isaac Sim 4.5.0；项目源码在运行时从仓库只读挂载。
ARG ISAAC_SIM_IMAGE=nvcr.io/nvidia/isaac-sim@sha256:c2f47dc82a7714af08d3766efe80ac9d084c2b37b5d0dfbd074797ec56390fc7
FROM ${ISAAC_SIM_IMAGE}
LABEL org.opencontainers.image.title="REASAN-A2 simulation runtime"
COPY requirements-sim45.lock /tmp/reasan-requirements.lock
# --no-deps 保持参考镜像已经使用的组合，不让解析器升级 Torch、CUDA 或 NumPy。
RUN /isaac-sim/python.sh -m pip install --no-cache-dir --no-build-isolation --no-deps -r /tmp/reasan-requirements.lock \
    && rm /tmp/reasan-requirements.lock
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
ENV OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /workspace/upstream/REASAN/training
ENTRYPOINT ["/isaac-sim/python.sh"]
