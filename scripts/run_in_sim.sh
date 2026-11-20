#!/usr/bin/env bash
# 在独立容器中运行脚本，优先读取本项目固定的 REASAN / Isaac Lab 源码。
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runtime_image="${REASAN_RUNTIME_IMAGE:-}"
if [[ -z "$runtime_image" ]]; then
  if docker image inspect reasan-a2:sim45-v1 >/dev/null 2>&1; then
    runtime_image=reasan-a2:sim45-v1
  else
    runtime_image=sha256:d86b2e1ce0db49f2ff08e42f07bb1ae8cbba22e4c727d17d7f4c68d280ab3b4e
  fi
fi
if ! docker image inspect "$runtime_image" >/dev/null 2>&1; then
  echo '请先运行：docker build -t reasan-a2:sim45-v1 runtime' >&2
  exit 2
fi
mkdir -p "$project_root/.cache/kit" "$project_root/.cache/ov" \
  "$project_root/.cache/warp" "$project_root/.cache/nvidia" "$project_root/.cache/compute"
lab_source=/workspace/upstream/REASAN/training/IsaacLab/source
source_paths="$lab_source/isaaclab:$lab_source/isaaclab_assets:$lab_source/isaaclab_tasks:$lab_source/isaaclab_rl:$lab_source/isaaclab_mimic"
source_paths="$source_paths:/workspace/upstream/REASAN/training/go2_lidar:/workspace/upstream/REASAN/training/rsl_rl:/workspace/upstream/REASAN/training/ray_predictor:/workspace"
filter_mount=()
if [[ -n "${REASAN_A2_FILTER_POLICY:-}" ]]; then
  if [[ ! -f "$REASAN_A2_FILTER_POLICY" ]]; then
    echo '找不到 REASAN_A2_FILTER_POLICY 指定的安全层权重' >&2
    exit 2
  fi
  filter_path="$(realpath -- "$REASAN_A2_FILTER_POLICY")"
  echo "本次容器使用安全层权重 $filter_path"
  filter_mount=(-v "$filter_path:/workspace/upstream/REASAN/training/logs/rsl_rl/go2_lidar/filter_1/exported/policy.pt:ro")
fi
exec docker run --rm --gpus all --network none --shm-size=1g --cpus=4 \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y -e OMNI_KIT_ALLOW_ROOT=1 \
  -e OMP_NUM_THREADS=4 -e MKL_NUM_THREADS=4 -e PYTHONUNBUFFERED=1 \
  -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH="$source_paths" \
  -v "$project_root:/workspace" \
  -v "$project_root/upstream/REASAN:/workspace/upstream/REASAN:ro" \
  "${filter_mount[@]}" \
  -v "$project_root/assets/isaac45:/asset-cache:ro" \
  -v "$project_root/.cache/kit:/isaac-sim/kit/cache" \
  -v "$project_root/.cache/ov:/root/.cache/ov" \
  -v "$project_root/.cache/warp:/root/.cache/warp" \
  -v "$project_root/.cache/nvidia:/root/.cache/nvidia" \
  -v "$project_root/.cache/compute:/root/.nv/ComputeCache" \
  -w /workspace/upstream/REASAN/training \
  --entrypoint /isaac-sim/python.sh "$runtime_image" "$@"
