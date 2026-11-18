"""原样生成公开测试地形并冻结网格；记录原生成器使用的时间种子。"""

import argparse
import hashlib
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("目标目录已存在，拒绝覆盖已冻结地形")
    args.out.mkdir(parents=True)
    app = None
    started = time.monotonic()
    try:
        from isaaclab.app import AppLauncher
        app = AppLauncher(headless=True, device="cuda:0", kit_args=(
            "--/plugins/carb.tasking.plugin/threadCount=4 "
            "--/persistent/isaac/asset_root/cloud=/asset-cache")).app
        import numpy as np
        import torch
        from go2_lidar.terrain.train_terrain_cfg import GO2_TEST_TERRAIN_CFG
        from go2_lidar.terrain.test_terrain_generator import TestTerrainGenerator

        np.random.seed(0)
        torch.manual_seed(0)
        cfg = GO2_TEST_TERRAIN_CFG.copy()
        seeds = []
        original_seed = np.random.seed

        def record_seed(value):
            seeds.append(int(value))
            return original_seed(value)

        # 只记录原函数实际采用的种子，不替换时间值或改变生成逻辑。
        np.random.seed = record_seed
        try:
            terrain = TestTerrainGenerator(cfg, device="cpu")
        finally:
            np.random.seed = original_seed

        arrays = {"vertices": np.asarray(terrain.terrain_mesh.vertices),
                  "faces": np.asarray(terrain.terrain_mesh.faces),
                  "terrain_origins": terrain.terrain_origins}
        arrays.update({"patch_" + name: value.cpu().numpy()
                       for name, value in terrain.flat_patches.items() if isinstance(value, torch.Tensor)})
        for name, array in arrays.items():
            if not np.isfinite(array).all():
                raise RuntimeError(f"Nonfinite terrain array: {name}")
        terrain_file = args.out / "terrain.npz"
        np.savez_compressed(terrain_file, **arrays)

        families = {"sparse": (3, "scatter_static_sparse", 15.0),
                    "dense": (4, "scatter_static_dense", 15.0),
                    "maze": (5, "maze", 30.0)}
        cases = []
        for family, (column, patch, deadline) in families.items():
            for variant in range(10):
                center = [(variant + 0.5) * 20.0 - 100.0, (column + 0.5) * 10.0 - 40.0]
                cases.append({"case_id": f"{family}_{variant:02d}", "family": family,
                              "variant": variant, "column": column,
                              "split": "validation" if variant < 2 else "test",
                              "start": arrays[f"patch_{patch}_start"][variant].tolist(),
                              "goal": arrays[f"patch_{patch}_goal"][variant].tolist(),
                              "center": center, "bounds": [center[0]-10, center[0]+10, center[1]-5, center[1]+5],
                              "deadline_s": deadline, "dynamic_obstacles": 0})
        metadata = {"source_commit": "655bf4f1cc23e835d87bf9a887972af6760af162",
                    "source_class": "go2_lidar.terrain.test_terrain_generator.TestTerrainGenerator",
                    "initial_numpy_seed": 0, "recorded_numpy_reseeds": seeds,
                    "terrain_sha256": hashlib.sha256(terrain_file.read_bytes()).hexdigest(),
                    "vertices": len(arrays["vertices"]), "triangles": len(arrays["faces"]),
                    "grid": [10, 8], "cell_size_m": [20, 10], "cases": cases,
                    "numpy_version": np.__version__, "wall_seconds": time.monotonic()-started}
        (args.out / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(json.dumps({k:v for k,v in metadata.items() if k != "cases"}), flush=True)
    finally:
        if app is not None:
            app.close(wait_for_replicator=False)


if __name__ == "__main__":
    main()
