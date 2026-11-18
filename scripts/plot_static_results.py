"""导出固定布局和实测路线的俯视图，供检查起终点与绕行过程。"""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, default=Path("assets/evaluation/public_static_v1"))
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--per-family", type=int, default=2, help="每类最多展示的回合数")
    parser.add_argument("--case-ids", nargs="+", help="只展示指定布局编号，例如 maze_07 maze_08")
    parser.add_argument("--output", default="routes.png", help="结果目录中的图片文件名")
    args = parser.parse_args()
    if args.per_family < 1:
        parser.error("per-family 必须大于零")
    trials = json.loads((args.result/"trials.json").read_text())
    episodes = {row["trial_id"]: row for row in map(json.loads, (args.result/"episodes.jsonl").read_text().splitlines())}
    paths = defaultdict(list)
    for row in map(json.loads, (args.result/"trajectory.jsonl").read_text().splitlines()):
        paths[row["trial_id"]].append(row["position"][:2])
    with np.load(args.assets/"terrain.npz") as data:
        vertices, faces = data["vertices"], data["faces"]
    selected = []
    # 默认每类展示两个回合；分析正式成绩时可指定数量或失败布局。
    for family in ("sparse", "dense", "maze"):
        selected.extend([trial for trial in trials if trial["family"] == family
                         and (not args.case_ids or trial["case_id"] in args.case_ids)][:args.per_family])
    if not selected:
        parser.error("没有匹配的路线")
    families = [family for family in ("sparse", "dense", "maze")
                if any(trial["family"] == family for trial in selected)]
    rows = max(sum(trial["family"] == family for trial in selected) for family in families)
    fig, axes = plt.subplots(rows, len(families), figsize=(5*len(families), 3*rows), squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)
    for trial in selected:
        family_index = families.index(trial["family"])
        row_index = [t for t in selected if t["family"] == trial["family"]].index(trial)
        ax = axes[row_index, family_index]
        ax.set_visible(True)
        xmin, xmax, ymin, ymax = trial["bounds"]
        center = np.asarray(trial["center"])
        inside = ((vertices[:, 0] >= xmin) & (vertices[:, 0] <= xmax)
                  & (vertices[:, 1] >= ymin) & (vertices[:, 1] <= ymax) & (vertices[:, 2] > 0.10))
        local_faces = faces[np.any(inside[faces], axis=1)]
        triangles = vertices[local_faces]
        triangles = triangles[triangles[:, :, 2].mean(axis=1) > 0.10]
        ax.add_collection(PolyCollection(triangles[:, :, :2]-center, facecolors="#a7abb1", edgecolors="none"))
        start, goal = np.asarray(trial["start"][:2])-center, np.asarray(trial["goal"][:2])-center
        path = np.asarray([trial["start"][:2]]+paths[trial["trial_id"]])-center
        ax.plot(path[:, 0], path[:, 1], color="#1769aa", linewidth=1.5, label="trajectory")
        ax.scatter(*start, color="#15803d", s=28, zorder=3, label="start")
        ax.scatter(*goal, color="#be123c", marker="*", s=65, zorder=3, label="goal")
        outcome = episodes.get(trial["trial_id"], {}).get("outcome", "incomplete")
        ax.set_title(f"{trial['trial_id']} : {outcome}", fontsize=10)
        ax.set_xlim(xmin-center[0], xmax-center[0])
        ax.set_ylim(ymin-center[1], ymax-center[1])
        ax.set_aspect("equal")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
    axes[0, 0].legend(loc="lower left", fontsize=7)
    fig.tight_layout()
    out = args.result/args.output
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
