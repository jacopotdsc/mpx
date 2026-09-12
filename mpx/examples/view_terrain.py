#!/usr/bin/env python3
"""Visualize a MuJoCo rough-terrain scene.

Reads the box geoms straight from the scene XML (no MuJoCo / g1.xml needed for
the 2D and 3D modes) and renders them so you can check the layout.

Modes:
  map     top-down view, each box colored by its top-surface height (default)
  3d      3D scatter of the box top surfaces
  mujoco  open the interactive MuJoCo viewer (requires mujoco + g1.xml locally)

Usage:
  python3 view_terrain.py                         # map of scene_rough_circular.xml
  python3 view_terrain.py --mode 3d
  python3 view_terrain.py path/to/scene.xml --mode map --out map.png
  python3 view_terrain.py --mode mujoco
"""

import argparse
import math
import os
import re
import sys
import xml.etree.ElementTree as ET


def parse_boxes(path):
    """Return lists xs, ys, tops for every box geom (skips the floor plane)."""
    tree = ET.parse(path)
    root = tree.getroot()
    xs, ys, tops, half = [], [], [], []
    for geom in root.iter("geom"):
        if geom.get("type") != "box":
            continue
        pos = [float(v) for v in geom.get("pos", "0 0 0").split()]
        size = [float(v) for v in geom.get("size", "0 0 0").split()]
        if len(pos) < 3 or len(size) < 3:
            continue
        xs.append(pos[0])
        ys.append(pos[1])
        tops.append(pos[2] + size[2])   # top surface height
        half.append(0.5 * (size[0] + size[1]))
    return xs, ys, tops, half


def view_map(xs, ys, tops, half, out):
    import matplotlib.pyplot as plt

    if not xs:
        sys.exit("No box geoms found in the scene.")

    rs = [math.hypot(x, y) for x, y in zip(xs, ys)]
    inner = min(rs)
    outer = max(rs)
    lim = outer * 1.05

    fig, ax = plt.subplots(figsize=(8, 8))
    sc = ax.scatter(xs, ys, c=[t * 100 for t in tops], s=6, cmap="terrain",
                    marker="s", linewidths=0)
    cb = fig.colorbar(sc, ax=ax, shrink=0.8)
    cb.set_label("top surface height (cm above floor)")

    for r, style in ((inner, ":"), (outer, "--")):
        ax.add_patch(plt.Circle((0, 0), r, fill=False, color="red",
                                 ls=style, lw=1.2))
    ax.plot(0, 0, "r+", ms=12, mew=2)

    ax.set_aspect("equal")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Terrain top-down  |  %d boxes  |  flat r=%.2f m, outer r=%.2f m"
                 % (len(xs), inner, outer))
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print("Saved", out)
    _maybe_show(plt)


def view_3d(xs, ys, tops, half, out):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    if not xs:
        sys.exit("No box geoms found in the scene.")

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(xs, ys, [t * 100 for t in tops],
                    c=[t * 100 for t in tops], s=4, cmap="terrain")
    fig.colorbar(sc, ax=ax, shrink=0.6, label="top height (cm)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("top height (cm)")
    ax.set_title("Terrain top surfaces (%d boxes)" % len(xs))
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print("Saved", out)
    _maybe_show(plt)


def view_mujoco(path, with_robot):
    try:
        import mujoco
        import mujoco.viewer
    except ImportError:
        sys.exit("mujoco not installed. Run: pip install mujoco")

    import os

    scene_dir = os.path.dirname(os.path.abspath(path))
    with open(path) as f:
        xml = f.read()

    # Decide whether to keep the robot. Keep it only if asked AND g1.xml exists.
    include_re = re.compile(r'[ \t]*<include\s+file="([^"]+)"\s*/>\s*\n?')
    m = include_re.search(xml)
    robot_file = m.group(1) if m else None
    robot_path = os.path.join(scene_dir, robot_file) if robot_file else None
    have_robot = robot_path is not None and os.path.exists(robot_path)

    if with_robot and not have_robot:
        print("--with-robot requested but '%s' not found next to the scene; "
              "showing terrain only." % robot_file)
    keep_robot = with_robot and have_robot

    if keep_robot:
        print("Loading full scene with robot from", path)
        model = mujoco.MjModel.from_xml_path(path)
    else:
        # Strip the <include .../> so the terrain loads on its own.
        terrain_xml = include_re.sub("", xml, count=1) if m else xml
        print("Loading terrain only (robot include removed).")
        model = mujoco.MjModel.from_xml_string(terrain_xml, {})

    data = mujoco.MjData(model)
    print("Model: %d geoms. Opening viewer..." % model.ngeom)
    mujoco.viewer.launch(model, data)


def _maybe_show(plt):
    try:
        plt.show()
    except Exception:
        pass


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path", nargs="?", default=None,
                   help="scene XML path (or use --scene)")
    p.add_argument("--scene", choices=["rough", "perlin", "stairs"], default=None,
                   help="shortcut: resolves to scene_<scene>.xml")
    p.add_argument("--mode", choices=["map", "3d", "mujoco"], default=None,
                   help="view mode (default: mujoco when --scene is used, else map)")
    p.add_argument("--out", default=None, help="output image path (map/3d modes)")
    p.add_argument("--with-robot", action="store_true",
                   help="mujoco mode: also load g1.xml if present (default: terrain only)")
    args = p.parse_args()

    # resolve the scene file
    path = args.path or (("scene_%s.xml" % args.scene) if args.scene else None)
    if path is None:
        path = "scene_rough_circular.xml"  # backward-compatible default
    if not os.path.exists(path):
        sys.exit("Scene file not found: %s\n"
                 "Generate it first, e.g.:  python generate_scene.py --scene %s"
                 % (path, args.scene or "rough"))

    # default mode: mujoco when a --scene shortcut is used, else the 2D map
    mode = args.mode or ("mujoco" if args.scene else "map")

    if mode == "mujoco":
        view_mujoco(path, args.with_robot)
        return

    xs, ys, tops, half = parse_boxes(path)
    out = args.out or ("terrain_%s.png" % mode)
    if mode == "map":
        view_map(xs, ys, tops, half, out)
    else:
        view_3d(xs, ys, tops, half, out)


if __name__ == "__main__":
    main()
