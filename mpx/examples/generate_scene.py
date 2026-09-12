#!/usr/bin/env python3
"""Generate a MuJoCo terrain scene, selectable with --scene.

Two scene types are available (declared in SCENES):

  rough    circular cobblestone field: many tilted box geoms filling a disc
           around spawn, with a flat clearance disc at the center.

  perlin   height-field of simple, uniform undulations. Unlike the original
           "tapered-island" version, there is NO radial taper/scaling and NO
           flattened center pad -- the gentle undulations you saw at the
           center are extended uniformly across the whole field. A grayscale
           PNG is written next to the XML and referenced by <hfield>.

Per-scene parameters live in the ROUGH_PARAMS / PERLIN_PARAMS dictionaries
below; edit them there. The CLI only picks the scene and a few overrides.

Robot include:
  (no --robot)     no <include> (terrain-only scene, default)
  --robot g1       -> <include file="g1.xml"/>   (".xml" appended if missing)
  --robot none     -> explicit no include
"""

import argparse
import math
import random

import numpy as np
from PIL import Image

# --------------------------- available scenes -------------------------------
SCENES = ("rough", "perlin", "stairs")

# --------------------------- rough scene params -----------------------------
ROUGH_PARAMS = {
    "radius": 10.0,          # m, outer terrain radius from spawn
    "inner": 0.5,            # m, flat clearance disc at the center
    "spacing": 0.15,         # m, cobblestone grid pitch
    "size_xy_min": 0.090,    # m, box half-width lower bound
    "size_xy_max": 0.110,    # m, box half-width upper bound
    "base_z": 0.25,          # m, nominal box half-height (top at floor z=0)
    "top_jitter": 0.015,     # m, +/- top-surface jitter (~1.5 cm)
    "tilt_max": 0.06,        # rad, max random per-box tilt
    "friction": "1.0 0.02 0.001",
    "seed": 0,
}

# --------------------------- perlin scene params ----------------------------
PERLIN_PARAMS = {
    "size_x": 5.0,           # m, hfield half-extent in x
    "size_y": 5.0,           # m, hfield half-extent in y
    "resolution": 256,       # grid samples per side (PNG size)
    "amplitude": 0.06,       # m, +/- undulation amplitude
    "base_cells": 5,         # low-frequency cells across the field (wavelength)
    "octaves": 3,            # number of noise octaves summed
    "persistence": 0.5,      # amplitude falloff per octave
    "base_thickness": 0.10,  # m, solid thickness below the terrain
    "friction": "0.6",
    "png_name": "height_field_perlin.png",
    "seed": 0,
}

# --------------------------- stairs scene params ----------------------------
STAIRS_PARAMS = {
    "start_x": 0.5,          # m, x where the first step begins
    "num_steps": 12,         # number of steps going up (same coming down)
    "step_height": 0.05,     # m, rise of each step
    "step_run": 0.30,        # m, tread depth of each step (along x)
    "width": 1.0,            # m, half-width in y (staircase is 2*width wide)
    "platform_length": 2.0,  # m, flat walk on top before descending
    "friction": "1.0 0.02 0.001",
}
# -----------------------------------------------------------------------------


# ============================ rough scene ===================================
def _small_random_quat(rng, max_angle):
    axis = [rng.gauss(0.0, 1.0) for _ in range(3)]
    norm = math.sqrt(sum(c * c for c in axis)) or 1.0
    axis = [c / norm for c in axis]
    angle = rng.uniform(0.0, max_angle)
    s = math.sin(angle / 2.0)
    return (math.cos(angle / 2.0), axis[0] * s, axis[1] * s, axis[2] * s)


def _frange(start, stop, step):
    n = int(math.floor((stop - start) / step)) + 1
    return [start + i * step for i in range(n)]


def generate_rough(p, robot, out):
    rng = random.Random(p["seed"])
    coords = _frange(-p["radius"], p["radius"], p["spacing"])
    geoms = []
    for x in coords:
        for y in coords:
            r = math.hypot(x, y)
            if r < p["inner"] or r > p["radius"]:
                continue
            sxy = rng.uniform(p["size_xy_min"], p["size_xy_max"])
            sz = p["base_z"] + rng.uniform(-p["top_jitter"], p["top_jitter"])
            q = _small_random_quat(rng, p["tilt_max"])
            geoms.append(
                '    <geom pos="{:.4f} {:.4f} -0.25" type="box" '
                'size="{:.4f} {:.4f} {:.4f}" quat="{:.4f} {:.4f} {:.4f} {:.4f}" '
                'friction="{}"/>'.format(
                    x, y, sxy, sxy, sz, q[0], q[1], q[2], q[3], p["friction"]
                )
            )

    model_name = ("%s rough scene" % _robot_stem(robot)) if robot else "rough scene"
    header = (
        '<mujoco model="{model}">\n'
        '{inc}'
        '  <statistic center="0 0 0.5" extent="{extent}"/>\n\n'
        '  <visual>\n'
        '    <headlight diffuse="0.6 0.6 0.6" ambient="0.1 0.1 0.1" specular="0.9 0.9 0.9"/>\n'
        '    <rgba haze="0.15 0.25 0.35 1"/>\n'
        '    <global azimuth="140" elevation="-20"/>\n'
        '  </visual>\n\n'
        '  <asset>\n'
        '    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>\n'
        '    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"\n'
        '      markrgb="0.8 0.8 0.8" width="300" height="300"/>\n'
        '    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>\n'
        '  </asset>\n\n'
        '  <worldbody>\n'
        '    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>\n\n'
        '    <!-- Circular cobblestone field: annulus {inner} m <= r <= {outer} m,\n'
        '         flat clearance disc of radius {inner} m at the center. -->\n'
    ).format(
        model=model_name,
        inc=('  <include file="%s"/>\n\n' % robot) if robot else "",
        extent=int(p["radius"]),
        inner=p["inner"],
        outer=p["radius"],
    )
    with open(out, "w") as f:
        f.write(header)
        f.write("\n".join(geoms))
        f.write("\n  </worldbody>\n</mujoco>\n")
    print("Wrote %s (rough): %d boxes, %s." % (
        out, len(geoms), _robot_note(robot)))


# ============================ perlin scene ==================================
def _fade(t):
    return t * t * t * (t * (t * 6 - 15) + 10)


def _value_noise_2d(res, cells, rng):
    """Smooth value noise on a res x res grid, `cells` lattice cells per side."""
    lat = rng.random((cells + 1, cells + 1))
    c = np.linspace(0.0, cells, res, endpoint=False)
    idx = np.floor(c).astype(int)
    frac = _fade(c - idx)
    idx1 = np.minimum(idx + 1, cells)
    # interpolate along x, then along y (separable)
    row = lat[:, idx] * (1 - frac)[None, :] + lat[:, idx1] * frac[None, :]
    out = row[idx, :] * (1 - frac)[:, None] + row[idx1, :] * frac[:, None]
    return out


def _fractal_noise(res, base_cells, octaves, persistence, rng):
    h = np.zeros((res, res))
    amp, total, cells = 1.0, 0.0, base_cells
    for _ in range(octaves):
        h += amp * _value_noise_2d(res, cells, rng)
        total += amp
        amp *= persistence
        cells *= 2
    h /= total
    h -= h.mean()
    peak = max(abs(h.min()), abs(h.max())) or 1.0
    return h / peak  # normalized to [-1, 1], mean ~0


def generate_perlin(p, robot, out):
    import os

    rng = np.random.default_rng(p["seed"])
    res = p["resolution"]
    h = _fractal_noise(res, p["base_cells"], p["octaves"], p["persistence"], rng)
    h = h * p["amplitude"]  # meters, roughly in [-amplitude, +amplitude]

    h_min, h_max = float(h.min()), float(h.max())
    elevation_z = h_max - h_min           # physical span mapped from PNG [0,1]
    geom_pos_z = -elevation_z / 2.0        # center the undulations around z=0
    floor_z = geom_pos_z - 0.003           # plain 3 mm below the lowest point

    # write grayscale PNG (higher value = higher terrain)
    out_dir = os.path.dirname(os.path.abspath(out))
    png_path = os.path.join(out_dir, p["png_name"])
    img = ((h - h_min) / (elevation_z if elevation_z else 1.0) * 255.0)
    Image.fromarray(img.astype(np.uint8), mode="L").save(png_path)

    model_name = ("%s perlin scene" % _robot_stem(robot)) if robot else "perlin scene"
    xml = (
        '<mujoco model="{model}">\n'
        '{inc}'
        '  <statistic center="0 0 0.1" extent="{extent:.1f}"/>\n\n'
        '  <visual>\n'
        '    <headlight diffuse="0.3 0.3 0.3" ambient="0.3 0.3 0.3" specular="0 0 0"/>\n'
        '    <rgba haze="0.15 0.25 0.35 1"/>\n'
        '    <global azimuth="-130" elevation="-20"/>\n'
        '  </visual>\n\n'
        '  <asset>\n'
        '    <texture type="skybox" builtin="gradient" rgb1="0.99 0.99 0.99" rgb2="0.99 0.99 0.99" width="512" height="3072"/>\n'
        '    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.80 0.80 0.80" rgb2="0.99 0.99 0.99"\n'
        '      markrgb="0.3 0.3 0.3" width="300" height="300"/>\n'
        '    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.0"/>\n'
        '    <!-- Uniform Perlin undulations (no island taper, no flat pad).\n'
        '         size = "half_x half_y elevation base"; elevation and geom pos\n'
        '         below are derived from the generated grid. -->\n'
        '    <hfield name="perlin_hfield" size="{sx:.4f} {sy:.4f} {elev:.4f} {base:.4f}" file="{png}"/>\n'
        '  </asset>\n\n'
        '  <worldbody>\n'
        '    <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>\n'
        '    <geom name="floor" size="0 0 0.05" type="plane" pos="0 0 {floor_z:.4f}" material="groundplane"\n'
        '          contype="1" conaffinity="0" priority="1" friction="{fr}" condim="3"/>\n'
        '    <geom type="hfield" hfield="perlin_hfield" pos="0 0 {gz:.4f}" quat="1 0 0 0" material="groundplane"\n'
        '          contype="1" conaffinity="0" condim="3" priority="1" friction="{fr}"/>\n'
        '  </worldbody>\n'
        '</mujoco>\n'
    ).format(
        model=model_name,
        inc=('  <include file="%s"/>\n\n' % robot) if robot else "",
        extent=max(p["size_x"], p["size_y"]),
        sx=p["size_x"], sy=p["size_y"], elev=elevation_z, base=p["base_thickness"],
        png=p["png_name"], floor_z=floor_z, gz=geom_pos_z, fr=p["friction"],
    )
    with open(out, "w") as f:
        f.write(xml)
    print("Wrote %s (perlin) + %s: %dx%d grid, undulation +/-%.1f cm, %s." % (
        out, p["png_name"], res, res, p["amplitude"] * 100, _robot_note(robot)))


# ============================ stairs scene ==================================
def generate_stairs(p, robot, out):
    h = p["step_height"]
    run = p["step_run"]
    n = p["num_steps"]
    w = p["width"]
    sx = run / 2.0
    x0 = p["start_x"]
    fr = p["friction"]

    def box(cx, top):
        # solid block from z=0 up to `top`, so risers are closed
        return (
            '    <geom pos="{:.4f} 0 {:.4f}" type="box" size="{:.4f} {:.4f} {:.4f}" '
            'friction="{}"/>'.format(cx, top / 2.0, sx, w, top / 2.0, fr)
        )

    geoms = []

    # ascending steps: step k reaches height k*h
    for k in range(1, n + 1):
        cx = x0 + (k - 0.5) * run
        geoms.append(box(cx, k * h))

    # top platform (flat walk of platform_length at height n*h)
    x1 = x0 + n * run
    plat = p["platform_length"]
    geoms.append(
        '    <geom pos="{:.4f} 0 {:.4f}" type="box" size="{:.4f} {:.4f} {:.4f}" '
        'friction="{}"/>'.format(
            x1 + plat / 2.0, n * h / 2.0, plat / 2.0, w, n * h / 2.0, fr)
    )

    # descending steps: mirror of the ascent, back down to ground
    x2 = x1 + plat
    for m in range(1, n + 1):
        top = (n - m) * h
        if top <= 0:
            continue
        cx = x2 + (m - 0.5) * run
        geoms.append(box(cx, top))

    total_len = x2 + n * run - x0
    center_x = x0 + total_len / 2.0

    model_name = ("%s stairs scene" % _robot_stem(robot)) if robot else "stairs scene"
    header = (
        '<mujoco model="{model}">\n'
        '{inc}'
        '  <statistic center="{cx:.2f} 0 0.3" extent="{extent:.1f}"/>\n\n'
        '  <visual>\n'
        '    <headlight diffuse="0.6 0.6 0.6" ambient="0.1 0.1 0.1" specular="0.9 0.9 0.9"/>\n'
        '    <rgba haze="0.15 0.25 0.35 1"/>\n'
        '    <global azimuth="140" elevation="-20"/>\n'
        '  </visual>\n\n'
        '  <asset>\n'
        '    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>\n'
        '    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"\n'
        '      markrgb="0.8 0.8 0.8" width="300" height="300"/>\n'
        '    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>\n'
        '  </asset>\n\n'
        '  <worldbody>\n'
        '    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>\n\n'
        '    <!-- Long staircase: starts at x={x0}, {n} steps up (rise {rise} m,\n'
        '         run {run} m), a {plat} m flat platform, then {n} steps down. -->\n'
    ).format(
        model=model_name,
        inc=('  <include file="%s"/>\n\n' % robot) if robot else "",
        cx=center_x, extent=max(total_len, n * h * 4),
        x0=x0, n=n, rise=h, run=run, plat=plat,
    )
    with open(out, "w") as f:
        f.write(header)
        f.write("\n".join(geoms))
        f.write("\n  </worldbody>\n</mujoco>\n")
    print("Wrote %s (stairs): %d steps up + %.1f m platform + %d steps down, "
          "top height %.2f m, %s." % (
              out, n, plat, n, n * h, _robot_note(robot)))


# ============================ shared helpers ================================
def _robot_stem(robot):
    return robot[:-4] if robot and robot.lower().endswith(".xml") else robot


def _robot_note(robot):
    return ("include '%s'" % robot) if robot else "no robot include"


def _normalize_robot(robot):
    if robot is None:
        return None
    r = robot.strip()
    if r.lower() in ("none", ""):
        return None
    if not r.lower().endswith(".xml"):
        r += ".xml"
    return r


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--scene", choices=SCENES, default="rough",
                   help="which scene to generate (default: %(default)s)")
    p.add_argument("--robot", default=None,
                   help='robot file to <include>, or "none". ".xml" appended '
                        'if missing. Default: no include.')
    p.add_argument("--out", default=None,
                   help="output XML path (default: scene_<scene>.xml)")
    p.add_argument("--seed", type=int, default=None,
                   help="override the scene's RNG seed")
    args = p.parse_args()

    robot = _normalize_robot(args.robot)
    out = args.out or ("scene_%s.xml" % args.scene)

    if args.scene == "rough":
        params = dict(ROUGH_PARAMS)
        if args.seed is not None:
            params["seed"] = args.seed
        generate_rough(params, robot, out)
    elif args.scene == "perlin":
        params = dict(PERLIN_PARAMS)
        if args.seed is not None:
            params["seed"] = args.seed
        generate_perlin(params, robot, out)
    else:  # stairs
        generate_stairs(dict(STAIRS_PARAMS), robot, out)


if __name__ == "__main__":
    main()
