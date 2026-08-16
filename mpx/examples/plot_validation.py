"""
plot_validation.py

Logger + plotting functions to validate the Tita MPC/WBC controller.
Each plotting function also saves its own CSV with the plotted data.

State layout (from main):
    tita_state = [pcom(3), vcom(3), pl_world(3), pr_world(3), dpl_world(3), dpr_world(3)]
    x0         = [pcom(3), vcom(3), c_world(3), vc_world_z(1), theta(1), v(1), w(1)]

Derived quantities:
    c_world  = (pl_world  + pr_world ) / 2
    vc_world = (dpl_world + dpr_world) / 2     # center velocity in world frame

Base orientation and angular velocity are read directly from MuJoCo (model/data),
since they are not available in tita_state/x0:
    base quaternion -> RPY (roll, pitch, yaw)
    base omega      -> angular velocity (yaw rate = omega[2])
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #
def _vec3(x):
    """Flatten any array (3,), (1,3), (3,1)... down to a (3,) vector.

    Always returns a fresh copy: inputs may be views into a live buffer
    (e.g. data.xfrc_applied), which would otherwise alias across samples.
    """
    a = np.array(x, dtype=np.float64).reshape(-1)   # np.array copies by default
    return a[:3].copy()


def _R_yaw(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0., 0., 1.0]])


def quat_to_rpy(quat):
    """
    Convert a MuJoCo quaternion [w, x, y, z] to (roll, pitch, yaw) [rad].
    Tait-Bryan ZYX convention.
    """
    q = np.asarray(quat, dtype=np.float64).reshape(-1)
    w, x, y, z = q[0], q[1], q[2], q[3]

    # roll (rotation about x)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # pitch (rotation about y)
    sinp = 2.0 * (w * y - z * x)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)

    # yaw (rotation about z)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw])


def _base_quat_omega(model, data, base_body_name="base_link"):
    """
    Extract the base quaternion [w,x,y,z] and angular velocity omega(3) from MuJoCo.
    Uses the base free-joint address when available; otherwise falls back to
    qpos[3:7] / qvel[3:6] (base free joint at the head of the state).
    """
    try:
        import mujoco
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, base_body_name)
        jadr = int(model.body_jntadr[bid])
        if jadr >= 0:
            qadr = int(model.jnt_qposadr[jadr])
            vadr = int(model.jnt_dofadr[jadr])
            quat = np.asarray(data.qpos[qadr + 3: qadr + 7], dtype=np.float64)
            omega = np.asarray(data.qvel[vadr + 3: vadr + 6], dtype=np.float64)
            return quat, omega
    except Exception:
        pass
    # fallback
    quat = np.asarray(data.qpos[3:7], dtype=np.float64)
    omega = np.asarray(data.qvel[3:6], dtype=np.float64)
    return quat, omega


def _save_csv(path, header, columns):
    """columns: list of 1D arrays of the same length."""
    if path is None:
        return
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    data = np.column_stack(columns)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(data)


def _vc_body(d):
    """vc_world rotated into the body frame, sample by sample."""
    vc_world = d["vc_world"]
    theta = d["theta"]
    out = np.empty_like(vc_world)
    for i in range(len(theta)):
        out[i] = _R_yaw(theta[i]).T @ vc_world[i]
    return out


# --------------------------------------------------------------------------- #
#  Logger
# --------------------------------------------------------------------------- #
class SimLogger:
    """
    Accumulates signals during the simulation.
    Call .append(...) every step (or every control step) and then pass the
    instance to the plotting functions.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.t = []
        self.pcom = []        # (3,)  CoM position (world)
        self.vcom = []        # (3,)  CoM velocity (world)
        self.c_world = []     # (3,)  center between feet (world)
        self.vc_world = []    # (3,)  center velocity (world)
        self.theta = []       # yaw from x0 (reconstructed from the feet)
        self.v = []           # forward velocity (body) from x0
        self.w = []           # yaw rate from x0 (reconstructed from the feet)
        self.cmd = []         # (3,) [vx, vy, wz] commanded
        self.F_com = []       # (3,) resultant force at the CoM (fl + fr)
        self.ext_force = []   # (3,) external disturbance force applied at the CoM
        self.rpy = []         # (3,) base orientation (roll, pitch, yaw) from MuJoCo
        self.omega = []       # (3,) base angular velocity from MuJoCo

    def append(self, t, model, data, tita_state, x0, cmd,
               fl=None, fr=None, ext_force=None, base_body_name="base_link"):
        ts = np.asarray(tita_state, dtype=np.float64).reshape(-1)
        x = np.asarray(x0, dtype=np.float64).reshape(-1)

        pcom = ts[0:3]
        vcom = ts[3:6]
        pl, pr = ts[6:9], ts[9:12]
        dpl, dpr = ts[12:15], ts[15:18]

        c_world = (pl + pr) / 2.0
        vc_world = (dpl + dpr) / 2.0

        if fl is not None and fr is not None:
            F = _vec3(fl) + _vec3(fr)
        else:
            F = np.full(3, np.nan)

        ext_F = _vec3(ext_force) if ext_force is not None else np.full(3, np.nan)

        # --- orientation and angular velocity read from the model ---
        quat, omega = _base_quat_omega(model, data, base_body_name)
        rpy = quat_to_rpy(quat)

        self.t.append(float(t))
        self.pcom.append(pcom.copy())
        self.vcom.append(vcom.copy())
        self.c_world.append(c_world.copy())
        self.vc_world.append(vc_world.copy())
        self.theta.append(float(x[10]))
        self.v.append(float(x[11]))
        self.w.append(float(x[12]))
        self.cmd.append(_vec3(cmd).copy())
        self.F_com.append(F.copy())
        self.ext_force.append(ext_F.copy())
        self.rpy.append(rpy.copy())
        self.omega.append(np.asarray(omega, dtype=np.float64).copy())

    # -- conversion to numpy arrays --------------------------------------- #
    def arrays(self):
        return dict(
            t=np.asarray(self.t),
            pcom=np.asarray(self.pcom),
            vcom=np.asarray(self.vcom),
            c_world=np.asarray(self.c_world),
            vc_world=np.asarray(self.vc_world),
            theta=np.asarray(self.theta),
            v=np.asarray(self.v),
            w=np.asarray(self.w),
            cmd=np.asarray(self.cmd),
            F_com=np.asarray(self.F_com),
            ext_force=np.asarray(self.ext_force),
            rpy=np.asarray(self.rpy),
            omega=np.asarray(self.omega),
        )

    # Full raw-data column layout (single CSV that fully reconstructs the log)
    _RAW_HEADER = [
        "t",
        "pcom_x", "pcom_y", "pcom_z",
        "vcom_x", "vcom_y", "vcom_z",
        "c_world_x", "c_world_y", "c_world_z",
        "vc_world_x", "vc_world_y", "vc_world_z",
        "theta", "v", "w",
        "cmd_vx", "cmd_vy", "cmd_wz",
        "F_x", "F_y", "F_z",
        "ext_x", "ext_y", "ext_z",
        "rpy_r", "rpy_p", "rpy_y",
        "omega_x", "omega_y", "omega_z",
    ]

    def save_csv_data(self, path):
        """Dump ALL raw logged fields to a single CSV (fully reloadable)."""
        d = self.arrays()
        columns = [
            d["t"],
            d["pcom"][:, 0], d["pcom"][:, 1], d["pcom"][:, 2],
            d["vcom"][:, 0], d["vcom"][:, 1], d["vcom"][:, 2],
            d["c_world"][:, 0], d["c_world"][:, 1], d["c_world"][:, 2],
            d["vc_world"][:, 0], d["vc_world"][:, 1], d["vc_world"][:, 2],
            d["theta"], d["v"], d["w"],
            d["cmd"][:, 0], d["cmd"][:, 1], d["cmd"][:, 2],
            d["F_com"][:, 0], d["F_com"][:, 1], d["F_com"][:, 2],
            d["ext_force"][:, 0], d["ext_force"][:, 1], d["ext_force"][:, 2],
            d["rpy"][:, 0], d["rpy"][:, 1], d["rpy"][:, 2],
            d["omega"][:, 0], d["omega"][:, 1], d["omega"][:, 2],
        ]
        _save_csv(path, self._RAW_HEADER, columns)

    def load_csv_data(self, path):
        """Fill the logger arrays from a raw CSV produced by save_csv_data."""
        self.reset()
        raw = np.genfromtxt(path, delimiter=",", names=True, dtype=np.float64)
        raw = np.atleast_1d(raw)
        n = raw.shape[0]
        for i in range(n):
            r = raw[i]
            self.t.append(float(r["t"]))
            self.pcom.append(np.array([r["pcom_x"], r["pcom_y"], r["pcom_z"]]))
            self.vcom.append(np.array([r["vcom_x"], r["vcom_y"], r["vcom_z"]]))
            self.c_world.append(np.array([r["c_world_x"], r["c_world_y"], r["c_world_z"]]))
            self.vc_world.append(np.array([r["vc_world_x"], r["vc_world_y"], r["vc_world_z"]]))
            self.theta.append(float(r["theta"]))
            self.v.append(float(r["v"]))
            self.w.append(float(r["w"]))
            self.cmd.append(np.array([r["cmd_vx"], r["cmd_vy"], r["cmd_wz"]]))
            self.F_com.append(np.array([r["F_x"], r["F_y"], r["F_z"]]))
            self.ext_force.append(np.array([r["ext_x"], r["ext_y"], r["ext_z"]]))
            self.rpy.append(np.array([r["rpy_r"], r["rpy_p"], r["rpy_y"]]))
            self.omega.append(np.array([r["omega_x"], r["omega_y"], r["omega_z"]]))
        return self


# --------------------------------------------------------------------------- #
#  Function 1: velocity tracking
# --------------------------------------------------------------------------- #
def plot_velocity_tracking(log, save_path=None, csv_path="vel_tracking.csv", show=True):
    """
    Three subplots:
      - vx:      vc_world rotated into the body frame (x)  vs  command (vx)
      - vy:      vc_world rotated into the body frame (y)  vs  command (vy)
      - angular: yaw rate measured from MuJoCo (omega_z)   vs  command (wz)
    Saves CSV with: t, vx_meas, vx_cmd, vy_meas, vy_cmd, w_meas, wz_cmd
    """
    d = log.arrays() if isinstance(log, SimLogger) else log
    t = d["t"]
    vc_body = _vc_body(d)
    cmd = d["cmd"]                 # [vx, vy, wz]
    w_meas = d["omega"][:, 2]      # yaw rate measured from the model

    _save_csv(
        csv_path,
        ["t", "vx_meas", "vx_cmd", "vy_meas", "vy_cmd", "w_meas", "wz_cmd"],
        [t, vc_body[:, 0], cmd[:, 0], vc_body[:, 1], cmd[:, 1], w_meas, cmd[:, 2]],
    )

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    ax = axes[0]
    ax.plot(t, vc_body[:, 0], color="tab:blue", label="vx measured")
    ax.plot(t, cmd[:, 0], "--", color="tab:blue", label="vx commanded")
    ax.set_ylabel("vx [m/s]")
    ax.set_title("Center linear velocity tracking - X")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    ax = axes[1]
    ax.plot(t, vc_body[:, 1], color="tab:orange", label="vy measured (body)")
    ax.plot(t, cmd[:, 1], "--", color="tab:orange", label="vy commanded")
    ax.set_ylabel("vy [m/s]")
    ax.set_title("Center linear velocity tracking - Y")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    ax = axes[2]
    ax.plot(t, w_meas, color="tab:green", label="w measured")
    ax.plot(t, cmd[:, 2], "--", color="tab:green", label="wz commanded")
    ax.set_ylabel("angular velocity [rad/s]")
    ax.set_xlabel("time [s]")
    ax.set_title("Angular velocity tracking (yaw)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig


# --------------------------------------------------------------------------- #
#  Function 2: velocity tracking errors
# --------------------------------------------------------------------------- #
def plot_velocity_error(log, save_path=None, csv_path="vel_error.csv", show=True):
    """
    Plots the velocity tracking errors together with their corresponding commands:
      - linear plot:   vx_cmd - vx_meas, vy_cmd - vy_meas, plus vx_cmd, vy_cmd
      - angular plot:  wz_cmd - wz_meas, plus wz_cmd

    Saves CSV with:
      t, err_vx, err_vy, err_w, vx_cmd, vy_cmd, wz_cmd
    """
    d = log.arrays() if isinstance(log, SimLogger) else log

    t = d["t"]
    vc_body = _vc_body(d)
    cmd = d["cmd"]
    w_meas = d["omega"][:, 2]

    err_vx = cmd[:, 0] - vc_body[:, 0]
    err_vy = cmd[:, 1] - vc_body[:, 1]
    err_w = cmd[:, 2] - w_meas

    _save_csv(
        csv_path,
        ["t", "err_vx", "err_vy", "err_w", "vx_cmd", "vy_cmd", "wz_cmd"],
        [t, err_vx, err_vy, err_w, cmd[:, 0], cmd[:, 1], cmd[:, 2]],
    )

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    # ============================================================
    # VX ERROR + VX COMMAND
    # ============================================================
    ax = axes[0]
    ax.plot(t, err_vx, color="tab:blue", label="vx error (cmd - meas)")
    ax.plot(t, cmd[:, 0], linestyle="--", color="tab:blue", label="vx cmd")
    ax.set_ylabel("vx error [m/s]")
    ax.set_title("Linear velocity tracking error - X")
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.legend(loc="best")

    # ============================================================
    # VY ERROR + VY COMMAND
    # ============================================================
    ax = axes[1]
    ax.plot(t, err_vy, color="tab:orange", label="vy error (cmd - meas)")
    ax.plot(t, cmd[:, 1], linestyle="--", color="tab:orange", label="vy cmd")
    ax.set_ylabel("vy error [m/s]")
    ax.set_title("Linear velocity tracking error - Y")
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.legend(loc="best")

    # ============================================================
    # ANGULAR VELOCITY ERROR + ANGULAR COMMAND
    # ============================================================
    ax = axes[2]
    ax.plot(t, err_w, color="tab:green", label="w error (cmd - meas)")
    ax.plot(t, cmd[:, 2], linestyle="--", color="tab:green", label="wz cmd")
    ax.set_ylabel("angular velocity error [rad/s]")
    ax.set_xlabel("time [s]")
    ax.set_title("Angular velocity tracking error (yaw)")
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.legend(loc="best")

    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig


# --------------------------------------------------------------------------- #
#  Function 3: CoM position (X, Y, Z) + forces at the CoM
# --------------------------------------------------------------------------- #
def plot_com_and_forces(log, save_path=None, csv_path="com_forces.csv", show=True):
    """
    Four subplots:
      - CoM X position
      - CoM Y position
      - CoM Z position
      - external disturbance force applied at the CoM (x, y, z components)
    Saves CSV with: t, x, y, z, Fx, Fy, Fz   (F = external force at the CoM)
    """
    d = log.arrays() if isinstance(log, SimLogger) else log
    t = d["t"]
    pcom = d["pcom"]
    F = d["ext_force"]

    _save_csv(
        csv_path,
        ["t", "x", "y", "z", "Fx", "Fy", "Fz"],
        [t, pcom[:, 0], pcom[:, 1], pcom[:, 2], F[:, 0], F[:, 1], F[:, 2]],
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    ax = axes[0, 0]
    ax.plot(t, pcom[:, 0], color="tab:blue")
    ax.set_ylabel("X [m]")
    ax.set_title("CoM position - X")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(t, pcom[:, 1], color="tab:orange")
    ax.set_ylabel("Y [m]")
    ax.set_title("CoM position - Y")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(t, pcom[:, 2], color="tab:green")
    ax.set_ylabel("Z [m]")
    ax.set_xlabel("time [s]")
    ax.set_title("CoM position - Z")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]

# --------------------------------------------------------------------------- #
#  Function 3: CoM position (X, Y, Z) + forces at the CoM
# --------------------------------------------------------------------------- #
def plot_com_and_forces(log, save_path=None, csv_path="com_forces.csv", show=True):
    """
    Four subplots:
      - CoM X position
      - CoM Y position
      - CoM Z position
      - external disturbance force applied at the CoM (x, y, z components)
    Saves CSV with: t, x, y, z, Fx, Fy, Fz   (F = external force at the CoM)
    """
    d = log.arrays() if isinstance(log, SimLogger) else log
    t = d["t"]
    pcom = d["pcom"]
    F = d["ext_force"]

    _save_csv(
        csv_path,
        ["t", "x", "y", "z", "Fx", "Fy", "Fz"],
        [t, pcom[:, 0], pcom[:, 1], pcom[:, 2], F[:, 0], F[:, 1], F[:, 2]],
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    ax = axes[0, 0]
    ax.plot(t, pcom[:, 0], color="tab:blue")
    ax.set_ylabel("X [m]")
    ax.set_title("CoM position - X")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(t, pcom[:, 1], color="tab:orange")
    ax.set_ylabel("Y [m]")
    ax.set_title("CoM position - Y")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(t, pcom[:, 2], color="tab:green")
    ax.set_ylabel("Z [m]")
    ax.set_xlabel("time [s]")
    ax.set_title("CoM position - Z")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(t, F[:, 0], color="tab:red", label="Fx")
    ax.plot(t, F[:, 1], color="tab:purple", label="Fy")
    ax.plot(t, F[:, 2], color="tab:brown", label="Fz")
    ax.set_ylabel("force [N]")
    ax.set_xlabel("time [s]")
    ax.set_title("External disturbance force at the CoM")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    ax.plot(t, F[:, 0], color="tab:red", label="Fx")
    ax.plot(t, F[:, 1], color="tab:purple", label="Fy")
    ax.plot(t, F[:, 2], color="tab:brown", label="Fz")
    ax.set_ylabel("force [N]")
    ax.set_xlabel("time [s]")
    ax.set_title("External disturbance force at the CoM")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    ax.axhline(0.0, color="k", lw=0.8, alpha=0.5)
    ax.set_ylabel("linear velocity [m/s]")
    ax.set_title("Tracking error and command - linear velocity")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    # ============================================================
    # ANGULAR VELOCITY ERROR + ANGULAR COMMAND
    # ============================================================
    ax = axes[1]

    ax.plot(t, err_w, color="tab:green", label="omega error (cmd - meas)")

    ax.plot(
        t,
        cmd[:, 2],
        linestyle="--",
        color="tab:green",
        label="omega cmd",
    )

    ax.axhline(0.0, color="k", lw=0.8, alpha=0.5)
    ax.set_ylabel("angular velocity [rad/s]")
    ax.set_xlabel("time [s]")
    ax.set_title("Tracking error and command - angular velocity")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig

# --------------------------------------------------------------------------- #
#  Function 3: CoM position (X, Y, Z) + forces at the CoM
# --------------------------------------------------------------------------- #
def plot_com_and_forces(log, save_path=None, csv_path="com_forces.csv", show=True):
    """
    Four subplots:
      - CoM X position
      - CoM Y position
      - CoM Z position
      - external disturbance force applied at the CoM (x, y, z components)
    Saves CSV with: t, x, y, z, Fx, Fy, Fz   (F = external force at the CoM)
    """
    d = log.arrays() if isinstance(log, SimLogger) else log
    t = d["t"]
    pcom = d["pcom"]
    F = d["ext_force"]

    _save_csv(
        csv_path,
        ["t", "x", "y", "z", "Fx", "Fy", "Fz"],
        [t, pcom[:, 0], pcom[:, 1], pcom[:, 2], F[:, 0], F[:, 1], F[:, 2]],
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    ax = axes[0, 0]
    ax.plot(t, pcom[:, 0], color="tab:blue")
    ax.set_ylabel("X [m]")
    ax.set_title("CoM position - X")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(t, pcom[:, 1], color="tab:orange")
    ax.set_ylabel("Y [m]")
    ax.set_title("CoM position - Y")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(t, pcom[:, 2], color="tab:green")
    ax.set_ylabel("Z [m]")
    ax.set_xlabel("time [s]")
    ax.set_title("CoM position - Z")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(t, F[:, 0], color="tab:red", label="Fx")
    ax.plot(t, F[:, 1], color="tab:purple", label="Fy")
    ax.plot(t, F[:, 2], color="tab:brown", label="Fz")
    ax.set_ylabel("force [N]")
    ax.set_xlabel("time [s]")
    ax.set_title("External disturbance force at the CoM")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig


# --------------------------------------------------------------------------- #
def raw_csv_path(save_path, prefix="tita"):
    """Path of the full raw CSV that fully reconstructs the logger."""
    return os.path.join(save_path, "csv", f"{prefix}_raw.csv")


def plot_all(log, save_path, prefix="tita", show=False):
    """Convenience: generate the three plots and the three per-plot CSVs,
    plus a full raw CSV that can be reloaded with SimLogger.load_csv_data."""
    print(f"Generating plots and CSVs in {save_path} with prefix '{prefix}'")
    os.makedirs(save_path, exist_ok=True)
    # full raw dump (reloadable round-trip)
    if isinstance(log, SimLogger):
        log.save_csv_data(raw_csv_path(save_path, prefix))
    plot_velocity_tracking(log,
                           save_path=os.path.join(save_path, f"{prefix}_vel_tracking.png"),
                           csv_path=os.path.join(save_path, "csv", f"{prefix}_vel_tracking.csv"), show=show)
    plot_velocity_error(log, save_path=os.path.join(save_path, f"{prefix}_vel_error.png"),
                        csv_path=os.path.join(save_path, "csv", f"{prefix}_vel_error.csv"), show=show)
    plot_com_and_forces(log, save_path=os.path.join(save_path, f"{prefix}_com_forces.png"),
                        csv_path=os.path.join(save_path, "csv", f"{prefix}_com_forces.csv"), show=show)


# --------------------------------------------------------------------------- #
#  Standalone entry point: reload a logged run from CSV and re-render plots
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import argparse


    dir_path = os.path.dirname(os.path.realpath(__file__))
    TITA_PATH = os.path.join(dir_path, "plots","tita_validation","plots")
    parser = argparse.ArgumentParser(
        description="Reload a logged run from the raw CSV and regenerate the plots."
    )
    parser.add_argument("--path", default=TITA_PATH,
                        help="Output directory used by plot_all (contains the csv/ subfolder).")
    parser.add_argument("--prefix", default="tita", help="File prefix used by plot_all.")
    parser.add_argument("--show", action="store_true", help="Show the figures interactively.")
    args = parser.parse_args()

    raw = raw_csv_path(args.path, args.prefix)
    if not os.path.isfile(raw):
        raise FileNotFoundError(f"Raw CSV not found: {raw}")

    logger = SimLogger()
    logger.load_csv_data(raw)
    plot_all(logger, args.path, prefix=args.prefix, show=args.show)
    print(f"Reloaded {raw} and regenerated plots in {args.path}")