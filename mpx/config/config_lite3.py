"""SRBD MPC configuration for the DeepRobotics Lite3.

Mirrors config_srbd.py (Aliengo) field by field. Every physical value is derived
from the Lite3 model itself, not scaled from Aliengo -- the two robots differ by
a factor 2 in mass and by ~20% in leg length.

Two deliberate differences from config_srbd.py:

* ``model_path`` points at the *same* XML the simulation uses (the Playground
  Lite3 flat scene). config_srbd.py instead points at mpx/data/aliengo/aliengo.xml,
  which is a different model from the Playground Aliengo: its hip spacing is
  y=+-0.194 against the Playground's +-0.134, so the whole-body controller there
  runs on a geometry the simulator does not have. Sharing one model removes that
  whole class of mismatch.
* ``inertia`` is the composite inertia of the whole robot at the nominal stance,
  about the CoM, expressed in the base frame. config_srbd.py's Aliengo inertia is
  ~1.62x the composite value of its own model in Iyy/Izz, i.e. hand-inflated.
"""

import jax
import jax.numpy as jnp

from mujoco_playground._src.locomotion.lite3 import lite3_constants as lite3_consts

# Same model the simulator steps, so WBC kinematics and simulation cannot drift.
model_path = lite3_consts.task_to_xml("flat_terrain").as_posix()

# Joint order as declared in lite3.xml: FL, FR, HL, HR x HipX, HipY, Knee.
# This is also the actuator order and therefore the order of the WBC torques.
joints_name = [
    'FL_HipX_joint', 'FL_HipY_joint', 'FL_Knee_joint',
    'FR_HipX_joint', 'FR_HipY_joint', 'FR_Knee_joint',
    'HL_HipX_joint', 'HL_HipY_joint', 'HL_Knee_joint',
    'HR_HipX_joint', 'HR_HipY_joint', 'HR_Knee_joint',
]

# Foot collision geoms and the bodies that carry them. Lite3 has a dedicated
# *_FOOT body per leg (Aliengo hangs the contact geom off the calf instead).
contact_frame = ['FL', 'FR', 'HL', 'HR']
body_name = ['FL_FOOT', 'FR_FOOT', 'HL_FOOT', 'HR_FOOT']

# Time and stage parameters. Same structure as Aliengo; the horizon covers
# N*dt = 0.5 s, i.e. ~0.7 gait cycles at step_freq below.
dt = 0.02
N = 25
mpc_frequency = 50
whole_body_frequency = 200

# Trot: FL and HR in phase, FR and HL in antiphase. The array is indexed in the
# contact_frame order above, so the Aliengo values carry over unchanged only
# because FL,FR,HL,HR has the same diagonal pairing as FL,FR,RL,RR.
timer_t = jnp.array([0.5, 0.0, 0.0, 0.5])
duty_factor = 0.65

# Aliengo: step_freq 1.35 at a 0.35 m hip height. Froude scaling f ~ sqrt(g/h)
# gives 1.35*sqrt(0.35/0.31) = 1.43 for Lite3's 0.31 m.
step_freq = 1.4
# Aliengo: 0.065 at 0.50 m of leg. Scaled by leg length 0.41/0.50 -> 0.053.
step_height = 0.055
clearence_speed = 0.4

# Base height tracked by the MPC. The `home` keyframe puts the base at 0.310 m
# with the feet 2.3 mm off the ground, so this is the standing height.
robot_height = 0.31
use_terrain_estimator = False

# Initial state: the `home` keyframe of lite3.xml.
p0 = jnp.array([0.0, 0.0, 0.31])
quat0 = jnp.array([1.0, 0.0, 0.0, 0.0])
q0 = jnp.array([
    0.0, -0.8, 1.6,
    0.0, -0.8, 1.6,
    0.0, -0.8, 1.6,
    0.0, -0.8, 1.6,
])
q0_init = q0

# Nominal foot positions in the base frame at the `home` pose, z flattened to 0
# (measured with mj_forward on the keyframe).
p_legs0 = jnp.array([
    0.181760,  0.159350, 0.0,   # FL
    0.181760, -0.159350, 0.0,   # FR
   -0.167240,  0.159350, 0.0,   # HL
   -0.167240, -0.159350, 0.0,   # HR
])

n_joints = len(joints_name)
n_contact = len(contact_frame)
n = 13                 # SRBD state: p(3) + quat(4) + v(3) + omega(3)
m = 3 * n_contact      # controls: one GRF per foot

# Whole-robot mass and composite inertia at the nominal stance (mj_forward on
# the `home` keyframe, summed over all bodies about the CoM, rotated into the
# base frame). Aliengo for reference: 24.638 kg.
mass = 11.9376
inertia = jnp.array([
    [ 0.170705, -0.000704, -0.012919],
    [-0.000704,  0.301469, -0.000045],
    [-0.012919, -0.000045,  0.371890],
])

grf_ref = jnp.zeros(3 * n_contact)
u_ref = jnp.concatenate([grf_ref])

# Whole-body controller: Cartesian foot tracking gains. Kept at the Aliengo
# values on purpose -- the WBC output is M @ pinv(J.T) @ (Kp*e + Kd*edot), so the
# mass matrix already scales the torque with the robot; the gains set the
# closed-loop foot bandwidth, which does not need to change with mass.
Kp = jnp.diag(jnp.tile(jnp.array([500, 500, 500]), n_contact))
Kd = jnp.diag(jnp.tile(jnp.array([20, 20, 20]), n_contact))

# MPC cost matrices, same structure and weights as Aliengo.
Qp = jnp.diag(jnp.array([0, 0, 1e4]))
Qrot = jnp.diag(jnp.array([1e3, 1e3, 0]))
Qdp = jnp.diag(jnp.array([1, 1, 1])) * 1e3
Qomega = jnp.diag(jnp.array([1, 1, 1])) * 1e1
Qgrf = jnp.diag(jnp.ones(3 * n_contact)) * 1e-2

W = jax.scipy.linalg.block_diag(Qp, Qrot, Qdp, Qomega, Qgrf)
