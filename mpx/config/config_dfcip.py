import jax.numpy as jnp
import jax 
import os 
import sys 
dir_path = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.abspath(os.path.join(dir_path, '..')) + '/data/tita/tita.xml'  # Path to the MuJoCo model XML file
#model_path = "/home/jacopo/miniconda3/envs/mjpl/lib/python3.11/site-packages/mujoco_playground/mujoco_playground/_src/locomotion/tita/xmls/tita.xml"

# Joint names and related configuration
joints_name = [
    'joint_left_leg_1', 'joint_left_leg_2', 'joint_left_leg_3', 'joint_left_leg_4',
    'joint_right_leg_1', 'joint_right_leg_2', 'joint_right_leg_3', 'joint_right_leg_4',
]

# Contact frame names and body names for feet (or calves)
contact_frame = ['left_leg_4_collision', 'right_leg_4_collision',]
body_name = ['left_leg_4', 'right_leg_4']

# ── Timing ───────────────────────────────────────────────────────────────
# Three independent rates. MuJoCo integrates at simulation_frequency; the MPC
# and the whole-body QP are updated at mpc_frequency / whole_body_frequency and
# the resulting torque / joint accelerations are held between updates
# (simulation_frequency // whole_body_frequency simulation steps).
# The derived quantities and all consistency checks live in
# mpx.utils.timing.derive_timing (called at the end of this file).
simulation_frequency = 500   # Hz  -> dt_sim = 0.002 s (must equal the XML timestep)
whole_body_frequency = 100   # Hz  -> dt_wbc = 0.01 s (WBC update period)
mpc_frequency = 100          # Hz  -> MPC update period 0.01 s
dt_mpc = 0.01                # s   MPC prediction step
N = 50                       # MPC stages -> horizon N * dt_mpc = 0.5 s
mpc_horizon_s = 0.5          # s   required prediction horizon (checked)
# FDDP iterations per MPC update (real-time iteration scheme: 1). Kept as an
# explicit parameter for ablations; the combined-command fix does not need > 1.
mpc_iterations = 1
# Lookahead [s] used to build the WBC position/velocity references from the
# measured state: pos_ref = p + dt*v, vel_ref = v + dt*a_mpc. The C++ baseline
# uses its 2 ms control period here, which turns the WBC task PD into a
# Kp*dt*v positive velocity feedback (bias) whose gain scales with the WBC
# period: at dt = dt_wbc = 0.01 s it destabilises the yaw (measured omega
# 5 rad/s for a 0.8 rad/s command). With 0 the references are evaluated at
# the current instant, the CoM/wheel task PD terms vanish identically and the
# WBC tracks the MPC accelerations as a pure feedforward inverse-dynamics
# stage (feedback is closed by the 100 Hz MPC). Validated: omega tracking
# error 0.8 -> 0.800 (0.0) vs 0.825 (0.002) vs 5.0 (0.01).
wbc_lookahead_dt = 0.0
T_TRAJECTORY = 60
grav = 9.81
# Timer values (make sure the values match your intended configuration)
timer_t = jnp.array([0.5, 0.0, 0.0, 0.5])  # Timer values for each leg
duty_factor = 0.65  # Duty factor for the gait
step_freq = 1.35   # Step frequency in Hz
step_height = 0.065  # Step height in meters
robot_height = 0.44  # Height of the robot's base in meters
com_z_to_track = 0.4
clearence_speed = 0.4
# NOTE: was 0.6. The C++ baseline (WholeBodyController's WBC friction-cone
# constraint) is initialized to 0.5 in getDefaultParams() but overridden to
# 0.9 at runtime in WalkingManager::init (TITA_MJ/src/WalkingManager.cpp).
# 0.6 gave the WBC QP a much tighter friction budget than the real
# controller, which starves the differential wheel forces needed for
# combined vx+omega commands and was a primary cause of instability under
# combined commands. Matched to the real C++ runtime value.
mu = 0.9  # Coefficient of friction (matches C++ runtime WalkingManager::init override)
use_terrain_estimator = False  # Whether to use state estimation
# Initial positions, orientations, and joint angles
p0 = jnp.array([0, 0, robot_height])  # Initial position of the robot's base
p0_c = jnp.array([0, 0, 0])  # Initial position of the robot's center of mass
quat0 = jnp.array([1, 0, 0, 0])  # Initial orientation of the robot's base (quaternion)
#alingo
q0      = jnp.array([0.0, 0.5, -1.0, 0.0, 0.0, 0.5, -1.0, 0.0])  # Initial joint angles
q0_init = jnp.array([0.0, 0.5, -1.0, 0.0, 0.0, 0.5, -1.0, 0.0])

#alingo
p_legs0 = jnp.array([
    0.27092872, 0.193   , .0,  # Initial position of the front left leg
    0.27092872, -0.193, .0, # Initial position of the front right leg
   -0.20887128, 0.193, .0,  # Initial position of the rear left leg
   -0.20887128, -0.193  , .0   # Initial position of the rear right leg
])

# Determine number of joints and contacts from the lists
n_joints = len(joints_name)  # Number of joints
n_contact = len(contact_frame)  # Number of contact points
nx =  13  # Number of states (pcom, dpcom, c, vcz, theta, v, omega)
nu = 9 # Number of controls (F)
mass = 27.6898
d = 0.567  # leg distance (m)
inertia = jnp.array([[ 1.1446753452439213,     -0.00002628867924503336, -0.024093265357648108],
                        [-0.00002628867924503336,  0.5535098529263547,     -0.00002104211459585719],
                        [-0.024093265357648108,   -0.00002104211459585719,  0.7015150712341189    ]])

# Reference torques and controls (using n_joints)
grf_ref = jnp.array(n_contact*[0.0, 0.0, (grav*mass)/2.0])  # Reference torques (all zeros)
u_ref = jnp.concatenate([jnp.array([0.0, 0.0, 0.0]), jnp.array(grf_ref)])  # Reference controls (concatenated torques)

Kp = jnp.diag(jnp.tile(jnp.array([500,500,500]),n_contact))
Kd = jnp.diag(jnp.tile(jnp.array([20,20,20]),n_contact))

# Whole-body controller
# ── which WBC closes the loop on the MPC plan ─────────────────────────────
#   "qp"          : mpc_utils.whole_body_interface_wheeled_legged_qp -- the
#                   task hierarchy solved as an inequality constrained QP
#                   (qpax), enforcing friction cones and joint limits.
#   "model_based" : mpx.utils.wbc_model_based -- the same tasks and the same
#                   inverse-dynamics torque map, but qddot comes from a single
#                   linear solve (constrained Jacobian pseudo-inverse), like
#                   the quadruped WBC used with the SRBD model. Cheaper and
#                   always finite, but the inequalities are only measured.
#   "wheeled"     : mpc_utils.whole_body_interface_wheeled -- the quadruped
#                   SRBD controller transposed to two wheels: the MPC contact
#                   forces projected on the joints through the contact
#                   Jacobian. Only the wheel task, no CoM / base / posture.
#
# Default: "model_based". It matches the QP's tracking to three decimals on the
# six commands of validate_dfcip_controller.py and costs ~5x less per call
# (0.29 ms vs 1.59 ms on CPU); "qp" stays available for the cases where the
# friction cones / joint limits have to be enforced rather than just measured.
wbc_type = "qp"
# Options used only when wbc_type == "model_based":
#   enforce the wheel rolling equality through the KKT system (recommended:
#   without it the solution can violate the non-holonomic rolling constraint),
wbc_mb_enforce_rolling = True
#   where the contact forces come from: "dynamics" (recover them from the six
#   unactuated rows given qddot, i.e. the equality the QP enforces) or "mpc"
#   (feed the MPC GRF forward, the literal quadruped choice -- it does not hold
#   here, the robot falls at vx>=0.6 because qddot and the MPC forces together
#   violate the floating-base rows; see mpx/utils/wbc_model_based.py),
wbc_mb_force_source = "dynamics"
#   damping added to the task Hessian to keep the inverse well conditioned.
wbc_mb_damping = 1e-6

base_body_name  = 'base_link'
wheel_radius    = 0.0925
Kp_motion = 5e1 
Kd_motion = 3e1
Kp_wheel  = 5e1
Kd_wheel  = 3e1 
Kp_reg    = 1e2
Kd_reg    = 2e1
# Joint posture regulation task (WBC). Restricted to the two hip-abduction
# joints (posture_joint_ids): they are the kinematic redundancy of the stance
# (track width) and nothing else anchors them -- under the lateral load of a
# turn they drift (measured 19 mrad at v=1, omega=0.8), the contact midpoint
# c shifts 3-5 mm sideways with respect to the CoM and the MPC terminal
# stability constraint pcom(N)=c(N) then bends the base path to chase it
# (steady omega +5%). Regulating all leg joints toward q0 instead fights the
# CoM-height task (q0 corresponds to a 0.396 m CoM, tracked height 0.4 m):
# steady pitch -0.02 rad. Validated at (1.0,0.8): omega 0.807 (abduction
# only, with w_base=1) vs 0.843 (no posture task) vs 0.806 with -0.019 rad
# pitch (all joints, w=1e-2).
w_posture = 0.1
posture_joint_ids = (0, 4)   # joint_left_leg_1, joint_right_leg_1 (indices in the 8 actuated joints)

w_qddot     = 1e-12
w_com       = 1e0
w_lwheel    = 1e0
w_rwheel    = 1e0
# Base-orientation task (roll/pitch to zero, yaw to the axle heading). The C++
# value 1e-2 leaves the base free to roll ~15 mrad into a turn under the
# lateral contact force (CoM 5 mm inside the base point, same omega bias as
# above through the terminal stability constraint). 1.0 keeps roll within
# 2 mrad at v=1, omega=0.8; 0.1 is not sufficient (omega 0.825 vs 0.807).
w_base      = 1e0

# Cost matrices (diagonal matrices created using jnp.diag)
Qp    = jnp.diag(jnp.array([1e2, 1e2, 1e3]))  # Cost matrix for position
Qrot  = jnp.diag(jnp.array([1e2, 1e2, 0]))  # Cost matrix for rotation
# Qq    = jnp.diag(jnp.ones(n_joints)) * 1e-1  # Cost matrix for joint angles
Qdp   = jnp.diag(jnp.array([1, 1, 1])) * 1e3  # Cost matrix for position derivatives
Qomega= jnp.diag(jnp.array([1, 1, 1])) * 1e1  # Cost matrix for angular velocity
# Qdq   = jnp.diag(jnp.ones(n_joints)) * 1e-1  # Cost matrix for joint angle derivatives
# Qtau  = jnp.diag(jnp.ones(n_joints)) * 1e-1  # Cost matrix for torques
Qacc = jnp.diag(jnp.array([1, 1, 1]))  # Cost matrix for accelerations
Qgrf = jnp.diag(jnp.array([1e0, 1e0, 1e0]))  # Cost matrix for
# For the leg contact cost, repeat the unit cost for each contact point.
# Qleg_unit represents the cost per leg contact, and we tile it for each contact.
# Qleg_x = jnp.tile(jnp.array([1e4]),n_contact)  # Unit cost for leg contact
# Qleg_y = jnp.tile(jnp.array([1e4]),n_contact)  # Unit cost for leg contact
# Qleg_z = jnp.tile(jnp.array([1e5]),n_contact)  # Unit cost for leg contact
# Qleg  = jnp.diag(jnp.concatenate([Qleg_x,Qleg_y,Qleg_z]))  # Cost matrix for leg contacts

# Combine all cost matrices into a block diagonal matrix
#W = jax.scipy.linalg.block_diag(Qp, Qrot, Qdp, Qomega, Qacc, Qgrf)

# ── MPC cost weights ──────────────────────────────────────────────
# State weights
#
# NOTE on w_pcomxy/w_pcomz/w_vcomxy/w_vcomz: an earlier pass rebalanced
# these four to match the C++ baseline's nominal DFIPActionModel ctor
# weights (C++: 10 / 100000 / 10 / 1 vs this file's original 0 / 2e4 / 3e2
# / 1e1) on the theory that the JAX port had drifted 30x too aggressive on
# CoM-xy-velocity tracking and 5x too weak on CoM-height tracking. That
# rebalance was tested head-to-head against the original values on the
# combined vx=0.6/omega=0.4 command and made things measurably WORSE: the
# original weights track that combined command stably (no NaN, no fall,
# steady-state ~0.24 m/s / ~0.43 rad/s over a 5s hold), while the
# "C++-matched" weights produced a NaN WBC torque and a fall within ~2-4s.
# Reverted to the original values. The likely explanation is that the C++
# weights were tuned in a controller with its own MPC internal-step /
# call-rate mismatch (see AUDIT_CPP_CONTROLLER.md S2 -- the C++ solver's
# warm-start trajectory advances 5x faster than real elapsed time), so its
# absolute weight magnitudes aren't actually transferable 1:1 to this
# port's clean (non-galloping) timing -- numeric parity with the C++
# source is not the same as behavioral parity here. Verified experimentally
# rather than assumed; see AUDIT_CPP_CONTROLLER.md / AUDIT_MPX_CONTROLLER.md.
w_pcomxy = 0e0      # posizione xy
w_pcomz  = 2e4     # altezza CoM
w_vcomxy = 3e2      # velocità xy CoM
w_vcomz  = 1e1      # velocità z CoM
w_c      = 0e0      # posizione com_ground projection
w_vcz    = 0e0      # velocità com_ground projection
w_theta  = 0e0      # heading
# w_v: C++'s equivalent weight (w_v_k_) is aliased with vc_z's weight and
# is left at 0.0 -- an accidental bug (DFIPActionModel.hpp reuses the same
# member for two different residuals), not an intentional design choice.
# Keeping forward-speed tracking active here (unlike the buggy C++ 0) is
# intentional: it is required to hit the vx tracking targets and does not
# reproduce a bug just for parity's sake.
w_v      = 1e1      # velocità com_ground projection
w_omega  = 5e0      # velocità angolare
# NOTE: raising w_v (15, 40) had been tried against the combined-command
# undershoot and reintroduced the NaN/fall; that was a symptom of the w_eq
# conditioning problem documented above, not of w_v itself. Left at 10.

# Control weights – ruote (attuatori principali, non troppo economici)
w_a      = 1e-1      # accelerazione lineare
w_ac_z   = 1e-1      # accelerazione verticale
w_alpha  = 1e-3      # accelerazione angolare

# Control weights – GRF (gambe = supporto verticale, non locomozione)
w_fcxy   = 1e-7      # forze orizzontali → penalizza, devono stare ~0
w_fcz    = 1e-4     # forza verticale  → libera di adattarsi

# Soft equality constraints (moment balance, flat contact, Fz >= 0, terminal
# CoM-over-base stability) -- a single penalty weight shared by all of them.
#
# NOTE: was 1e8 (the C++ baseline value). With the real-time-iteration scheme
# (one FDDP iteration per 10 ms update) a 1e8 penalty on the bilinear moment
# residual makes the problem so ill-conditioned that the Goldstein line search
# rejects the step whenever the reference rotates and the GRF plan has to
# rotate with it (combined v and omega): the plan then stops converging, the
# multiple-shooting defects grow, the moment residual reaches 0.4-0.9 N m and
# the WBC QP fails (NaN torque). Measured on the mandatory validation set:
#   w_eq = 1e8 : (0.6,0.4) -> 0.51/0.43, (1.0,+-0.8) -> fall, 18-24 rejected steps
#   w_eq = 1e7 : all cases tracked, 1-3 rejected steps, residual 0.2-0.35 N m
#   w_eq = 1e6 : all cases tracked, 0 rejected steps, residual 0.01 N m
#   w_eq = 1e5 : all cases tracked, 0 rejected steps, residual 0.03 N m
# i.e. the larger penalty produces LARGER constraint violations because the
# optimiser cannot converge. 1e6 keeps the violation physically negligible
# (0.01 N m = 0.04 mm lever arm at body weight) while the single iteration
# converges; it is still 5 orders of magnitude above the tracking weights.
w_eq     = 1e6


# ── Assemble W (15×15 diagonal) ──────────────────────────────────
W = 1e0 * jnp.diag(jnp.array([
    w_pcomxy, w_pcomz,
    w_vcomxy, w_vcomz,
    w_c,      w_vcz,
    w_theta,  w_v,    w_omega,
    w_a,      w_ac_z, w_alpha,
    w_fcxy,   w_fcz,
    w_eq,
]))

# ── Derived timing (validated at import, see mpx.utils.timing) ─────────────
import types as _types
from mpx.utils.timing import derive_timing as _derive_timing
_timing = _derive_timing(_types.SimpleNamespace(
    simulation_frequency=simulation_frequency, mpc_frequency=mpc_frequency,
    whole_body_frequency=whole_body_frequency, dt_mpc=dt_mpc, N=N,
    mpc_horizon_s=mpc_horizon_s))
dt_sim = _timing["dt_sim"]                              # 0.002 s
dt_wbc = _timing["dt_wbc"]                              # 0.01 s
mpc_period_sim_steps = _timing["mpc_period_sim_steps"]  # 5
wbc_period_sim_steps = _timing["wbc_period_sim_steps"]  # 5
mpc_shift_nodes = _timing["mpc_shift_nodes"]            # 1 MPC node per update
