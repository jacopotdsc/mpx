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

# Time and stage parameters
#dt = 0.002  # Time step in seconds
dt_mpc = 0.002
N = 250        # Number of stages
T_TRAJECTORY = 60
mpc_frequency = 100  # Frequency of MPC updates in Hz
grav = 9.81
whole_body_frequency = 500
dt_ref = 1.0 / whole_body_frequency
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
base_body_name  = 'base_link'
wheel_radius    = 0.0925
Kp_motion = 5e1 
Kd_motion = 3e1
Kp_wheel  = 5e1
Kd_wheel  = 3e1 
Kp_reg    = 1e2
Kd_reg    = 2e1
# NOTE: was 1e-1. The C++ baseline's joint-posture-regulation WBC task has
# weight_regulation = 0.0 both in getDefaultParams() and after the
# WalkingManager::init override (TITA_MJ/src/WholeBodyController.cpp /
# WalkingManager.cpp) -- it is fully dead in the reference controller. A
# nonzero posture task here pulls the legs toward the fixed q0 pose and
# competes with the CoM/wheel/base tasks specifically when both a forward
# velocity and a yaw-rate command are active simultaneously (the leg
# configuration needed to satisfy both differs from q0). Zeroed to match
# the real C++ runtime behavior.
w_posture = 0.0

w_qddot     = 1e-12
w_com       = 1e0
w_lwheel    = 1e0
w_rwheel    = 1e0
w_base      = 1e-2

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
# NOTE: tried raising w_v to 15 and 40 to fix the residual forward-speed
# undershoot under the combined vx=0.6/omega=0.4 command (see fix log in
# AUDIT_MPX_CONTROLLER.md). Both reintroduced the same NaN-WBC-torque
# fall that mu/posture/h_fz/fz_min fixed for the single-axis and milder
# combined cases -- even +50% (15) fell within ~4s. This operating point
# is right at a stability boundary that's sensitive to small increases in
# forward-velocity-tracking aggressiveness while turning; left at the
# stable value (10). The remaining combined-command undershoot looks like
# it needs a structural fix (WBC/QP feasibility margin, more than 1 FDDP
# iteration under a stiffer cost, or better constraint softening), not
# further cost-weight tuning -- see AUDIT_MPX_CONTROLLER.md.

# Control weights – ruote (attuatori principali, non troppo economici)
w_a      = 1e-1      # accelerazione lineare
w_ac_z   = 1e-1      # accelerazione verticale
w_alpha  = 1e-3      # accelerazione angolare

# Control weights – GRF (gambe = supporto verticale, non locomozione)
w_fcxy   = 1e-7      # forze orizzontali → penalizza, devono stare ~0
w_fcz    = 1e-4     # forza verticale  → libera di adattarsi

# Equality constraints
w_eq     = 1e8    # momento + contatto


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