import jax.numpy as jnp
import jax 
import os 
import sys 
dir_path = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.abspath(os.path.join(dir_path, '..')) + '/data/tita/tita_world.xml'  # Path to the MuJoCo model XML file
# Joint names and related configuration
joints_name = [
    'joint_left_leg_1', 'joint_left_leg_2', 'joint_left_leg_3', 'joint_left_leg_4',
    'joint_right_leg_1', 'joint_right_leg_2', 'joint_right_leg_3', 'joint_right_leg_4',
]

# Contact frame names and body names for feet (or calves)
contact_frame = ['left_leg_4_collision', 'right_leg_4_collision',]
body_name = ['left_leg_4', 'right_leg_4']

# Time and stage parameters
dt = 0.002  # Time step in seconds
dt_mpc = 0.02
N = 50        # Number of stages
mpc_frequency = 500  # Frequency of MPC updates in Hz
grav = 9.81
whole_body_frequency = 500
# Timer values (make sure the values match your intended configuration)
timer_t = jnp.array([0.5, 0.0, 0.0, 0.5])  # Timer values for each leg
duty_factor = 0.65  # Duty factor for the gait
step_freq = 1.35   # Step frequency in Hz
step_height = 0.065  # Step height in meters
robot_height = 0.44  # Height of the robot's base in meters
clearence_speed = 0.4
mu = 0.6  # Coefficient of friction
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
d = 0.567  # half-wheelbase (m)
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
Kp_motion = 3e1 
Kd_motion = 8e0 
Kp_wheel  = 8e1
Kd_wheel  = 1e1 
Kp_reg    = 5e0 
Kd_reg    = 1e0 

w_qddot     = 1e-4
w_com       = 1e1
w_lwheel    = 5e1
w_rwheel    = 5e1
w_base      = 5e0
w_friction  = 1e2
w_joint_vel = 1e-2

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
w_pcomxy = 1e1      # posizione xy (0 = non tracki, segui il ref generator)
w_pcomz  = 1e3     # altezza CoM
w_vcomxy = 1e2      # velocità xy CoM
w_vcomz  = 1e1      # velocità z CoM
w_c      = 1e2      # posizione contact point
w_vcz    = 1e1      # velocità verticale contact point
w_theta  = 1e2      # heading
w_v      = 1e1      # velocità longitudinale
w_omega  = 1e1      # velocità angolare

# Control weights – ruote (attuatori principali, non troppo economici)
w_a      = 1e-1      # accelerazione lineare
w_ac_z   = 1e-1      # accelerazione verticale
w_alpha  = 1e-1      # accelerazione angolare

# Control weights – GRF (gambe = supporto verticale, non locomozione)
w_fcxy   = 1e-2      # forze orizzontali → penalizza, devono stare ~0
w_fcz    = 1e-3     # forza verticale  → libera di adattarsi

# Equality constraints
w_eq     = 1e3     # momento + contatto

# ── Assemble W (15×15 diagonal) ──────────────────────────────────
W = jnp.diag(jnp.array([
    w_pcomxy, w_pcomz,
    w_vcomxy, w_vcomz,
    w_c,      w_vcz,
    w_theta,  w_v,    w_omega,
    w_a,      w_ac_z, w_alpha,
    w_fcxy,   w_fcz,
    w_eq,
]))