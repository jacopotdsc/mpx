import jax
from jax import numpy as jnp
from functools import partial
from mujoco.mjx._src import math
from jax.scipy.spatial.transform import Rotation
import mpx.jax_ocp_solvers.optimizers as optimizers

def timer_run(duty_factor,step_freq, leg_time, dt):
    # Extract relevant fields
    # Update timer
    leg_time = leg_time + dt * step_freq
    leg_time = jnp.where(leg_time > 1, leg_time - 1, leg_time)
    contact = jnp.where(leg_time < duty_factor, 1, 0)

    return contact, leg_time
def terrain_orientation(liftoff_pos,Ryaw):

    # Calculate the vectors between the legs
    vec_front_back = (liftoff_pos[:3] + liftoff_pos[3:6] - liftoff_pos[6:9] - liftoff_pos[9:12])/2
    # vec_left_right = (liftoff_pos[:3] + liftoff_pos[6:9] - liftoff_pos[3:6] - liftoff_pos[9:12])/2
    #DO NOT ADJUST THE ROLL
    vec_left_right = Ryaw@jnp.array([0,1,0])
    # Compute the normal vector to the plane
    normal_vector = jnp.cross(vec_front_back, vec_left_right)

    # Normalize the vectors
    vec_front_back = vec_front_back / math.norm(vec_front_back)
    vec_left_right = vec_left_right / math.norm(vec_left_right)
    normal_vector = normal_vector / math.norm(normal_vector)

    # Create the rotation matrix
    rotation_matrix = Rotation.from_matrix(jnp.stack([vec_front_back, vec_left_right, normal_vector], axis=1))

    # Convert the rotation matrix to a quaternion
    quat = rotation_matrix.as_quat()

    return jnp.roll(quat,1)

@partial(jax.jit, static_argnums=(0,1,2,3,4,5))
def reference_generator(use_terrain_estimator,N,dt,n_joints,n_contact,mass,foot0,q0,t_timer, x, foot, input, duty_factor, step_freq,step_height,liftoff,contact,clearence_speed):
    p = x[:3]
    quat = x[3:7]
    # q = x[7:7+n_joints]
    dp = x[7+n_joints:10+n_joints]
    # omega = x[10+n_joints:13+n_joints]
    # dq = x[13+n_joints:13+2*n_joints]
    yaw = jnp.arctan2(2*(quat[0]*quat[3] + quat[1]*quat[2]), 1 - 2*(quat[2]*quat[2] + quat[3]*quat[3]))
    Ryaw = jnp.array([[jnp.cos(yaw), -jnp.sin(yaw), 0],[jnp.sin(yaw), jnp.cos(yaw), 0],[0, 0, 1]])
    proprio_height = input[6] + jnp.sum(liftoff[2::3])/n_contact
    p = jnp.array([p[0], p[1], proprio_height])
    if use_terrain_estimator:
        quat_ref = jnp.tile(terrain_orientation(liftoff,Ryaw), (N+1, 1))
    else:
        quat_ref = jnp.tile(jnp.array([1, 0, 0, 0]), (N+1, 1))
    q_ref = jnp.tile(q0, (N+1, 1))
    contact_sequence = jnp.zeros(((N+1), n_contact))
    pitch = jnp.arcsin(2 * (quat_ref[0,0] * quat_ref[0,2] - quat_ref[0,3] * quat_ref[0,1]))
    Rpitch = jnp.array([[jnp.cos(pitch), 0, jnp.sin(pitch)], [0, 1, 0], [-jnp.sin(pitch), 0, jnp.cos(pitch)]])
    
    ref_lin_vel = Ryaw@Rpitch@input[:3]
    ref_ang_vel = input[3:6]
    p_ref_x = jnp.arange(N+1) * dt * ref_lin_vel[0] + p[0]
    p_ref_y = jnp.arange(N+1) * dt * ref_lin_vel[1] + p[1]
    p_ref_z = jnp.ones(N+1) * proprio_height
    p_ref = jnp.stack([p_ref_x, p_ref_y, p_ref_z], axis=1)
    dp_ref = jnp.tile(ref_lin_vel, (N+1, 1))
    omega_ref = jnp.tile(ref_ang_vel, (N+1, 1))
    foot_ref = jnp.tile(foot, (N+1, 1))
    foot0_projected = jnp.tile(p, n_contact) + foot0 @ jax.scipy.linalg.block_diag(*([Ryaw] * n_contact)).T
    grf_ref = jnp.zeros((N+1, 3*n_contact))

    #Estimate Early contact
    des_contact, current_timer = timer_run(duty_factor, step_freq, t_timer, dt)
    early_contact = jnp.where(jnp.logical_and(jnp.logical_and(des_contact==0,contact==1),current_timer > 0.5 + 0.5*duty_factor),1,0)    

    def foot_fn(t,carry):

        timer_seq, contact_sequence,new_foot,liftoff_x,liftoff_y,liftoff_z,grf_new = carry

        new_foot_x = new_foot[t-1,::3]
        new_foot_y = new_foot[t-1,1::3]
        new_foot_z = new_foot[t-1,2::3]

        new_contact_sequence, new_t = timer_run(duty_factor, step_freq, timer_seq[t-1,:], dt)

        contact_sequence = contact_sequence.at[t,:].set(new_contact_sequence)
        timer_seq = timer_seq.at[t,:].set(new_t)

        liftoff_x = jnp.where(jnp.logical_and(jnp.logical_not(contact_sequence[t,:]),contact_sequence[t-1,:]),new_foot_x,liftoff_x)
        liftoff_y = jnp.where(jnp.logical_and(jnp.logical_not(contact_sequence[t,:]),contact_sequence[t-1,:]),new_foot_y,liftoff_y)
        liftoff_z = jnp.where(jnp.logical_and(jnp.logical_not(contact_sequence[t,:]),contact_sequence[t-1,:]),new_foot_z,liftoff_z)

        def calc_foothold(direction):
            f1 = 0.5*ref_lin_vel[direction]*duty_factor/step_freq
            f2 = jnp.sqrt(input[6]/9.81)*(dp[direction]-ref_lin_vel[direction])
            f = f1 + f2 + foot0_projected[direction::3]
            return f

        foothold_x = calc_foothold(0)
        foothold_y = calc_foothold(1)

        def cubic_splineXY(current_foot, foothold,initial_velocity,val):
            a0 = current_foot
            a1 = initial_velocity
            a2 = 3*(foothold - current_foot) - 2*initial_velocity
            a3 = initial_velocity - 2*(foothold - current_foot)
            return a0 + a1*val + a2*val**2 + a3*val**3

        def cubic_splineZ(current_foot, foothold, step_height,val):
            
            initial_speed = 0.7

            a = 16*step_height - 8*foothold - 8*current_foot - 2*initial_speed
            b = 5*initial_speed + 14*foothold + 18*current_foot - 32*step_height
            c = 16*step_height - 5*foothold - 11*current_foot - 4*initial_speed
            d = initial_speed
            e = current_foot
            return a*val**4 + b*val**3 + c*val**2 + d*val + e
        
        initial_speed = - ref_lin_vel / (jnp.linalg.norm(ref_lin_vel) + 1e-6) * clearence_speed

        new_foot_x = jnp.where(jnp.logical_or(new_contact_sequence>0,early_contact==1), new_foot[t-1,::3], cubic_splineXY(liftoff_x, foothold_x,initial_speed[0],(new_t-duty_factor)/(1-duty_factor)))
        new_foot_y = jnp.where(jnp.logical_or(new_contact_sequence>0,early_contact==1), new_foot[t-1,1::3], cubic_splineXY(liftoff_y, foothold_y,initial_speed[1],(new_t-duty_factor)/(1-duty_factor)))
        new_foot_z = jnp.where(jnp.logical_or(new_contact_sequence>0,early_contact==1), new_foot[t-1,2::3], cubic_splineZ(liftoff_z,liftoff_z,liftoff_z + step_height,(new_t-duty_factor)/(1-duty_factor)))

        new_foot = new_foot.at[t,::3].set(new_foot_x)
        new_foot = new_foot.at[t,1::3].set(new_foot_y)
        new_foot = new_foot.at[t,2::3].set(new_foot_z)

        grf_new = grf_new.at[t,2::3].set((new_contact_sequence*mass*9.81/(jnp.sum(new_contact_sequence)+1e-5)))

        return (timer_seq, contact_sequence,new_foot,liftoff_x,liftoff_y,liftoff_z,grf_new)

    liftoff_x = liftoff[::3]
    liftoff_y = liftoff[1::3]
    liftoff_z = liftoff[2::3]
    timer_sequence_in = jnp.tile(t_timer, (N+1, 1))
    init_carry = (timer_sequence_in, contact_sequence,foot_ref,liftoff_x,liftoff_y,liftoff_z,grf_ref)
    timer_sequence, contact_sequence,foot_ref, liftoff_x,liftoff_y,liftoff_z,grf_ref = jax.lax.fori_loop(0,N+1,foot_fn, init_carry)

    liftoff = liftoff.at[::3].set(liftoff_x)
    liftoff = liftoff.at[1::3].set(liftoff_y)
    liftoff = liftoff.at[2::3].set(liftoff_z)

    return jnp.concatenate([p_ref, quat_ref, q_ref, dp_ref, omega_ref, foot_ref, contact_sequence,grf_ref], axis=1),jnp.concatenate([contact_sequence], axis=1), liftoff

@partial(jax.jit, static_argnums=(0,1,2,3))
def reference_generator_srbd(use_terrain_estimator,N,dt,n_contact,mass,foot0,t_timer, x, foot, input, duty_factor, step_freq,step_height,liftoff,contact,clearence_speed):
    p = x[:3]
    quat = x[3:7]
    dp = x[7:10]
    yaw = jnp.arctan2(2*(quat[0]*quat[3] + quat[1]*quat[2]), 1 - 2*(quat[2]*quat[2] + quat[3]*quat[3]))
    Ryaw = jnp.array([[jnp.cos(yaw), -jnp.sin(yaw), 0],[jnp.sin(yaw), jnp.cos(yaw), 0],[0, 0, 1]])
    proprio_height = input[6] + jnp.sum(liftoff[2::3])/n_contact
    p = jnp.array([p[0], p[1], proprio_height])
    if use_terrain_estimator:
        quat_ref = jnp.tile(terrain_orientation(liftoff,Ryaw), (N+1, 1))
    else:
        quat_ref = jnp.tile(jnp.array([1, 0, 0, 0]), (N+1, 1))
    contact_sequence = jnp.zeros(((N+1), n_contact))
    pitch = jnp.arcsin(2 * (quat_ref[0,0] * quat_ref[0,2] - quat_ref[0,3] * quat_ref[0,1]))
    Rpitch = jnp.array([[jnp.cos(pitch), 0, jnp.sin(pitch)], [0, 1, 0], [-jnp.sin(pitch), 0, jnp.cos(pitch)]])
    ref_lin_vel = Ryaw@Rpitch@input[:3]
    ref_ang_vel = input[3:6]
    p_ref_x = jnp.arange(N+1) * dt * ref_lin_vel[0] + p[0]
    p_ref_y = jnp.arange(N+1) * dt * ref_lin_vel[1] + p[1]
    p_ref_z = jnp.ones(N+1) * proprio_height
    p_ref = jnp.stack([p_ref_x, p_ref_y, p_ref_z], axis=1)
    dp_ref = jnp.tile(ref_lin_vel, (N+1, 1))
    omega_ref = jnp.tile(ref_ang_vel, (N+1, 1))
    foot_ref = jnp.tile(foot, (N+1, 1))
    foot_ref_dot = jnp.zeros(((N+1), 3*n_contact))
    foot0_projected = jnp.tile(p, n_contact) + foot0 @ jax.scipy.linalg.block_diag(*([Ryaw] * n_contact)).T
    grf_ref = jnp.zeros((N+1, 3*n_contact))

    #Estimate Early contact
    des_contact, current_timer = timer_run(duty_factor, step_freq, t_timer, dt)
    early_contact = jnp.where(jnp.logical_and(jnp.logical_and(des_contact==0,contact==1),current_timer > 0.5 + 0.5*duty_factor),1,0)
    
    def foot_fn(t,carry):

        new_t, contact_sequence,new_foot,new_foot_dot,liftoff_x,liftoff_y,liftoff_z,grf_new = carry

        new_foot_x = new_foot[t-1,::3]
        new_foot_y = new_foot[t-1,1::3]
        new_foot_z = new_foot[t-1,2::3]

        new_contact_sequence, new_t = timer_run(duty_factor, step_freq, new_t, dt)

        contact_sequence = contact_sequence.at[t,:].set(new_contact_sequence)

        liftoff_x = jnp.where(jnp.logical_and(jnp.logical_not(contact_sequence[t,:]),contact_sequence[t-1,:]),new_foot_x,liftoff_x)
        liftoff_y = jnp.where(jnp.logical_and(jnp.logical_not(contact_sequence[t,:]),contact_sequence[t-1,:]),new_foot_y,liftoff_y)
        liftoff_z = jnp.where(jnp.logical_and(jnp.logical_not(contact_sequence[t,:]),contact_sequence[t-1,:]),new_foot_z,liftoff_z)

        def calc_foothold(direction):
            f1 = 0.5*ref_lin_vel[direction]*duty_factor/step_freq
            f2 = jnp.sqrt(input[6]/9.81)*(dp[direction]-ref_lin_vel[direction])
            f = f1 + f2 + foot0_projected[direction::3]
            return f

        foothold_x = calc_foothold(0)
        foothold_y = calc_foothold(1)

        def cubic_splineXY(current_foot, foothold,initial_velocity,val):
            a0 = current_foot
            a1 = initial_velocity
            a2 = 3*(foothold - current_foot) - 2*initial_velocity
            a3 = initial_velocity - 2*(foothold - current_foot)
            return a0 + a1*val + a2*val**2 + a3*val**3

        def cubic_splineZ(current_foot, foothold, step_height,val):
            
            initial_speed = 0.7

            a = 16*step_height - 8*foothold - 8*current_foot - 2*initial_speed
            b = 5*initial_speed + 14*foothold + 18*current_foot - 32*step_height
            c = 16*step_height - 5*foothold - 11*current_foot - 4*initial_speed
            d = initial_speed
            e = current_foot
            return a*val**4 + b*val**3 + c*val**2 + d*val + e

        def cubic_splineXY_dot(current_foot, foothold,initial_velocity,val):
            a1 = initial_velocity
            a2 = 3*(foothold - current_foot) - 2*initial_velocity
            a3 = initial_velocity - 2*(foothold - current_foot)
            return 2*a2*val + 3*a3*val**2 + a1

        def cubic_splineZ_dot(current_foot, foothold, step_height,val):
            
            initial_speed = 0.7
            a = 16*step_height - 8*foothold - 8*current_foot - 2*initial_speed
            b = 5*initial_speed + 14*foothold + 18*current_foot - 32*step_height
            c = 16*step_height - 5*foothold - 11*current_foot - 4*initial_speed
            d = initial_speed
            return 4*a*val**3 + 3*b*val**2 + 2*c*val + d

        new_foot_x = jnp.where(jnp.logical_or(new_contact_sequence>0,early_contact==1), new_foot[t-1,::3], cubic_splineXY(liftoff_x, foothold_x,initial_speed[0],(new_t-duty_factor)/(1-duty_factor)))
        new_foot_y = jnp.where(jnp.logical_or(new_contact_sequence>0,early_contact==1), new_foot[t-1,1::3], cubic_splineXY(liftoff_y, foothold_y,initial_speed[1],(new_t-duty_factor)/(1-duty_factor)))
        new_foot_z = jnp.where(jnp.logical_or(new_contact_sequence>0,early_contact==1), new_foot[t-1,2::3], cubic_splineZ(liftoff_z,liftoff_z,liftoff_z + step_height,(new_t-duty_factor)/(1-duty_factor)))

        new_foot = new_foot.at[t,::3].set(new_foot_x)
        new_foot = new_foot.at[t,1::3].set(new_foot_y)
        new_foot = new_foot.at[t,2::3].set(new_foot_z)

        new_foot_dot = new_foot_dot.at[t,::3].set(jnp.where(jnp.logical_or(new_contact_sequence>0,early_contact==1), 0, cubic_splineXY_dot(liftoff_x, foothold_x,initial_speed[0],(new_t-duty_factor)/(1-duty_factor))))
        new_foot_dot = new_foot_dot.at[t,1::3].set(jnp.where(jnp.logical_or(new_contact_sequence>0,early_contact==1), 0, cubic_splineXY_dot(liftoff_y, foothold_y,initial_speed[1],(new_t-duty_factor)/(1-duty_factor))))
        new_foot_dot = new_foot_dot.at[t,2::3].set(jnp.where(jnp.logical_or(new_contact_sequence>0,early_contact==1), 0, cubic_splineZ_dot(liftoff_z,liftoff_z,liftoff_z + step_height,(new_t-duty_factor)/(1-duty_factor))))

        # new_foot_ddot = new_foot_ddot.at[t,::3].set(jnp.where(new_contact_sequence>0, 0, cubic_splineXY_ddot(liftoff_x, foothold_x,(new_t-duty_factor)/(1-duty_factor))))
        # new_foot_ddot = new_foot_ddot.at[t,1::3].set(jnp.where(new_contact_sequence>0, 0, cubic_splineXY_ddot(liftoff_y, foothold_y,(new_t-duty_factor)/(1-duty_factor))))
        # new_foot_ddot = new_foot_ddot.at[t,2::3].set(jnp.where(new_contact_sequence>0, 0, cubic_splineZ_ddot(liftoff_z,liftoff_z,liftoff_z + step_height,(new_t-duty_factor)/(1-duty_factor))))

        grf_new = grf_new.at[t,2::3].set((new_contact_sequence*mass*9.81/(jnp.sum(new_contact_sequence)+1e-5)))

        return (new_t, contact_sequence,new_foot,new_foot_dot,liftoff_x,liftoff_y,liftoff_z,grf_new)

    liftoff_x = liftoff[::3]
    liftoff_y = liftoff[1::3]
    liftoff_z = liftoff[2::3]

    initial_speed = - ref_lin_vel / (jnp.linalg.norm(ref_lin_vel) + 1e-6) * clearence_speed

    init_carry = (t_timer, contact_sequence,foot_ref,foot_ref_dot,liftoff_x,liftoff_y,liftoff_z,grf_ref)
    _, contact_sequence,foot_ref,foot_ref_dot, liftoff_x,liftoff_y,liftoff_z,grf_ref = jax.lax.fori_loop(0,N+1,foot_fn, init_carry)

    liftoff = liftoff.at[::3].set(liftoff_x)
    liftoff = liftoff.at[1::3].set(liftoff_y)
    liftoff = liftoff.at[2::3].set(liftoff_z)

    return jnp.concatenate([p_ref, quat_ref, dp_ref, omega_ref,contact_sequence], axis=1),jnp.concatenate([ contact_sequence,foot_ref], axis=1), liftoff , foot_ref_dot

def reference_generator_dfcip_offline(
    vel_lin: float = 0.0,
    vel_ang: float = 0.0,
    vel_z:   float = 0.0,
    pcom     = (0.0, 0.0, 0.4),
    nx:      int = 13,
    nu:      int = 9,
    t_sec:   float = 6,
    dt:      float = 0.002,
    m:       float = 27.68978,
    grav:    float = 9.81,
) -> jax.Array:

        T = t_sec
        dt_ = dt
        N_STEP_ = int(t_sec / dt_)

        x_ref = jnp.full((nx, N_STEP_), fill_value=0.0)
        u_ref = jnp.full((nx, N_STEP_-1), fill_value=0.0)

        T_const = 2 * T/3
        T_acc = (T-T_const)/2

        v_peak = vel_lin
        omega_peak = vel_ang

        a_max = v_peak / T_acc
        alpha_max = omega_peak / T_acc

        vz          = vel_z
        v_contact_z = 0.0
        v           = 0.0
        omega       = 0.0
        theta0      = 0.0

        a           = 0.0
        alpha       = 0.0

        x0          = pcom[0]
        y0          = pcom[1]
        z0          = pcom[2]
        z0_contact  = 0.0

        z_min       = 0.25
        z_max       = 0.50

        x = x0
        y = y0
        z = z0

        theta = theta0

        def scan_step(carry, t_step):

            t = t_step * dt_
            x, y, z, theta, v_peak, omega_peak, vz, z0_contact, v_contact_z = carry
            v_contact_z = 0.0

            td = t - (T_acc + T_const)

            a = jnp.where(
                    t < T_acc,
                    a_max,
                                            jnp.where(
                    t < T_acc + T_const,
                    0.0,
                                            jnp.where(
                    t < T,
                    -a_max,
                    0.0                     )
                                            )
            )

            v = jnp.where(
                    t < T_acc,
                    a_max * t,
                                            jnp.where(
                    t < T_acc + T_const,
                    v_peak,
                                            jnp.where(
                    t < T,
                    v_peak - a_max * td,
                    0.0                     )
                                            )
            )

            alpha = jnp.where(
                    t < T_acc,
                    alpha_max,
                                            jnp.where(
                    t < T_acc + T_const,
                    0.0,
                                            jnp.where(
                    t < T,
                    -alpha_max,
                    0.0                     )
                                            )
            )

            omega = jnp.where(
                    t < T_acc,
                    alpha_max * t,
                                            jnp.where(
                    t < T_acc + T_const,
                    omega_peak,
                                            jnp.where(
                    t < T,
                    omega_peak - alpha_max * td,
                    0.0                     )
                                            )
            )


            vx = v * jnp.cos(theta)
            vy = v * jnp.sin(theta)

            x  = x  + vx  * dt_
            y  = y  + vy  * dt_
            theta = theta + omega * dt_

            z = jnp.clip(z + vz * dt_, z_min, z_max)
            z_contact = z0_contact + v_contact_z * t

            vz = jnp.where(
                z <= z_min,
                0.0,
                jnp.where(
                    z >= z_max,
                    0.0,
                    vz
                )
            )

            #jax.debug.print("---- frame: {i}", i=t)
            #jax.debug.print("x={x} y={y} z={z}", x=x, y=y, z=z)
            #jax.debug.print("vx={vx} vy={vy} vz={vz}", vx=vx, vy=vy, vz=vz)

            x_ref_t = jnp.array([

                # CoM information
                x,    # [0]  com_x
                y,    # [1]  com_y
                z,    # [2]  com_z
                vx,    # [3]  vx
                vy,   # [4]  vy
                vz,   # [5]  vz

                # Midle point of contact points informations
                x,    # [6]  pc_x
                y,    # [7]  pc_y
                z_contact,          # [8]  pc_z
                v_contact_z,        # [9]  v_contact_z

                # Command
                theta,      # [10] theta
                v,          # [11] v
                omega,      # [12] omega
            ])

            u_ref_t = jnp.array([
                0.0,     # a
                0.0,     # ac_z
                0.0,

                0.0,        # fl_x
                0.0,        # fl_y
                m*grav/2,   # fl_z

                0.0,         # fl_x
                0.0,         # fl_y
                m*grav/2     # fl_z
            ])

            carry_new = x, y, z, theta, v_peak, omega_peak, vz, z0_contact, v_contact_z

            return carry_new, (x_ref_t, u_ref_t)

        v_peak = vel_lin
        omega_peak = vel_ang
        vz = vel_z
        z0_contact = 0.0
        v_contact_z = 0.0

        carry = x0, y0, z0, theta, v_peak, omega_peak, vz, z0_contact, v_contact_z

        _, (x_ref_T, u_ref_T) = jax.lax.scan(scan_step, carry, jnp.arange(N_STEP_))

        x_ref = x_ref_T#.T   # (N_STEP_, NX).T = (NX, N_STEP_)
        u_ref = u_ref_T#.T   # (N_STEP_, NU).T = (NX, N_STEP_)

        return x_ref, u_ref

import mujoco
from mujoco import mjx

def _wbc_qp_dynamics(x, u, t, parameter):
    del u, t, parameter
    return x

def _wbc_qp_cost(W, reference, x, u, t):
    del x
    qacc_var = u
    n = u.shape[0]
    packed = reference[0]
    h_flat = packed[: n * n]
    f = packed[n * n : n * n + n]
    H = h_flat.reshape((n, n))
    w_diag = W[t]
    W_reg = jnp.diag(w_diag)
    stage_cost = 0.5 * (qacc_var @ ((H + W_reg) @ qacc_var)) - (f @ qacc_var)
    return state_cost #jnp.where(t == 0, stage_cost, 0.0)

@partial(jax.jit, static_argnums=(0))
def whole_body_interface(model, mjx_model, contact_id, body_id,sim_frequency,Kp,Kd,qpos,qvel,grf,foot_ref,foot_ref_dot,contact):

    mjx_data = mjx.make_data(model)
    # Update the position and velocity in the data object
    mjx_data = mjx_data.replace(qpos=qpos, qvel=qvel)
    # Perform forward kinematics and dynamics computations
    mjx_data = mjx.fwd_position(mjx_model, mjx_data)
    mjx_data = mjx.fwd_velocity(mjx_model, mjx_data)

    # Extract the mass matrix and bias forces
    M = mjx_data.qM
    D = mjx_data.qfrc_bias

    # Get the positions of the contact points on the legs
    FL_leg = mjx_data.geom_xpos[contact_id[0]]
    FR_leg = mjx_data.geom_xpos[contact_id[1]]
    RL_leg = mjx_data.geom_xpos[contact_id[2]]
    RR_leg = mjx_data.geom_xpos[contact_id[3]]

    # Compute the Jacobians for each leg
    # Return geometric and rotational jacobian ( joint space -> cartesian space)
    # Each of them has shape (18, 3) 
    J_FL, _ = mjx.jac(mjx_model, mjx_data, FL_leg, body_id[0])
    J_FR, _ = mjx.jac(mjx_model, mjx_data, FR_leg, body_id[1])
    J_RL, _ = mjx.jac(mjx_model, mjx_data, RL_leg, body_id[2])
    J_RR, _ = mjx.jac(mjx_model, mjx_data, RR_leg, body_id[3])

    # Concatenate the Jacobians into a single matrix
    J = jnp.concatenate([J_FL, J_FR, J_RL, J_RR], axis=1)
    # Concatenate the positions of the legs into a single vector
    current_leg = jnp.concatenate([FL_leg, FR_leg, RL_leg, RR_leg], axis=0)
    current_leg_dot = J.T @ mjx_data.qvel

    '''
    v_task = J @ qvel -> a_task = J @ qacc + J_dot @ qvel
    qacc = J_pinv @ a_task - J_dot @ qvel -> for small movement, J_dot @ qvel can be negletted
    J_FL actually is the mapping from joint space to task space
    so when using J_pinv we have to use the jacobian from task to join space with J_FL^T
    '''
    # definition of a_task
    cartesian_space_action = Kp@(foot_ref-current_leg) + Kd@(foot_ref_dot-current_leg_dot)

    # M @ qacc + n = tau + J.T @ grf -> tau = M @ qacc + n - J.T @ grf
    tau_fb_lin = D[6:] + (M @ jnp.linalg.pinv(J.T) @ (cartesian_space_action))[6:]
    tau_mpc = -(J@grf)[6:]
    tau_PD = (J @ cartesian_space_action)[6:]
    contact_mask = jnp.array([contact[0],contact[0],contact[0],contact[1],contact[1],contact[1],contact[2],contact[2],contact[2],contact[3],contact[3],contact[3]])
    
    # tau in contact use only
    tau = tau_mpc*contact_mask + (1-contact_mask)*(tau_PD + tau_fb_lin) 

    return tau , J

@partial(jax.jit, static_argnums=(0))
def whole_body_interface_qp(
    model,
    mjx_model,
    contact_id,
    body_id,
    sim_frequency,
    Kp,
    Kd,
    qpos,
    qvel,
    grf,
    foot_ref,
    foot_ref_dot,
    contact,
):
    del sim_frequency

    mjx_data = mjx.make_data(model)
    mjx_data = mjx_data.replace(qpos=qpos, qvel=qvel)
    mjx_data = mjx.fwd_position(mjx_model, mjx_data)
    mjx_data = mjx.fwd_velocity(mjx_model, mjx_data)

    M = mjx_data.qM
    D = mjx_data.qfrc_bias

    FL_leg = mjx_data.geom_xpos[contact_id[0]]
    FR_leg = mjx_data.geom_xpos[contact_id[1]]
    RL_leg = mjx_data.geom_xpos[contact_id[2]]
    RR_leg = mjx_data.geom_xpos[contact_id[3]]

    J_FL, _ = mjx.jac(mjx_model, mjx_data, FL_leg, body_id[0])
    J_FR, _ = mjx.jac(mjx_model, mjx_data, FR_leg, body_id[1])
    J_RL, _ = mjx.jac(mjx_model, mjx_data, RL_leg, body_id[2])
    J_RR, _ = mjx.jac(mjx_model, mjx_data, RR_leg, body_id[3])

    J = jnp.concatenate([J_FL, J_FR, J_RL, J_RR], axis=1)

    current_leg = jnp.concatenate([FL_leg, FR_leg, RL_leg, RR_leg], axis=0)
    current_leg_dot = J.T @ mjx_data.qvel
    cartesian_space_action = Kp @ (foot_ref - current_leg) + Kd @ (foot_ref_dot - current_leg_dot)

    contact_mask = jnp.array(
        [
            contact[0], contact[0], contact[0],
            contact[1], contact[1], contact[1],
            contact[2], contact[2], contact[2],
            contact[3], contact[3], contact[3],
        ],
        dtype=J.dtype,
    )
    swing_mask = 1.0 - contact_mask
    task_weights = 1e-2 * contact_mask + 1.0 * swing_mask
    W_task = jnp.diag(task_weights)

    A_task = W_task @ J.T
    b_task = W_task @ cartesian_space_action

    n_v = M.shape[0]
    reg = 1e-4
    H = A_task.T @ A_task + reg * jnp.eye(n_v, dtype=J.dtype)
    f = A_task.T @ b_task

    packed = jnp.concatenate([H.reshape(-1), f], axis=0)
    reference = jnp.tile(packed[None, :], (2, 1))
    parameter = jnp.zeros((2, 1), dtype=J.dtype)
    W = 1e-3 * jnp.ones((2, n_v), dtype=J.dtype)

    x0 = jnp.zeros((n_v,), dtype=J.dtype)
    X0 = jnp.zeros((2, n_v), dtype=J.dtype)
    U0 = jnp.zeros((1, n_v), dtype=J.dtype)
    V0 = jnp.zeros((2, n_v), dtype=J.dtype)

    X_sol, U_sol, _ = optimizers.mpc(
        _wbc_qp_cost,
        _wbc_qp_dynamics,
        None,
        True,
        reference,
        parameter,
        W,
        x0,
        X0,
        U0,
        V0,
        num_alpha=7,
    )
    qacc = U_sol[0]

    tau_full = M @ qacc + D - J @ grf
    tau = tau_full[6:]

    return tau, J

@partial(jax.jit, static_argnums=(0))
def whole_body_interface_wheeled_legged_old(model, mjx_model, contact_id, body_id,sim_frequency,Kp,Kd,qpos,qvel,grf):

    mjx_data = mjx.make_data(model)
    # Update the position and velocity in the data object
    mjx_data = mjx_data.replace(qpos=qpos, qvel=qvel)
    # Perform forward kinematics and dynamics computations
    mjx_data = mjx.fwd_position(mjx_model, mjx_data)
    mjx_data = mjx.fwd_velocity(mjx_model, mjx_data)

    # Get the positions of the contact points on the legs
    L_leg = mjx_data.geom_xpos[contact_id[0]]
    R_leg = mjx_data.geom_xpos[contact_id[1]]

    # Compute the Jacobians for each leg
    J_L, _ = mjx.jac(mjx_model, mjx_data, L_leg, body_id[0])
    J_R, _ = mjx.jac(mjx_model, mjx_data, R_leg, body_id[1])

    # Concatenate the Jacobians into a single matrix
    J = jnp.concatenate([J_L, J_R], axis=1)

    tau_mpc = -(J@grf)[6:]

    return tau_mpc, J

def quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return jnp.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])

def quat_error(q_des, q_curr):
    """Orientation error (3,) from two quaternions (w,x,y,z)."""
    q_curr_inv = q_curr.at[1:].set(-q_curr[1:])
    q_err = quat_multiply(q_des, q_curr_inv)
    q_err = jnp.where(q_err[0] < 0, -q_err, q_err)
    return 2.0 * q_err[1:]

def err_rotation(Ra, Rb):
    """Port fedele di Eigen::AngleAxisd — con clip esplicito che Eigen fa implicitamente."""
    Rdiff = Rb.T @ Ra
    cos_angle = jnp.clip((jnp.trace(Rdiff) - 1.0) / 2.0, -1.0, 1.0)
    angle = jnp.arccos(cos_angle)
    skew_vec = jnp.array([
        Rdiff[2, 1] - Rdiff[1, 2],
        Rdiff[0, 2] - Rdiff[2, 0],
        Rdiff[1, 0] - Rdiff[0, 1],
    ]) * 0.5
    axis_norm = jnp.linalg.norm(skew_vec)
    safe_norm = jnp.where(axis_norm < 1e-7, 1.0, axis_norm)
    axis = skew_vec / safe_norm
    err = angle * (Ra @ axis)
    return jnp.where(angle < 1e-7, jnp.zeros(3), err)

def skew(v):
    """Skew-symmetric matrix from (3,) vector."""
    return jnp.array([
        [ 0,   -v[2],  v[1]],
        [ v[2], 0,    -v[0]],
        [-v[1], v[0],  0   ],
    ])

def jac_and_dot(mjx_model, mjx_data, mjx_data_pert, point, point_pert, body_id, eps=1e-7):
    """Jacobiana + derivata temporale. Restituisce Jp, Jr, Jp_dot, Jr_dot (3,nv) ciascuna.
    mjx.jac returns (nv,3); transpose here so callers always get (3,nv)."""
    Jp, Jr = mjx.jac(mjx_model, mjx_data, point, body_id)
    Jp_p, Jr_p = mjx.jac(mjx_model, mjx_data_pert, point_pert, body_id)
    return Jp.T, Jr.T, (Jp_p - Jp).T / eps, (Jr_p - Jr).T / eps

def com_jacobian(mjx_model, mjx_data, nv):
    """J_com (3, nv) — mass-weighted sum of body Jacobians."""
    total_mass = jnp.sum(mjx_model.body_mass[0:])
    nb = mjx_model.body_mass.shape[0]

    def _acc(i, J):
        Jp_i, _ = mjx.jac(mjx_model, mjx_data, mjx_data.subtree_com[i], i)
        #jax.debug.print("body {i} mass {m} Jp_i {Jp_i} J {J}", i=i, m=mjx_model.body_mass[i], Jp_i=Jp_i.shape, Jp_i=Jp_i, J=J)
        return J + mjx_model.body_mass[i] * Jp_i.T

    return jax.lax.fori_loop(1, nb, _acc, jnp.zeros((3, nv))) / total_mass

def compute_virtual_frame(wheel_R):
    I = jnp.eye(3)                                     # Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
    z_0 = jnp.array([0.0, 0.0, 1.0])                  # Eigen::Vector3d z_0 = Eigen::Vector3d(0,0,1);
    n = wheel_R @ z_0                                  # Eigen::Vector3d n = wheel_R * z_0;
    a = (I - jnp.outer(n, n)) @ z_0                    # Eigen::Vector3d a = (I - n*n.transpose()) * z_0;
    s = a / jnp.linalg.norm(a)                         # Eigen::Vector3d s = a / a.norm();
    t = jnp.cross(n, s)                                # Eigen::Vector3d t = n.cross(s);
    t = t / jnp.linalg.norm(t)                         # t = t/t.norm();
    R = jnp.column_stack([t, n, s])                    # R.col(0) = t; R.col(1) = n; R.col(2) = s;
    return R                                           # return R;

def compute_contact_frame(wheel_R):
    z_0 = jnp.array([0.0, 0.0, 1.0])                  # Eigen::Vector3d z_0 = Eigen::Vector3d(0,0,1);
    n = wheel_R @ z_0                                  # Eigen::Vector3d n = wheel_R * z_0;
    a = jnp.cross(n, z_0)                              # Eigen::Vector3d a = n.cross(z_0);
    t = a / jnp.linalg.norm(a)                         # Eigen::Vector3d t = a / a.norm();
    R = jnp.column_stack([t, jnp.cross(z_0, t), z_0]) # R.col(0) = t; R.col(1) = z_0.cross(t); R.col(2) = z_0;
    return R                                           # return R;

def get_rCP(wheel_R, wheel_radius):
    I = jnp.eye(3)                                     # Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
    z_0 = jnp.array([0.0, 0.0, 1.0])                  # Eigen::Vector3d z_0 = Eigen::Vector3d(0,0,1);
    n = wheel_R @ z_0                                  # Eigen::Vector3d n = wheel_R * z_0;
    a = (I - jnp.outer(n, n)) @ z_0                    # Eigen::Vector3d a = (I - n*n.transpose()) * z_0;
    s = a / jnp.linalg.norm(a)                         # Eigen::Vector3d s = a/a.norm();
    rCP = -s * wheel_radius                            # Eigen::Vector3d rCP = - s * wheel_radius;
    return rCP                                         # return rCP;

def _wbc_wheeled_qp_dynamics(x, u, t, parameter):
    """
    DFCIP dynamics — Euler, identico al C++.

    State  x = [pcom(3), vcom(3), pl(3), pr(3), dpl(3), dpr(3), theta, v, omega]  (21,)
    Input  u = [a, acz, alpha, fcl(3), fcr(3)]                                     (9,)
    """
    del t

    pcom  = x[0:3]
    vcom  = x[3:6]
    pl    = x[6:9]
    pr    = x[9:12]
    dpl   = x[12:15]
    dpr   = x[15:18]
    theta = x[18]
    v     = x[19]
    w     = x[20]

    a     = u[0]
    acz   = u[1]
    alpha = u[2]
    fcl   = u[3:6]
    fcr   = u[6:9]

    dt   = parameter[0]
    m    = parameter[1]
    grav = parameter[2]
    d    = parameter[3]

    g_vec      = jnp.array([0.0, 0.0, -grav])
    vector_off = jnp.array([0.0, d / 2.0, 0.0])

    dR = jnp.array([
        [-jnp.sin(theta), -jnp.cos(theta), 0.0],
        [ jnp.cos(theta), -jnp.sin(theta), 0.0],
        [ 0.0,             0.0,             0.0],
    ])

    ddR = jnp.array([
        [-jnp.cos(theta),  jnp.sin(theta), 0.0],
        [-jnp.sin(theta), -jnp.cos(theta), 0.0],
        [ 0.0,             0.0,             0.0],
    ])

    ddc = jnp.array([
        a * jnp.cos(theta) - v * jnp.sin(theta) * w,
        a * jnp.sin(theta) + v * jnp.cos(theta) * w,
        acz,
    ])

    # ── COM ──
    acc_com  = (1.0 / m) * (fcl + fcr) + g_vec
    vel_com  = vcom + dt * acc_com
    pos_com  = pcom + dt * vcom

    # ── left wheel ──
    acc_pl   = ddc + (ddR * w**2 + dR * alpha) @ vector_off
    vel_pl   = dpl + dt * acc_pl
    pos_pl   = pl  + dt * dpl

    # ── right wheel ──
    acc_pr   = ddc - (ddR * w**2 + dR * alpha) @ vector_off
    vel_pr   = dpr + dt * acc_pr
    pos_pr   = pr  + dt * dpr

    # ── unicycle ──
    theta_next = theta + dt * w
    v_next     = v     + dt * a
    omega_next = w     + dt * alpha

    return jnp.concatenate([
        pos_com, vel_com,
        pos_pl, pos_pr,
        vel_pl, vel_pr,
        jnp.array([theta_next, v_next, omega_next]),
    ])

def _wbc_wheeled_qp_cost(W, reference, x, u, t):

    n_track = 3
    u_track = 9
    
    position_reference = reference[0:n_track]
    wbc_input_reference = reference[n_track:n_track+u_track]
    constraint_encoding = reference[n_track+u_track:]

    W_pos_track = W[t][:n_track]
    W_input_track = W[t][n_track:n_track+u_track]
    W_constraint = W[t][n_track+u_track:]

    #qacc_var = u
    #n = u.shape[0]
    #packed = reference[0]
    #h_flat = packed[: n * n]
    #f = packed[n * n : n * n + n]
    #H = h_flat.reshape((n, n))
    #w_diag = W[t]
    #W_reg = jnp.diag(w_diag)

    cost_pos_track   = W_pos_track @ (position_reference - x[:n_track])**2
    cost_input_track = W_input_track @ (wbc_input_reference - u)**2
    cost_constraint  = W_constraint @ constraint_encoding**2

    stage_cost = 0.5 * (qacc_var @ ((H + W_reg) @ qacc_var)) - (f @ qacc_var)


    return jnp.where(t == 0, stage_cost, 0.0)

_REF_COM_POS   = 0
_REF_COM_VEL   = 3
_REF_COM_ACC   = 6
_REF_LW_POS    = 9
_REF_LW_VEL    = 12
_REF_LW_ACC    = 15
_REF_RW_POS    = 18
_REF_RW_VEL    = 21
_REF_RW_ACC    = 24
_REF_BASE_ROT = 27
_REF_BASE_OMG  = 36
_REF_BASE_ALP  = 39
_REF_JOINTS    = 42  # da qui: qjnt | qjntdot | qjntddot (3*nj)


def pack_reference(desired, nj):
    """Dict 'desired' → flat jnp array."""
    return jnp.concatenate([
        desired['com_pos'],       # 3
        desired['com_vel'],       # 3
        desired['com_acc'],       # 3
        desired['lwheel_pos'],    # 3
        desired['lwheel_vel'],    # 3
        desired['lwheel_acc'],    # 3
        desired['rwheel_pos'],    # 3
        desired['rwheel_vel'],    # 3
        desired['rwheel_acc'],    # 3
        desired['base_rot'],     # 9
        desired['base_omega'],    # 3
        desired['base_alpha'],    # 3
        desired['qjnt'],          # nj
        desired['qjntdot'],       # nj
        desired['qjntddot'],     # nj
    ])

def unpack_reference(ref, nj):
    """Flat array → dict di desired."""
    j = _REF_JOINTS
    return dict(
        com_pos    = ref[_REF_COM_POS   : _REF_COM_VEL],
        com_vel    = ref[_REF_COM_VEL   : _REF_COM_ACC],
        com_acc    = ref[_REF_COM_ACC   : _REF_LW_POS],
        lwheel_pos = ref[_REF_LW_POS   : _REF_LW_VEL],
        lwheel_vel = ref[_REF_LW_VEL   : _REF_LW_ACC],
        lwheel_acc = ref[_REF_LW_ACC   : _REF_RW_POS],
        rwheel_pos = ref[_REF_RW_POS   : _REF_RW_VEL],
        rwheel_vel = ref[_REF_RW_VEL   : _REF_RW_ACC],
        rwheel_acc = ref[_REF_RW_ACC   : _REF_BASE_ROT],
        base_rot   = ref[_REF_BASE_ROT: _REF_BASE_OMG].reshape((3, 3)),
        base_omega = ref[_REF_BASE_OMG : _REF_BASE_ALP],
        base_alpha = ref[_REF_BASE_ALP : j],
        qjnt       = ref[j          : j + nj],
        qjntdot    = ref[j + nj     : j + 2*nj],
        qjntddot   = ref[j + 2*nj   : j + 3*nj],
    )

def make_wbc_cost(
    mjx_model,
    contact_id,       # [left_wheel_geom_id, right_wheel_geom_id]
    body_id,          # [left_wheel_body_id, right_wheel_body_id]
    base_body_id,
    wheel_radius,     # wheel_radius_ = 0.0925  in C++
    n_contacts_,      # n_contacts_ = 1          in C++
    # ── gains (scalar or (3,3) diagonal) ──────────────
    Kp_motion, Kd_motion,     # 13000, 300
    Kp_wheel,  Kd_wheel,      # 90000, 200
    Kp_reg,    Kd_reg,        # 1000, 50
    # ── task weights ──────────────────────────────────
    weight_q_ddot,            # 1e-4
    weight_com,               # 1.0
    weight_lwheel,            # 2.0
    weight_rwheel,            # 2.0
    weight_base,              # 0.1
):
    """
    Returns cost_fn(W, reference, x, u, t) → scalar.
 
    Decision vector  u = [q_ddot (nv) | fl (3*n_contacts_) | fr (3*n_contacts_)]
    dimensions identical to the C++ QP variable layout:
        n_wbc_variables_ = (6 + n_joints_) + 2 * 3 * n_contacts_
    """
 
    eps = 1e-7
 
    # ── dimensions (mirror of C++ constructor) ──────────────────────
    nv  = mjx_model.nv                                    # robot_model_.nv
    nq  = mjx_model.nq                                    # robot_model_.nq
    n_joints_ = nv - 6                                    # n_joints_ = robot_model_.nv - 6
    n_wbc_variables_    = 6 + n_joints_ + 2 * 3 * n_contacts_
    n_wbc_equalities_   = 6 + 2 * 3 + 2 * 3 * n_contacts_
    n_wbc_inequalities_ = 2 * 4 * n_contacts_ + 2 * n_joints_
 
    # ================================================================
    def cost_fn(W, reference, x, u, t):
        # ── unpack state ──────────────────────────────────────────
        qpos = x[:nq]
        qvel = x[nq : nq + nv]
 
        # ── unpack decision variables ─────────────────────────────
        # u = [q_ddot(nv) | fl(3*n_contacts_) | fr(3*n_contacts_)]
        q_ddot = u[:nv]
        fl    = u[nv : nv + 3 * n_contacts_]
        fr    = u[nv + 3 * n_contacts_ : nv + 3 * n_contacts_ * 2]
 
        # ── unpack desired (from reference) ───────────────────────
        des = unpack_reference(reference[t], n_joints_)
 
        # ══════════════════════════════════════════════════════════
        #  Forward kinematics
        #  C++: pinocchio::framesForwardKinematics(…)
        #       pinocchio::computeJointJacobiansTimeVariation(…)
        # ══════════════════════════════════════════════════════════
        mjx_data = mjx.make_data(mjx_model)
        mjx_data = mjx_data.replace(qpos=qpos, qvel=qvel)
        mjx_data = mjx.fwd_position(mjx_model, mjx_data)
        mjx_data = mjx.fwd_velocity(mjx_model, mjx_data)

        # perturbed state for J̇  (finite-diff of Jacobians)
        q = qpos[3:7]          # quaternione corrente (w, x, y, z)
        omega = qvel[3:6]      # velocità angolare

        # dq/dt = 0.5 * Q(q) * omega  (formula standard)
        dqw = 0.5 * (-q[1]*omega[0] - q[2]*omega[1] - q[3]*omega[2])
        dqx = 0.5 * ( q[0]*omega[0] + q[2]*omega[2] - q[3]*omega[1])
        dqy = 0.5 * ( q[0]*omega[1] - q[1]*omega[2] + q[3]*omega[0])
        dqz = 0.5 * ( q[0]*omega[2] + q[1]*omega[1] - q[2]*omega[0])

        dqpos = jnp.concatenate([
            qvel[:3],                                    # dx, dy, dz
            jnp.array([dqw, dqx, dqy, dqz]),            # dq (quaternione)
            qvel[6:],                                    # d_joints
        ])
        mjx_data_pert = mjx_data.replace(qpos=qpos + eps * dqpos)
        mjx_data_pert = mjx.fwd_position(mjx_model, mjx_data_pert)

        # ══════════════════════════════════════════════════════════
        #  Jacobians + J̇
        #  C++: getFrameJacobian / getFrameJacobianTimeVariation
        # ══════════════════════════════════════════════════════════
        # left wheel  (left_leg4_idx_ in C++)
        J_left_wheel_lin, J_left_wheel_rot, \
        J_left_wheel_dot_lin, J_left_wheel_dot_rot = jac_and_dot(
            mjx_model, mjx_data, mjx_data_pert,
            mjx_data.geom_xpos[contact_id[0]],
            mjx_data_pert.geom_xpos[contact_id[0]],
            body_id[0], eps,
        )
        # right wheel (right_leg4_idx_ in C++)
        J_right_wheel_lin, J_right_wheel_rot, \
        J_right_wheel_dot_lin, J_right_wheel_dot_rot = jac_and_dot(
            mjx_model, mjx_data, mjx_data_pert,
            mjx_data.geom_xpos[contact_id[1]],
            mjx_data_pert.geom_xpos[contact_id[1]],
            body_id[1], eps,
        )
        # base link  (base_link_idx_ in C++)
        _, J_base_link_rot, _, J_base_link_dot_rot = jac_and_dot(
            mjx_model, mjx_data, mjx_data_pert,
            mjx_data.xpos[base_body_id],
            mjx_data_pert.xpos[base_body_id],
            base_body_id, eps,
        )

        #jax.debug.print("---\nJ_left_wheel_lin {Jlw}\n J_left_wheel_dot_lin {Jlw_dot}", Jlw=J_left_wheel_lin, Jlw_dot=J_left_wheel_dot_lin)
        #jax.debug.print("J_right_wheel_lin {Jrw}\n J_right_wheel_dot_lin {Jrw_dot}", Jrw=J_right_wheel_lin, Jrw_dot=J_right_wheel_dot_lin)
        #jax.debug.print("J_base_link_rot {Jbl}\n J_base_link_dot_rot {Jbl_dot}", Jbl=J_base_link_rot, Jbl_dot=J_base_link_dot_rot)

        # COM Jacobian  (pinocchio::jacobianCenterOfMass)
        J_com      = com_jacobian(mjx_model, mjx_data, nv)
        J_com_pert = com_jacobian(mjx_model, mjx_data_pert, nv)
        J_com_dot  = (J_com_pert - J_com) / eps

        # ══════════════════════════════════════════════════════════
        #  Current positions / velocities
        # ══════════════════════════════════════════════════════════
        current_com_pos = mjx_data.subtree_com[0]         # robot_data_.com[0]
        current_com_vel = J_com @ qvel                    # robot_data_.vcom[0]

        l_wheel_center = mjx_data.geom_xpos[contact_id[0]]
        r_wheel_center = mjx_data.geom_xpos[contact_id[1]]
        current_lwheel_vel = J_left_wheel_lin @ qvel      # J_left_wheel_.topRows(3) * qdot
        current_rwheel_vel = J_right_wheel_lin @ qvel     # J_right_wheel_.topRows(3) * qdot

        current_base_quat  = mjx_data.xquat[base_body_id]
        current_base_quat_xyzw = jnp.array([
            current_base_quat[1],
            current_base_quat[2],
            current_base_quat[3],
            current_base_quat[0],
        ])
        current_base_rot = jax.scipy.spatial.transform.Rotation.from_quat(
            current_base_quat_xyzw
        ).as_matrix()
        current_base_link_vel = J_base_link_rot @ qvel    # J_base_link_.bottomRows<3>() * qdot

        jax.debug.print("current_base_quat {q}", q=current_base_quat)
        jax.debug.print("current_base_quat_xyzw {q}", q=current_base_quat_xyzw)
        jax.debug.print("J_base_link_rot {J}", J=J_base_link_rot)
        jax.debug.print("qvel {qvel}", qvel=qvel)

        # ══════════════════════════════════════════════════════════
        #  Drift terms  (= J̇ q̇)
        # ══════════════════════════════════════════════════════════
        a_com_drift              = J_com_dot @ qvel               # robot_data_.acom[0]
        a_lwheel_drift           = J_left_wheel_dot_lin @ qvel    # J_left_wheel_dot_.topRows(3) * qdot
        a_rwheel_drift           = J_right_wheel_dot_lin @ qvel   # J_right_wheel_dot_.topRows(3) * qdot
        a_base_orientation_drift = J_base_link_dot_rot @ qvel     # J_base_link_dot_.bottomRows<3>() * qdot

        # ══════════════════════════════════════════════════════════
        #  PD errors → desired accelerations
        # ══════════════════════════════════════════════════════════

        # ── COM ──
        err_com     = des['com_pos'] - current_com_pos
        err_com_vel = des['com_vel'] - current_com_vel
        a_com_total = (des['com_acc']
                       + Kp_motion * err_com
                       + Kd_motion * err_com_vel)
 
        # ── left wheel ──
        err_lwheel     = des['lwheel_pos'] - l_wheel_center
        err_lwheel_vel = des['lwheel_vel'] - current_lwheel_vel
        a_lwheel_total = (des['lwheel_acc']
                          + Kp_wheel * err_lwheel
                          + Kd_wheel * err_lwheel_vel)
 
        # ── right wheel ──
        err_rwheel     = des['rwheel_pos'] - r_wheel_center
        err_rwheel_vel = des['rwheel_vel'] - current_rwheel_vel
        a_rwheel_total = (des['rwheel_acc']
                          + Kp_wheel * err_rwheel
                          + Kd_wheel * err_rwheel_vel)
 
        # ── base orientation ──
        err_base_orientation     = err_rotation(des['base_rot'], current_base_rot) #quat_error(des['base_quat'], current_base_quat)
        err_base_orientation_vel = des['base_omega'] - current_base_link_vel
        a_base_orientation_total = (des['base_alpha']
                                    + Kp_motion * err_base_orientation
                                    + Kd_motion * err_base_orientation_vel)


        #jax.debug.print("err_com {err_com} err_com_vel {err_com_vel} a_com_total {a_com_total}", err_com=err_com, err_com_vel=err_com_vel, a_com_total=a_com_total)
        #jax.debug.print("des_com_pos {des_com_pos} current_com_pos {current_com_pos}", des_com_pos=des['com_pos'], current_com_pos=current_com_pos)
        #jax.debug.print("des_com_vel {des_com_vel} current_com_vel {current_com_vel}", des_com_vel=des['com_vel'], current_com_vel=current_com_vel)
        #jax.debug.print("err_lwheel {err_lwheel} err_lwheel_vel {err_lwheel_vel} a_lwheel_total {a_lwheel_total}", err_lwheel=err_lwheel, err_lwheel_vel=err_lwheel_vel, a_lwheel_total=a_lwheel_total)
        #jax.debug.print("des_lwheel_pos {des_lwheel_pos} current_lwheel_pos {current_lwheel_pos}", des_lwheel_pos=des['lwheel_pos'], current_lwheel_pos=l_wheel_center)
        #jax.debug.print("des_lwheel_vel {des_lwheel_vel} current_lwheel_vel {current_lwheel_vel}", des_lwheel_vel=des['lwheel_vel'], current_lwheel_vel=current_lwheel_vel)
        #jax.debug.print("err_rwheel {err_rwheel} err_rwheel_vel {err_rwheel_vel} a_rwheel_total {a_rwheel_total}", err_rwheel=err_rwheel, err_rwheel_vel=err_rwheel_vel, a_rwheel_total=a_rwheel_total)
        #jax.debug.print("des_rwheel_pos {des_rwheel_pos} current_rwheel_pos {current_rwheel_pos}", des_rwheel_pos=des['rwheel_pos'], current_rwheel_pos=r_wheel_center)
        #jax.debug.print("des_rwheel_vel {des_rwheel_vel} current_rwheel_vel {current_rwheel_vel}", des_rwheel_vel=des['rwheel_vel'], current_rwheel_vel=current_rwheel_vel)
        
        jax.debug.print("err_base_orientation {err_base_orientation} err_base_orientation_vel {err_base_orientation_vel} a_base_orientation_total {a_base_orientation_total}", err_base_orientation=err_base_orientation, err_base_orientation_vel=err_base_orientation_vel, a_base_orientation_total=a_base_orientation_total)
        jax.debug.print("des_base_rot {des_base_rot} current_base_rot {current_base_rot}", des_base_rot=des['base_rot'], current_base_rot=current_base_rot)
        jax.debug.print("des_base_omega {des_base_omega} current_base_omega {current_base_omega}", des_base_omega=des['base_omega'], current_base_omega=current_base_link_vel)


        jax.debug.print("------------------")
        # ══════════════════════════════════════════════════════════
        #  Build cost matrices H_acc, f_acc   (nv × nv), (nv,)
        #  Mirrors the H_acc, f_acc assembly in C++:
        #    H_acc += weight_q_ddot  * I
        #    H_acc += weight_com     * J_com.T @ J_com
        #    H_acc += weight_lwheel  * J_left_wheel_.topRows(3).T  @ J_left_wheel_.topRows(3)
        #    H_acc += weight_rwheel  * J_right_wheel_.topRows(3).T @ J_right_wheel_.topRows(3)
        #    H_acc += weight_base    * J_base_link_.bottomRows(3).T @ J_base_link_.bottomRows(3)
        # ══════════════════════════════════════════════════════════
        H_acc = (
            weight_q_ddot  * jnp.eye(nv)
            + weight_com     * (J_com.T                @ J_com)
            + weight_lwheel  * (J_left_wheel_lin.T     @ J_left_wheel_lin)
            + weight_rwheel  * (J_right_wheel_lin.T    @ J_right_wheel_lin)
            + weight_base    * (J_base_link_rot.T      @ J_base_link_rot)
        )
 
        # f_acc (nv,)
        #   f_acc += weight_com    * J_com.T    @ (a_com_drift    - a_com_total)
        #   f_acc += weight_lwheel * J_lw.T     @ (a_lwheel_drift - a_lwheel_total)
        #   f_acc += weight_rwheel * J_rw.T     @ (a_rwheel_drift - a_rwheel_total)
        #   f_acc += weight_base   * J_base.T   @ (a_base_drift   - a_base_total)
        f_acc = (
              weight_com     * J_com.T             @ (a_com_drift              - a_com_total)
            + weight_lwheel  * J_left_wheel_lin.T  @ (a_lwheel_drift           - a_lwheel_total)
            + weight_rwheel  * J_right_wheel_lin.T @ (a_rwheel_drift           - a_rwheel_total)
            + weight_base    * J_base_link_rot.T   @ (a_base_orientation_drift - a_base_orientation_total)
        )
 
        # ══════════════════════════════════════════════════════════
        #  Force regularisation
        #  C++:  H_force_one = 1e-9 * I(3*n_contacts_, 3*n_contacts_)
        #        f_force_one = 0
        # ══════════════════════════════════════════════════════════
        H_force_one = 1e-9 * jnp.eye(3 * n_contacts_)
 
        # ══════════════════════════════════════════════════════════
        #  Quadratic cost:  0.5 uᵀ H u + fᵀ u
        #  H is block-diagonal:
        #    ┌─────────┬─────────────┬─────────────┐
        #    │  H_acc  │      0      │      0      │
        #    ├─────────┼─────────────┼─────────────┤
        #    │    0    │ H_force_one │      0      │
        #    ├─────────┼─────────────┼─────────────┤
        #    │    0    │      0      │ H_force_one │
        #    └─────────┴─────────────┴─────────────┘
        #  f = [f_acc | f_force_one | f_force_one]
        # ══════════════════════════════════════════════════════════
        cost_qddot = 0.5 * q_ddot @ (H_acc @ q_ddot) + f_acc @ q_ddot
        cost_fl    = 0.5 * fl @ (H_force_one @ fl)
        cost_fr    = 0.5 * fr @ (H_force_one @ fr)

        stage_cost = cost_qddot + cost_fl + cost_fr
 
        # ── optional W regularisation from the solver ─────────────
        w_diag = W[t][:n_wbc_variables_]
        stage_cost = stage_cost + 0.5 * jnp.sum(w_diag * u**2)

        # single-step horizon: only t == 0 matters
        return stage_cost #jnp.where(t == 0, stage_cost, 0.0)

    # ================================================================
    return cost_fn

def make_wbc_equality_cost(
    mjx_model,
    contact_id,        # [left_wheel_geom_id, right_wheel_geom_id]
    body_id,           # [left_wheel_body_id, right_wheel_body_id]
    base_body_id,
    wheel_radius_,     # 0.0925
    n_contacts_,       # 1
    # ── penalty weights ──
    w_eq_roll,         # weight for rolling constraint penalty
    w_eq_dyn,          # weight for floating-base dynamics penalty
):
    """
    Returns eq_cost_fn(W, reference, x, u, t) → scalar penalty.
 
    Penalises violation of equality constraints as:
        w_eq_roll * ||res_roll_L||² + w_eq_roll * ||res_roll_R||²
      + w_eq_dyn  * ||res_dyn||²
    """
 
    eps = 1e-7
    nv  = mjx_model.nv
    nq  = mjx_model.nq
    n_joints_           = nv - 6
    n_wbc_variables_    = nv + 2 * 3 * n_contacts_
    n_wbc_equalities_   = 6 + 2 * 3 + 2 * 3 * n_contacts_
 
    # ================================================================
    def eq_cost_fn(W, reference, x, u, t):
 
        # ── unpack state ──────────────────────────────────────────
        qpos = x[:nq]
        qvel = x[nq : nq + nv]
 
        # ── unpack decision variables ─────────────────────────────
        q_ddot = u[:nv]
        fl     = u[nv                    : nv + 3 * n_contacts_]
        fr     = u[nv + 3 * n_contacts_  : nv + 2 * 3 * n_contacts_]
 
        # ══════════════════════════════════════════════════════════
        #  Forward kinematics  (same as cost_fn — can be shared via
        #  custom_vjp in production)
        # ══════════════════════════════════════════════════════════
        mjx_data = mjx.make_data(mjx_model)
        mjx_data = mjx_data.replace(qpos=qpos, qvel=qvel)
        mjx_data = mjx.fwd_position(mjx_model, mjx_data)
        mjx_data = mjx.fwd_velocity(mjx_model, mjx_data)
 
        M = mjx_data.qM                   # mass matrix   (nv, nv)
        c = mjx_data.qfrc_bias #+ mjx_data.qfrc_gravcomp           # bias forces   (nv,)
 
        # perturbed state for J̇
        q = qpos[3:7]          # quaternione corrente (w, x, y, z)
        omega = qvel[3:6]      # velocità angolare

        # dq/dt = 0.5 * Q(q) * omega  (formula standard)
        dqw = 0.5 * (-q[1]*omega[0] - q[2]*omega[1] - q[3]*omega[2])
        dqx = 0.5 * ( q[0]*omega[0] + q[2]*omega[2] - q[3]*omega[1])
        dqy = 0.5 * ( q[0]*omega[1] - q[1]*omega[2] + q[3]*omega[0])
        dqz = 0.5 * ( q[0]*omega[2] + q[1]*omega[1] - q[2]*omega[0])

        dqpos = jnp.concatenate([
            qvel[:3],                                    # dx, dy, dz
            jnp.array([dqw, dqx, dqy, dqz]),            # dq (quaternione)
            qvel[6:],                                    # d_joints
        ])
        #dqpos = jnp.concatenate([qvel[:3], jnp.zeros(1), qvel[3:]])
        mjx_data_pert = mjx_data.replace(qpos=qpos + eps * dqpos)
        mjx_data_pert = mjx.fwd_position(mjx_model, mjx_data_pert)
 
        # ══════════════════════════════════════════════════════════
        #  Jacobians + J̇  (left wheel, right wheel)
        # ══════════════════════════════════════════════════════════
        # C++: J_left_wheel_  = [topRows(3)=Jp_L ; bottomRows(3)=Jr_L]  (6, nv)
        #      J_left_wheel_dot_ idem
        Jp_L, Jr_L, Jp_L_dot, Jr_L_dot = jac_and_dot(
            mjx_model, mjx_data, mjx_data_pert,
            mjx_data.geom_xpos[contact_id[0]],
            mjx_data_pert.geom_xpos[contact_id[0]],
            body_id[0], eps,
        )
 
        Jp_R, Jr_R, Jp_R_dot, Jr_R_dot = jac_and_dot(
            mjx_model, mjx_data, mjx_data_pert,
            mjx_data.geom_xpos[contact_id[1]],
            mjx_data_pert.geom_xpos[contact_id[1]],
            body_id[1], eps,
        )
 
        # full 6×nv Jacobians  (stacked linear + rotational)
        J_left_wheel_  = jnp.vstack([Jp_L, Jr_L])     # (6, nv)
        J_right_wheel_ = jnp.vstack([Jp_R, Jr_R])     # (6, nv)
 
        # ── wheel rotations, angular velocities ───────────────────
        l_wheel_R = mjx_data.xmat[body_id[0]].reshape(3, 3)
        r_wheel_R = mjx_data.xmat[body_id[1]].reshape(3, 3)
 
        # C++: w_l = J_left_wheel_.bottomRows<3>() * qdot
        w_l = Jr_L @ qvel                              # (3,)
        w_r = Jr_R @ qvel                              # (3,)
 
        I3 = jnp.eye(3)
 
        # ── rCP  (Center → Contact Point offset) ─────────────────
        # C++: left_rCP  = get_rCP(l_wheel_R, wheel_radius_)
        left_rCP  = get_rCP(l_wheel_R, wheel_radius_)
        right_rCP = get_rCP(r_wheel_R, wheel_radius_)
 
        # ── T_l, T_r  (contact wrench transform) ─────────────────
        # C++: T_l << I3, pinocchio::skew(pcis_l[0]);   (6, 3)
        T_l = jnp.vstack([I3, skew(left_rCP)])         # (6, 3*n_contacts_)
        T_r = jnp.vstack([I3, skew(right_rCP)])        # (6, 3*n_contacts_)
 
        # ══════════════════════════════════════════════════════════
        #  1. Rolling constraint  (6 rows = 3 left + 3 right)
        #     A_acc @ q_ddot = b_acc
        #
        #  C++:
        #    nl = l_virtual_frame_R.col(1)   // = wheel axis = wheel_R * z_0
        #    wl_virtual = (I3 - nl*nl') * w_l
        #
        #    A_acc.topRows(3) = Jp_L - skew(left_rCP) * Jr_L
        #    b_acc.topRows(3) = (-Jp_L_dot + skew(left_rCP)*Jr_L_dot)*qdot
        #                       - w_l × (wl_virtual × left_rCP)
        # ══════════════════════════════════════════════════════════
 
        # C++: nl = l_virtual_frame_R.col(1) = wheel_R * z_0 = wheel_R[:, 2]
        nl = l_wheel_R[:, 2]
        nr = r_wheel_R[:, 2]
 
        # C++: wl_virtual = (I3 - nl*nl') * w_l
        wl_virtual = (I3 - jnp.outer(nl, nl)) @ w_l
        wr_virtual = (I3 - jnp.outer(nr, nr)) @ w_r
 
        # A_acc rows  (3, nv) each
        # C++: A_acc.topRows(3) = J_left_wheel_.topRows(3)
        #                       - skew(left_rCP) * J_left_wheel_.bottomRows(3)
        A_acc_L = Jp_L - skew(left_rCP)  @ Jr_L
        A_acc_R = Jp_R - skew(right_rCP) @ Jr_R
 
        # residual = A_acc @ q_ddot - b_acc
        #          = A_acc @ q_ddot + (J̇_contact) @ qvel + cross_term
        #
        # C++: b_acc.topRows(3) = (-Jp_L_dot + skew(left_rCP)*Jr_L_dot)*qdot
        #                         - w_l.cross(wl_virtual.cross(left_rCP))
        # So:  residual = A_acc_L @ q_ddot
        #               + (Jp_L_dot - skew(left_rCP) @ Jr_L_dot) @ qvel
        #               + cross(w_l, cross(wl_virtual, left_rCP))
        res_roll_L = (
            A_acc_L @ q_ddot
            + (Jp_L_dot - skew(left_rCP) @ Jr_L_dot) @ qvel
            + jnp.cross(w_l, jnp.cross(wl_virtual, left_rCP))
        )
 
        res_roll_R = (
            A_acc_R @ q_ddot
            + (Jp_R_dot - skew(right_rCP) @ Jr_R_dot) @ qvel
            + jnp.cross(w_r, jnp.cross(wr_virtual, right_rCP))
        )
 
        # ══════════════════════════════════════════════════════════
        #  2. No-contact force zeroing  (2*3*n_contacts_ rows)
        #     Currently disabled in C++ (A_no_contact = 0, b_no_contact = 0)
        # ══════════════════════════════════════════════════════════
        # (omitted — same as C++)
 
        # ══════════════════════════════════════════════════════════
        #  3. Floating-base dynamics constraint  (6 rows)
        #     Mu @ q_ddot + cu - Jlu.T @ T_l @ fl - Jru.T @ T_r @ fr = 0
        #
        #  C++:
        #    Mu  = M.block(0, 0, 6, 6+n_joints_)       // (6, nv)
        #    cu  = c.block(0, 0, 6, 1)                  // (6,)
        #    Jlu = J_left_wheel_.block(0, 0, 6, 6)      // (6, 6)
        #    Jru = J_right_wheel_.block(0, 0, 6, 6)     // (6, 6)
        #    A_dyn << Mu, -Jlu' * T_l, -Jru' * T_r
        #    b_dyn = -cu
        # ══════════════════════════════════════════════════════════
        Mu  = M[:6, :]                                  # (6, nv)
        cu  = c[:6]                                     # (6,)
        Jlu = J_left_wheel_[:, :6]                      # (6, 6)
        Jru = J_right_wheel_[:, :6]                     # (6, 6)
 
        res_dyn = Mu @ q_ddot + cu - Jlu.T @ T_l @ fl - Jru.T @ T_r @ fr
 
        # ══════════════════════════════════════════════════════════
        #  Quadratic penalty
        # ══════════════════════════════════════════════════════════
        penalty = (
              w_eq_roll * jnp.sum(res_roll_L ** 2)
            + w_eq_roll * jnp.sum(res_roll_R ** 2)
            + w_eq_dyn  * jnp.sum(res_dyn ** 2)
        )
 
        return penalty #jnp.where(t == 0, penalty, 0.0)
 
    # ================================================================
    return eq_cost_fn

def make_wbc_friction_cost(
    mjx_model,
    contact_id,        # [left_wheel_geom_id, right_wheel_geom_id]
    body_id,           # [left_wheel_body_id, right_wheel_body_id]
    n_contacts_,       # 1
    mu_,               # params_.mu = 0.5 in C++
    w_friction,        # penalty weight
):
    """
    Returns friction_cost_fn(W, reference, x, u, t) → scalar penalty.
 
    Penalises friction cone violation:
        w_friction * Σ_i max(0, C_force_block @ R_contact.T @ f_i)²
    for each contact point on each wheel.
    """
 
    nv = mjx_model.nv
    nq = mjx_model.nq
 
    # C++: C_force_block << 1,0,-mu,  0,1,-mu,  -1,0,-mu,  0,-1,-mu
    C_force_block = jnp.array([
        [ 1.0,  0.0, -mu_],
        [ 0.0,  1.0, -mu_],
        [-1.0,  0.0, -mu_],
        [ 0.0, -1.0, -mu_],
    ])   # (4, 3)
 
    # ================================================================
    def friction_cost_fn(W, reference, x, u, t):
 
        # ── unpack state & decision variables ─────────────────────
        qpos = x[:nq]
        qvel = x[nq : nq + nv]
 
        fl = u[nv                    : nv + 3 * n_contacts_]
        fr = u[nv + 3 * n_contacts_  : nv + 2 * 3 * n_contacts_]
 
        # ══════════════════════════════════════════════════════════
        #  FK — need wheel_R for contact frame
        # ══════════════════════════════════════════════════════════
        mjx_data = mjx.make_data(mjx_model)
        mjx_data = mjx_data.replace(qpos=qpos, qvel=qvel)
        mjx_data = mjx.fwd_position(mjx_model, mjx_data)
 
        # C++: l_wheel_R = robot_data_.oMf[left_leg4_idx_].rotation()
        l_wheel_R = mjx_data.xmat[body_id[0]].reshape(3, 3)
        r_wheel_R = mjx_data.xmat[body_id[1]].reshape(3, 3)
 
        # C++: l_contact_frame = compute_contact_frame(l_wheel_R)
        l_contact_frame = compute_contact_frame(l_wheel_R)    # (3, 3)
        r_contact_frame = compute_contact_frame(r_wheel_R)    # (3, 3)
 
        # ══════════════════════════════════════════════════════════
        #  Friction cone penalty per wheel
        #
        #  C++:  for i in 0..n_contacts_:
        #          C_force_left.block(4*i, 3*i, 4, 3)
        #              = C_force_block * l_contact_frame.T
        #
        #  QP inequality:  C @ f <= 0   (d_max = 0)
        #  Penalty:        max(0, C_force_block @ R.T @ f_i)²
        # ══════════════════════════════════════════════════════════
        def cone_penalty_one_wheel(forces, contact_frame):
            """
            forces: (3 * n_contacts_,)
            contact_frame: (3, 3)
            """
            total = 0.0
            for i in range(n_contacts_):
                f_i = forces[3 * i : 3 * i + 3]
                # C++: C_force_block * l_contact_frame.transpose() * f_i
                violation = C_force_block @ contact_frame.T @ f_i    # (4,)
                total = total + jnp.sum(jnp.maximum(0.0, violation) ** 2)
            return total
 
        penalty = w_friction * (
            cone_penalty_one_wheel(fl, l_contact_frame)
            + cone_penalty_one_wheel(fr, r_contact_frame)
        )
 
        return penalty #jnp.where(t == 0, penalty, 0.0)
 
    # ================================================================
    return friction_cost_fn

def whole_body_interface_wheeled_legged(
    mjx_model,
    # ── static (via partial) ──
    mass, grav, d,
    contact_id, body_id, base_body_id,
    wheel_radius, sample_time, n_contacts,
    # ── gains ──
    X0_prev, U0_prev, V0_prev,
    Kp_motion, Kd_motion,
    Kp_wheel,  Kd_wheel,
    Kp_reg,    Kd_reg,
    # ── weights ──
    w_qddot, w_com, w_lwheel, w_rwheel, w_base,
    w_eq_roll, w_eq_dyn,
    mu_, w_friction,
    # ── runtime ──
    qpos, qvel, desired,
):
    nq = qpos.shape[0]
    nv = mjx_model.nv
    nj = nv - 6
    n_f = 3 
    n_var = nv + 2 * n_f
    WBC_STEP = 1

    # ── 1. pack state ─────────────────────────────────────────────
    x0 = jnp.concatenate([qpos, qvel])

    # ── 2. pack reference ───────────────────────────────────────────
    # `desired` is already a packed flat array (from pack_reference).
    # For horizon T=WBC_STEP: X,V,reference,W need (T+1) rows; U needs T rows.
    ref_flat = desired
    reference = jnp.tile(ref_flat[None, :], (WBC_STEP + 1, 1))

    # ── 3. build cost function ────────────────────────────────────
    #    Tutto il calcolo pesante (Jacobiane, H, f) vive qui dentro.
    qp_cost = make_wbc_cost(
        mjx_model, contact_id, body_id, base_body_id,
        wheel_radius, 1,
        Kp_motion, Kd_motion, Kp_wheel, Kd_wheel, Kp_reg, Kd_reg,
        w_qddot, w_com, w_lwheel, w_rwheel, w_base,
    )

    eq_cost = make_wbc_equality_cost(
        mjx_model, contact_id, body_id, base_body_id,
        wheel_radius, 1,
        w_eq_roll, w_eq_dyn,
    )

    friction_cost = make_wbc_friction_cost(
        mjx_model, contact_id, body_id,
        1,
        mu_, w_friction,
    )

    # cost totale = QP objective + penalità vincoli
    def total_cost(W, reference, x, u, t):
        #return (qp_cost(W, reference, x, u, t)
        #        + eq_cost(W, reference, x, u, t)
        #        + friction_cost(W, reference, x, u, t)
        #)
    
        def _stage_zero(_):
            return (
                qp_cost(W, reference, x, u, t)
                + eq_cost(W, reference, x, u, t)
                + friction_cost(W, reference, x, u, t)
            )

        def _terminal_stage(_):
            return jnp.array(0.0, dtype=x.dtype)

        return jax.lax.cond(t == 0, _stage_zero, _terminal_stage, operand=None)


    # ── 4. MPC init ───────────────────────────────────────────────
    parameter = jnp.tile(jnp.array([sample_time, mass, grav, d])[None, :], (WBC_STEP + 1, 1))  # [dt, m, grav, d]
    W = 1e-3 * jnp.ones((WBC_STEP + 1, n_var))

    X0 = jnp.tile(x0[None, :], (WBC_STEP + 1, 1))
    U0 = jnp.tile(U0_prev[None, :], (WBC_STEP, 1))
    V0 = jnp.tile(V0_prev[None, :], (WBC_STEP + 1, 1))
    jax.debug.print("x0 shape {x0_shape}, X0 prev {X0_shape} U0 shape {U0_shape} V0 shape {V0_shape}", x0_shape=x0.shape, X0_shape=X0_prev.shape, U0_shape=U0_prev.shape, V0_shape=V0_prev.shape)
    jax.debug.print("reference shape {reference_shape}, X0 shape {X0_shape} U0 shape {U0_shape} V0 shape {V0_shape}", reference_shape=reference.shape, X0_shape=X0.shape, U0_shape=U0.shape, V0_shape=V0.shape)

    # ── 5. solve ──────────────────────────────────────────────────
    X_sol, U_sol, V_sol = optimizers.mpc(
        total_cost,
        _wbc_qp_dynamics,
        None,
        True,
        reference,
        parameter,
        W,
        x0,
        X0,
        U0,
        V0,
        num_alpha=1,
    )

    finite_solution = (
        jnp.all(jnp.isfinite(X_sol))
        & jnp.all(jnp.isfinite(U_sol))
        & jnp.all(jnp.isfinite(V_sol))
    )

    def _use_solver(_):
        return X_sol, U_sol, V_sol

    def _use_warm_start(_):
        return X0, U0, V0

    #X_sol, U_sol, V_sol = jax.lax.cond(finite_solution, _use_solver, _use_warm_start, operand=None)

    sol   = U_sol[0]
    qddot = sol[:nv]
    fl    = sol[nv : nv + n_f]
    fr    = sol[nv + n_f : nv + 2 * n_f]

    # ── 6. Inverse dynamics → tau ─────────────────────────────────
    #    Serve ancora FK per M, c, J (unavoidable per il torque).
    mjx_data = mjx.make_data(mjx_model)
    mjx_data = mjx_data.replace(qpos=qpos, qvel=qvel)
    mjx_data = mjx.fwd_position(mjx_model, mjx_data)

    M = mjx_data.qM
    c = mjx_data.qfrc_bias #+ mjx_data.qfrc_gravcomp

    Jp_L, Jr_L, _, _ = jac_and_dot(
        mjx_model, mjx_data, mjx_data,  # no pert needed
        mjx_data.geom_xpos[contact_id[0]],
        mjx_data.geom_xpos[contact_id[0]],
        body_id[0], eps=1.0)  # eps irrilevante, non usiamo J̇

    Jp_R, Jr_R, _, _ = jac_and_dot(
        mjx_model, mjx_data, mjx_data,
        mjx_data.geom_xpos[contact_id[1]],
        mjx_data.geom_xpos[contact_id[1]],
        body_id[1], eps=1.0)

    l_wheel_R = mjx_data.xmat[body_id[0]].reshape(3, 3)
    r_wheel_R = mjx_data.xmat[body_id[1]].reshape(3, 3)

    I3 = jnp.eye(3)
    T_l = jnp.vstack([I3, skew(get_rCP(l_wheel_R, wheel_radius))])
    T_r = jnp.vstack([I3, skew(get_rCP(r_wheel_R, wheel_radius))])

    J_left_wheel  = jnp.vstack([Jp_L, Jr_L])
    J_right_wheel = jnp.vstack([Jp_R, Jr_R])

    Ma  = M[6:, :]
    ca  = c[6:]
    Jla = J_left_wheel[:, 6:]
    Jra = J_right_wheel[:, 6:]

    tau = Ma @ qddot + ca - Jla.T @ T_l @ fl - Jra.T @ T_r @ fr

    # Warm-start vectors for the next WBC call: keep the next initial guess as
    # 1D vectors so the wrapper preserves the expected per-env shapes.
    X_warm = X_sol[1]
    U_warm = U_sol[1]
    V_warm = V_sol[1]

    return tau, qddot, fl, fr, X_warm, U_warm, V_warm

def whole_body_interface_wheeled_legged_qp(
    mjx_model,
    # ── static (via partial) ──
    mass, grav, d,
    contact_id, body_id, base_body_id,
    wheel_radius, sample_time, n_contacts,
    # ── gains ──
    X0_prev, U0_prev, V0_prev,
    Kp_motion, Kd_motion,
    Kp_wheel,  Kd_wheel,
    Kp_reg,    Kd_reg,
    # ── weights ──
    w_qddot, w_com, w_lwheel, w_rwheel, w_base,
    w_eq_roll, w_eq_dyn,
    mu_, w_friction,
    # ── runtime ──
    qpos, qvel, desired,
):
    nq    = qpos.shape[0]
    nv    = mjx_model.nv
    nj    = nv - 6
    n_f   = 3
    n_var = nv + 2 * n_f

    # ── unpack desired ────────────────────────────────────────────────────────
    des = unpack_reference(desired, nj)

    # ══════════════════════════════════════════════════════════════════════════
    #  1. FORWARD KINEMATICS  (una sola FK, niente data perturbata)
    # ══════════════════════════════════════════════════════════════════════════
    mjx_data = mjx.make_data(mjx_model)
    mjx_data = mjx_data.replace(qpos=qpos, qvel=qvel)
    mjx_data = mjx.fwd_position(mjx_model, mjx_data)
    mjx_data = mjx.fwd_velocity(mjx_model, mjx_data)

    M = mjx_data.qM
    c = mjx_data.qfrc_bias #+ mjx_data.qfrc_gravcomp

    # ── tangente dqpos/dt per jvp ─────────────────────────────────────────────
    q_wxyz = qpos[3:7]
    omega  = qvel[3:6]
    dqw = 0.5 * (-q_wxyz[1]*omega[0] - q_wxyz[2]*omega[1] - q_wxyz[3]*omega[2])
    dqx = 0.5 * ( q_wxyz[0]*omega[0] + q_wxyz[2]*omega[2] - q_wxyz[3]*omega[1])
    dqy = 0.5 * ( q_wxyz[0]*omega[1] - q_wxyz[1]*omega[2] + q_wxyz[3]*omega[0])
    dqz = 0.5 * ( q_wxyz[0]*omega[2] + q_wxyz[1]*omega[1] - q_wxyz[2]*omega[0])
    dqpos = jnp.concatenate([
        qvel[:3],
        jnp.array([dqw, dqx, dqy, dqz]),
        qvel[6:],
    ])

    # ── funzioni per jvp: ritornano (Jp, Jr) in funzione di qpos ─────────────
    def _jac_geom(qpos_, geom_id, bid):
        d_ = mjx.make_data(mjx_model)
        d_ = d_.replace(qpos=qpos_, qvel=qvel)
        d_ = mjx.fwd_position(mjx_model, d_)
        Jp_, Jr_ = mjx.jac(mjx_model, d_, d_.geom_xpos[geom_id], bid)
        return Jp_.T, Jr_.T   # (3, nv), (3, nv)

    def _jac_body(qpos_, bid):
        d_ = mjx.make_data(mjx_model)
        d_ = d_.replace(qpos=qpos_, qvel=qvel)
        d_ = mjx.fwd_position(mjx_model, d_)
        Jp_, Jr_ = mjx.jac(mjx_model, d_, d_.xpos[bid], bid)
        return Jp_.T, Jr_.T

    def _jac_com(qpos_):
        d_ = mjx.make_data(mjx_model)
        d_ = d_.replace(qpos=qpos_, qvel=qvel)
        d_ = mjx.fwd_position(mjx_model, d_)
        total_mass = jnp.sum(mjx_model.body_mass)
        nb = mjx_model.body_mass.shape[0]
        def _acc(i, J):
            Jp_i, _ = mjx.jac(mjx_model, d_, d_.subtree_com[i], i)
            return J + mjx_model.body_mass[i] * Jp_i.T
        return jax.lax.fori_loop(1, nb, _acc, jnp.zeros((3, nv))) / total_mass

    # ── Jacobiane + derivate temporali esatte via jax.jvp ────────────────────
    (J_left_wheel_lin,  J_left_wheel_rot),  \
    (J_left_wheel_dot_lin,  J_left_wheel_dot_rot)  = jax.jvp(
        lambda qp: _jac_geom(qp, contact_id[0], body_id[0]),
        (qpos,), (dqpos,),
    )
    (J_right_wheel_lin, J_right_wheel_rot), \
    (J_right_wheel_dot_lin, J_right_wheel_dot_rot) = jax.jvp(
        lambda qp: _jac_geom(qp, contact_id[1], body_id[1]),
        (qpos,), (dqpos,),
    )
    (_, J_base_link_rot), \
    (_, J_base_link_dot_rot) = jax.jvp(
        lambda qp: _jac_body(qp, base_body_id),
        (qpos,), (dqpos,),
    )
    J_com, J_com_dot = jax.jvp(
        _jac_com,
        (qpos,), (dqpos,),
    )

    # ── posizioni / velocità correnti ─────────────────────────────────────────
    current_com_pos        = mjx_data.subtree_com[0]
    current_com_vel        = J_com @ qvel

    l_wheel_center         = mjx_data.geom_xpos[contact_id[0]]
    r_wheel_center         = mjx_data.geom_xpos[contact_id[1]]
    current_lwheel_vel     = J_left_wheel_lin  @ qvel
    current_rwheel_vel     = J_right_wheel_lin @ qvel

    current_base_quat      = mjx_data.xquat[base_body_id]
    current_base_quat_xyzw = jnp.array([
        current_base_quat[1], current_base_quat[2],
        current_base_quat[3], current_base_quat[0],
    ])
    current_base_quat_xyzw = current_base_quat_xyzw / (
        jnp.linalg.norm(current_base_quat_xyzw) + 1e-9
    )
    current_base_rot       = jax.scipy.spatial.transform.Rotation.from_quat(
        current_base_quat_xyzw
    ).as_matrix()
    current_base_link_vel  = J_base_link_rot @ qvel

    l_wheel_R = mjx_data.xmat[body_id[0]].reshape(3, 3)
    r_wheel_R = mjx_data.xmat[body_id[1]].reshape(3, 3)

    # ══════════════════════════════════════════════════════════════════════════
    #  2. DRIFT TERMS  (J̇ q̇)
    # ══════════════════════════════════════════════════════════════════════════
    a_com_drift              = J_com_dot             @ qvel
    a_lwheel_drift           = J_left_wheel_dot_lin  @ qvel
    a_rwheel_drift           = J_right_wheel_dot_lin @ qvel
    a_base_orientation_drift = J_base_link_dot_rot   @ qvel

    # ══════════════════════════════════════════════════════════════════════════
    #  3. PD ERRORS → DESIRED ACCELERATIONS
    # ══════════════════════════════════════════════════════════════════════════
    err_com              = des['com_pos'] - current_com_pos
    err_com_vel          = des['com_vel'] - current_com_vel
    a_com_total          = des['com_acc'] + Kp_motion * err_com + Kd_motion * err_com_vel

    err_lwheel           = des['lwheel_pos'] - l_wheel_center
    err_lwheel_vel       = des['lwheel_vel'] - current_lwheel_vel
    a_lwheel_total       = des['lwheel_acc'] + Kp_wheel * err_lwheel + Kd_wheel * err_lwheel_vel

    err_rwheel           = des['rwheel_pos'] - r_wheel_center
    err_rwheel_vel       = des['rwheel_vel'] - current_rwheel_vel
    a_rwheel_total       = des['rwheel_acc'] + Kp_wheel * err_rwheel + Kd_wheel * err_rwheel_vel

    err_base_orientation     = err_rotation(des['base_rot'], current_base_rot)
    err_base_orientation_vel = des['base_omega'] - current_base_link_vel
    a_base_orientation_total = (
        des['base_alpha']
        + Kp_motion * err_base_orientation
        + Kd_motion * err_base_orientation_vel
    )

    # ══════════════════════════════════════════════════════════════════════════
    #  4. ASSEMBLA H e f  (n_var x n_var)  e  (n_var,)
    # ══════════════════════════════════════════════════════════════════════════
    H_acc = (
          w_qddot  * jnp.eye(nv)
        + w_com    * (J_com.T            @ J_com)
        + w_lwheel * (J_left_wheel_lin.T  @ J_left_wheel_lin)
        + w_rwheel * (J_right_wheel_lin.T @ J_right_wheel_lin)
        + w_base   * (J_base_link_rot.T   @ J_base_link_rot)
    )
    f_acc = (
          w_com    * J_com.T            @ (a_com_drift              - a_com_total)
        + w_lwheel * J_left_wheel_lin.T  @ (a_lwheel_drift           - a_lwheel_total)
        + w_rwheel * J_right_wheel_lin.T @ (a_rwheel_drift           - a_rwheel_total)
        + w_base   * J_base_link_rot.T   @ (a_base_orientation_drift - a_base_orientation_total)
    )

    # ── force regularisation + friction cone penalty (smooth) ────────────────
    C_force_block = jnp.array([
        [ 1.0,  0.0, -mu_],
        [ 0.0,  1.0, -mu_],
        [-1.0,  0.0, -mu_],
        [ 0.0, -1.0, -mu_],
    ])
    Cl   = C_force_block @ compute_contact_frame(l_wheel_R).T   # (4, 3)
    Cr   = C_force_block @ compute_contact_frame(r_wheel_R).T   # (4, 3)
    H_fl = 1e-9 * jnp.eye(n_f) + w_friction * (Cl.T @ Cl)
    H_fr = 1e-9 * jnp.eye(n_f) + w_friction * (Cr.T @ Cr)

    H_full = jax.scipy.linalg.block_diag(H_acc, H_fl, H_fr)           # (n_var, n_var)
    f_full = jnp.concatenate([f_acc, jnp.zeros(n_f), jnp.zeros(n_f)]) # (n_var,)
    
    # ══════════════════════════════════════════════════════════════════════════
    #  5. VINCOLI DI UGUAGLIANZA  A_eq u = b_eq
    #
    #  righe:  rolling L (3) | rolling R (3) | floating-base dynamics (6)
    # ══════════════════════════════════════════════════════════════════════════
    I3 = jnp.eye(3)

    left_rCP  = get_rCP(l_wheel_R, wheel_radius)
    right_rCP = get_rCP(r_wheel_R, wheel_radius)
    T_l = jnp.vstack([I3, skew(left_rCP)])    # (6, 3)
    T_r = jnp.vstack([I3, skew(right_rCP)])   # (6, 3)

    J_left_wheel_  = jnp.vstack([J_left_wheel_lin,  J_left_wheel_rot])   # (6, nv)
    J_right_wheel_ = jnp.vstack([J_right_wheel_lin, J_right_wheel_rot])  # (6, nv)

    # ── rolling ───────────────────────────────────────────────────────────────
    nl = l_wheel_R[:, 2]
    nr = r_wheel_R[:, 2]
    w_l = J_left_wheel_rot  @ qvel
    w_r = J_right_wheel_rot @ qvel
    wl_virtual = (I3 - jnp.outer(nl, nl)) @ w_l
    wr_virtual = (I3 - jnp.outer(nr, nr)) @ w_r

    A_roll_L = J_left_wheel_lin  - skew(left_rCP)  @ J_left_wheel_rot   # (3, nv)
    A_roll_R = J_right_wheel_lin - skew(right_rCP) @ J_right_wheel_rot  # (3, nv)
    b_roll_L = (
        -(J_left_wheel_dot_lin  - skew(left_rCP)  @ J_left_wheel_dot_rot)  @ qvel
        - jnp.cross(w_l, jnp.cross(wl_virtual, left_rCP))
    )
    b_roll_R = (
        -(J_right_wheel_dot_lin - skew(right_rCP) @ J_right_wheel_dot_rot) @ qvel
        - jnp.cross(w_r, jnp.cross(wr_virtual, right_rCP))
    )

    # ── floating-base dynamics ────────────────────────────────────────────────
    Mu  = M[:6, :]
    cu  = c[:6]
    Jlu = J_left_wheel_[:, :6]    # (6, 6)
    Jru = J_right_wheel_[:, :6]   # (6, 6)
    A_dyn = jnp.hstack([Mu, -(Jlu.T @ T_l), -(Jru.T @ T_r)])   # (6, n_var)
    b_dyn = -cu

    # ── assembla A_eq, b_eq ───────────────────────────────────────────────────
    zeros_f = jnp.zeros((3, 2 * n_f))
    A_eq = jnp.vstack([
        jnp.hstack([A_roll_L, zeros_f]),   # (3, n_var)
        jnp.hstack([A_roll_R, zeros_f]),   # (3, n_var)
        A_dyn,                             # (6, n_var)
    ])                                     # (12, n_var)
    b_eq = jnp.concatenate([b_roll_L, b_roll_R, b_dyn])   # (12,)

    # ══════════════════════════════════════════════════════════════════════════
    #  6. RISOLVI KKT
    #
    #  [ H_full   A_eq' ] [ u      ]   [ -f_full ]
    #  [ A_eq      0    ] [ lambda ] = [  b_eq   ]
    # ══════════════════════════════════════════════════════════════════════════
    n_eq = b_eq.shape[0]    # 12
    KKT  = jnp.block([
        [H_full,                          A_eq.T                    ],
        [A_eq,    jnp.zeros((n_eq, n_eq))],
    ])                                    # (n_var+12, n_var+12)
    rhs  = jnp.concatenate([-f_full, b_eq])
    sol  = jnp.linalg.solve(KKT, rhs)

    qddot = sol[:nv]
    fl    = sol[nv       : nv + n_f]
    fr    = sol[nv + n_f : nv + 2 * n_f]

    # ══════════════════════════════════════════════════════════════════════════
    #  7. INVERSE DYNAMICS → TAU
    # ══════════════════════════════════════════════════════════════════════════
    Ma  = M[6:, :]
    ca  = c[6:]
    Jla = J_left_wheel_[:, 6:]    # (6, nj)
    Jra = J_right_wheel_[:, 6:]   # (6, nj)
    tau = Ma @ qddot + ca - Jla.T @ T_l @ fl - Jra.T @ T_r @ fr

    # warm-start (interfaccia compatibile col wrapper)
    X_warm = X0_prev
    U_warm = sol[:n_var]
    V_warm = V0_prev

    return tau, qddot, fl, fr, X_warm, U_warm, V_warm

@partial(jax.jit, static_argnums=(0,1,2,3))
def reference_barell_roll(N,dt,n_joints,n_contact,foot0,q0):
    t1 = 0.2
    t2 = 0.2
    t3 = 0.3
    t4 = 0.1
    z_start = 0.4
    z_land = 0.28
    v_lateral = -0.25/(t2+t3)
    v0 = (z_land - z_start + 0.5*9.81*t3*t3)/t3 
    total_roll_time = t2+t3+t4
    roll_speed = 2*3.14/total_roll_time
    def z_position(t):
        return z_start - 0.5*9.81*t**2 + v0*t
    def z_speed(t):
        return -9.81*t + v0
    acc = v0/t2
    print("v0", v0)
    print("acc", acc)
    #first part full stance 0.1s
    n1 = int(t1/dt)
    p1 = jnp.tile(jnp.array([0,0,0.33]), (n1, 1))
    p1 = p1.at[:,1].set(jnp.arange(n1)*dt*(v_lateral))
    dp1 = jnp.tile(jnp.array([0,v_lateral,0]), (n1, 1))
    contact1 = jnp.tile(jnp.array([1,1,1,1]), (n1, 1))
    quat1 = jnp.tile(jnp.array([1, 0, 0, 0]), (n1, 1))
    omega1 = jnp.tile(jnp.array([0, 0, 0]), (n1, 1))
    #second part lateral support 0.2s
    n2 = int(t2/dt)
    p2 = jnp.tile(jnp.array([0,p1[-1,1],0.33]), (n2, 1))
    p2 = p2.at[:,2].set(0.5*jnp.arange(n2)*dt*jnp.arange(n2)*dt*acc + 0.33)
    p2 = p2.at[:,1].set(jnp.arange(n2)*dt*(v_lateral))
    dp2 = jnp.tile(jnp.array([0,v_lateral,0]), (n2, 1))
    dp2 = dp2.at[:,2].set(jnp.arange(n2)*dt*acc)
    contact2 = jnp.tile(jnp.array([0,1,0,1]), (n2, 1))
    # for i in range(n2):
    #     p2 = p2.at[i,2].set(z_position(i*dt))
    #     dp2 = dp2.at[i,2].set(z_speed(i*dt))
    #third part flying phase 0.4s
    n3 = int(t3/dt)
    p3 = jnp.tile(jnp.array([0,p2[-1,1],p2[-1,2]]), (n3, 1))
    p3 = p3.at[:,1].set(jnp.arange(n3)*dt*(v_lateral))
    dp3 = jnp.tile(jnp.array([0,v_lateral,0]), (n3, 1))
    for i in range(n3):
        p3 = p3.at[i,2].set(z_position(i*dt))
        dp3 = dp3.at[i,2].set(z_speed(i*dt))
    def fn(t,carry):
        quat_new = math.quat_integrate(carry[t-1,:], jnp.array([roll_speed,0,0]), dt)
        carry_new = carry.at[t,:].set(quat_new)
        return carry_new
    
    
    contact3 = jnp.tile(jnp.array([0,0,0,0]), (n3, 1))
    #fourth part full stance 0.2s
    n4 = int(t4/dt)
    p4 = jnp.tile(jnp.array([0,p3[-1,1],z_land]), (n4, 1))
    dp4 = jnp.tile(jnp.array([0,0,0]), (n4, 1))
    quat5 = jnp.tile(jnp.array([1, 0, 0, 0]), (n4, 1))
    omega5 = jnp.tile(jnp.array([0, 0, 0]), (n4, 1))
    contact4 = jnp.tile(jnp.array([1,1,1,1]), (n4, 1))

    init_carry = jnp.tile(jnp.array([1.0, 0.0, 0, 0]), (n2+n3+n4, 1))
    quat234 = jax.lax.fori_loop(1, n2+n3+n4, fn, init_carry)
    omega234 = jnp.tile(jnp.array([roll_speed, 0, 0]), (n2+n3+n4, 1))

    n5 = N - (n1+n2+n3+n4)

    p5 = jnp.tile(jnp.array([0,p4[-1,1],z_land]), (n5, 1))
    dp5 = jnp.tile(jnp.array([0,0,0]), (n5, 1))
    quat5 = jnp.tile(jnp.array([1, 0, 0, 0]), (n5, 1))
    omega5 = jnp.tile(jnp.array([0, 0, 0]), (n5, 1))
    contact5 = jnp.tile(jnp.array([1,1,1,1]), (n5, 1))

    p_ref = jnp.concatenate([p1, p2, p3, p4,p5], axis=0)
    quat_ref = jnp.concatenate([quat1,quat234,quat5], axis=0)
    q_ref = jnp.tile(q0, (n1+n2+n3+n4+n5, 1))
    dp_ref = jnp.concatenate([dp1, dp2, dp3, dp4,dp5], axis=0)
    omega_ref = jnp.concatenate([omega1,omega234,omega5], axis=0)
    foot_ref = jnp.tile(foot0, (n1+n2+n3+n4+n5, 1)) + jnp.tile(p_ref, n_contact)
    foot_ref = foot_ref.at[:,2::3].set(jnp.zeros((n1+n2+n3+n4+n5, n_contact)))
    contact_sequence = jnp.concatenate([contact1, contact2, contact3, contact4,contact5], axis=0)

    grf_ref = jnp.zeros((N, 3*n_contact))

    return jnp.concatenate([p_ref, quat_ref, q_ref, dp_ref, omega_ref, foot_ref, contact_sequence, grf_ref], axis=1), jnp.concatenate([contact_sequence, foot_ref], axis=1)


def reference_humanoid_jump_forward(
    N,
    dt,
    n_joints,
    n_contact,
    foot0,
    q0,
    *,
    base_height=0.9,
    crouch_height=0.82,
    apex_height=1.02,
    jump_distance=0.35,
    foot_shift=0.18,
    foot_lift=0.12,
):
    n_crouch = max(2, int(0.20 / dt))
    n_flight = max(2, int(0.28 / dt))
    n_land = max(2, int(0.18 / dt))
    n_settle = max(0, N - (n_crouch + n_flight + n_land))

    x_crouch = jnp.linspace(0.0, 0.05, n_crouch)
    x_flight = jnp.linspace(x_crouch[-1], jump_distance, n_flight)
    x_land = jnp.linspace(jump_distance, jump_distance, n_land)
    x_settle = jnp.linspace(jump_distance, jump_distance, n_settle)

    z_crouch = jnp.linspace(base_height, crouch_height, n_crouch)
    phase = jnp.linspace(0.0, 1.0, n_flight)
    z_flight = crouch_height + (base_height - crouch_height) * phase + (apex_height - base_height) * 4.0 * phase * (1.0 - phase)
    z_land = jnp.linspace(base_height, base_height, n_land)
    z_settle = jnp.linspace(base_height, base_height, n_settle)

    x_ref = jnp.concatenate([x_crouch, x_flight, x_land, x_settle], axis=0)
    z_ref = jnp.concatenate([z_crouch, z_flight, z_land, z_settle], axis=0)
    y_ref = jnp.zeros_like(x_ref)
    p_ref = jnp.stack([x_ref, y_ref, z_ref], axis=1)

    quat_ref = jnp.tile(jnp.array([1.0, 0.0, 0.0, 0.0]), (N, 1))
    omega_ref = jnp.zeros((N, 3))

    crouch_q = q0.at[2].set(-0.8).at[3].set(1.5).at[4].set(-0.8)
    crouch_q = crouch_q.at[7].set(-0.8).at[8].set(1.5).at[9].set(-0.8)
    q_crouch = jnp.stack([q0 + (crouch_q - q0) * alpha for alpha in jnp.linspace(0.0, 1.0, n_crouch)], axis=0)
    q_flight = jnp.tile(crouch_q, (n_flight, 1))
    q_land = jnp.stack([crouch_q + (q0 - crouch_q) * alpha for alpha in jnp.linspace(0.0, 1.0, n_land)], axis=0)
    q_settle = jnp.tile(q0, (n_settle, 1))
    q_ref = jnp.concatenate([q_crouch, q_flight, q_land, q_settle], axis=0)

    dp_ref = jnp.zeros((N, 3))
    dp_ref = dp_ref.at[:-1].set((p_ref[1:] - p_ref[:-1]) / dt)
    dp_ref = dp_ref.at[-1].set(dp_ref[-2])

    foot_ref = jnp.tile(foot0, (N, 1))
    flight_shift = foot_shift * phase
    flight_lift = foot_lift * 4.0 * phase * (1.0 - phase)
    foot_flight = jnp.tile(foot0, (n_flight, 1))
    foot_flight = foot_flight.at[:, ::3].set(foot_flight[:, ::3] + flight_shift[:, None])
    foot_flight = foot_flight.at[:, 2::3].set(foot_flight[:, 2::3] + flight_lift[:, None])
    foot_land = jnp.tile(
        foot0.at[::3].set(foot0[::3] + foot_shift),
        (n_land + n_settle, 1),
    )
    foot_ref = foot_ref.at[n_crouch : n_crouch + n_flight].set(foot_flight)
    if n_land + n_settle > 0:
        foot_ref = foot_ref.at[n_crouch + n_flight :].set(foot_land)

    contact_crouch = jnp.tile(jnp.ones(n_contact), (n_crouch, 1))
    contact_flight = jnp.tile(jnp.zeros(n_contact), (n_flight, 1))
    contact_land = jnp.tile(jnp.ones(n_contact), (n_land + n_settle, 1))
    contact_sequence = jnp.concatenate([contact_crouch, contact_flight, contact_land], axis=0)

    grf_ref = jnp.zeros((N, 3 * n_contact))

    reference = jnp.concatenate(
        [p_ref, quat_ref, q_ref, dp_ref, omega_ref, foot_ref, contact_sequence, grf_ref],
        axis=1,
    )
    parameter = jnp.concatenate([contact_sequence, foot_ref], axis=1)
    return reference, parameter


def reference_quadruped_trot_two_step(
    N,
    dt,
    n_joints,
    n_contact,
    foot0,
    q0,
    *,
    base_height=0.36,
    total_forward=0.45,
    step_length=0.16,
    step_height=0.08,
    settle_time=0.10,
    phase_time=0.16,
):
    del n_joints
    n_stance = max(2, int(settle_time / dt))
    n_phase = max(2, int(phase_time / dt))
    n_phases = 4
    n_settle = max(0, N - (n_stance + n_phases * n_phase))

    p_ref = jnp.zeros((N, 3))
    p_ref = p_ref.at[:, 0].set(jnp.linspace(0.0, total_forward, N))
    p_ref = p_ref.at[:, 2].set(base_height)
    quat_ref = jnp.tile(jnp.array([1.0, 0.0, 0.0, 0.0]), (N, 1))
    q_ref = jnp.tile(q0, (N, 1))
    dp_ref = jnp.zeros((N, 3))
    dp_ref = dp_ref.at[:, 0].set(total_forward / ((N - 1) * dt + 1e-6))
    omega_ref = jnp.zeros((N, 3))

    foot_ref = jnp.tile(foot0, (N, 1))
    footholds = foot0.reshape(n_contact, 3)
    contact_sequence = jnp.tile(jnp.ones(n_contact), (N, 1))

    trot_a = jnp.array([1.0, 0.0, 0.0, 1.0])
    trot_b = jnp.array([0.0, 1.0, 1.0, 0.0])
    patterns = [trot_a, trot_b, trot_a, trot_b]

    start_idx = n_stance
    for pattern in patterns:
        end_idx = min(start_idx + n_phase, N)
        contact_sequence = contact_sequence.at[start_idx:end_idx].set(
            jnp.tile(pattern, (end_idx - start_idx, 1))
        )
        swing_ids = jnp.where(pattern == 0.0)[0]
        phase = jnp.linspace(0.0, 1.0, end_idx - start_idx)
        start_feet = footholds
        end_feet = footholds.at[swing_ids, 0].add(step_length)
        swing_xyz = start_feet[None, :, :] + (end_feet - start_feet)[None, :, :] * phase[:, None, None]
        swing_xyz = swing_xyz.at[:, swing_ids, 2].set(
            start_feet[swing_ids, 2][None, :] + step_height * 4.0 * phase[:, None] * (1.0 - phase[:, None])
        )
        foot_ref = foot_ref.at[start_idx:end_idx].set(swing_xyz.reshape(end_idx - start_idx, -1))
        footholds = end_feet
        start_idx = end_idx

    if start_idx < N:
        foot_ref = foot_ref.at[start_idx:].set(jnp.tile(footholds.reshape(-1), (N - start_idx, 1)))
        contact_sequence = contact_sequence.at[start_idx:].set(jnp.tile(jnp.ones(n_contact), (N - start_idx, 1)))

    grf_ref = jnp.zeros((N, 3 * n_contact))
    reference = jnp.concatenate(
        [p_ref, quat_ref, q_ref, dp_ref, omega_ref, foot_ref, contact_sequence, grf_ref],
        axis=1,
    )
    parameter = jnp.concatenate([contact_sequence, foot_ref], axis=1)
    return reference, parameter