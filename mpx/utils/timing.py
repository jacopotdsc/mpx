"""Timing bookkeeping for the DFCIP MPC + WBC controller.

All rates are derived from the primitive configuration values
(``simulation_frequency``, ``mpc_frequency``, ``whole_body_frequency``,
``dt_mpc``, ``N``) so that any override of a primitive propagates consistently,
and every derivation is checked explicitly: no silent rounding can alter the
timing. Shared by the config module (self-check at import), the controller
wrapper, the example drivers and the validation harness.
"""
from __future__ import annotations

_TOL = 1e-9


def derive_timing(config) -> dict:
    """Return a dict with the simulation / MPC / WBC timing implied by ``config``.

    Keys:
        sim_f, dt_sim                      MuJoCo integration rate and step
        mpc_f, mpc_period_s                MPC update rate and period
        wbc_f, dt_wbc                      WBC update rate and step (also the WBC
                                           joint-limit prediction step)
        dt_mpc, N, horizon_s               MPC discretisation and horizon
        mpc_period_sim_steps               simulation steps between MPC updates
        wbc_period_sim_steps               simulation steps between WBC updates
        mpc_shift_nodes                    MPC nodes the warm start is shifted by
                                           at every MPC update (NOT simulation steps)

    Raises ``ValueError`` on any inconsistency.
    """
    sim_f = int(config.simulation_frequency)
    mpc_f = int(config.mpc_frequency)
    wbc_f = int(config.whole_body_frequency)
    dt_mpc = float(config.dt_mpc)
    N = int(config.N)
    horizon_required = float(getattr(config, "mpc_horizon_s", 0.5))

    if sim_f <= 0 or mpc_f <= 0 or wbc_f <= 0:
        raise ValueError("simulation_frequency, mpc_frequency and whole_body_frequency must be positive")
    if float(config.simulation_frequency) != sim_f or float(config.mpc_frequency) != mpc_f \
            or float(config.whole_body_frequency) != wbc_f:
        raise ValueError("frequencies must be integers (Hz)")
    if sim_f % mpc_f != 0:
        raise ValueError(f"simulation_frequency ({sim_f} Hz) must be an integer multiple of mpc_frequency ({mpc_f} Hz)")
    if sim_f % wbc_f != 0:
        raise ValueError(f"simulation_frequency ({sim_f} Hz) must be an integer multiple of whole_body_frequency ({wbc_f} Hz)")
    if mpc_f % wbc_f != 0 and wbc_f % mpc_f != 0:
        raise ValueError(f"mpc_frequency ({mpc_f} Hz) and whole_body_frequency ({wbc_f} Hz) must be integer multiples of each other")

    horizon_s = N * dt_mpc
    if abs(horizon_s - horizon_required) > _TOL:
        raise ValueError(f"MPC horizon N*dt_mpc = {N}*{dt_mpc} = {horizon_s:.6f} s differs from the required {horizon_required} s")

    mpc_period_s = 1.0 / mpc_f
    shift_exact = mpc_period_s / dt_mpc
    shift = int(round(shift_exact))
    if shift < 1:
        raise ValueError(f"the MPC update period ({mpc_period_s} s) is shorter than dt_mpc ({dt_mpc} s): shift = {shift_exact:.4f} < 1 node")
    if abs(shift - shift_exact) > _TOL:
        raise ValueError(f"the MPC update period ({mpc_period_s} s) is not an integer multiple of dt_mpc ({dt_mpc} s): shift = {shift_exact:.6f} nodes")

    return dict(
        sim_f=sim_f, dt_sim=1.0 / sim_f,
        mpc_f=mpc_f, mpc_period_s=mpc_period_s,
        wbc_f=wbc_f, dt_wbc=1.0 / wbc_f,
        dt_mpc=dt_mpc, N=N, horizon_s=horizon_s,
        mpc_period_sim_steps=sim_f // mpc_f,
        wbc_period_sim_steps=sim_f // wbc_f,
        mpc_shift_nodes=shift,
    )


def check_model_timestep(model_timestep: float, timing: dict, tol: float = 1e-12) -> None:
    """Raise if the MuJoCo model timestep does not equal 1/simulation_frequency."""
    if abs(float(model_timestep) - timing["dt_sim"]) > tol:
        raise ValueError(
            f"model.opt.timestep = {float(model_timestep):.6g} s but 1/simulation_frequency = "
            f"{timing['dt_sim']:.6g} s ({timing['sim_f']} Hz); make the XML timestep and "
            f"config.simulation_frequency consistent")


def describe_timing(timing: dict, mpc_iterations: int = 1) -> str:
    """One-line human readable summary of the timing scheme."""
    return (f"sim {timing['sim_f']} Hz (dt_sim {timing['dt_sim']:.4f} s) | "
            f"MPC {timing['mpc_f']} Hz: dt_mpc {timing['dt_mpc']} s, N {timing['N']}, horizon {timing['horizon_s']:.2f} s, "
            f"update every {timing['mpc_period_sim_steps']} sim steps = {timing['mpc_shift_nodes']} MPC node(s), "
            f"{mpc_iterations} FDDP iteration(s) per update | "
            f"WBC {timing['wbc_f']} Hz: dt_wbc {timing['dt_wbc']:.4f} s, every {timing['wbc_period_sim_steps']} sim steps")
