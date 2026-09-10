"""Initialisation of the residual policy parameters (Brax PPO, tanh_normal).

With Brax's ``NormalTanhDistribution`` the policy MLP outputs
``[loc | scale_raw]`` and ``scale = softplus(scale_raw) + min_std``. A plain
zero-initialised output layer therefore gives ``loc = 0`` (good: the residual
starts at zero) but ``std = softplus(0) + 0.001 = 0.694`` (bad: essentially
random actions in [-1, 1] from the first step). ``init_noise_std`` of
``make_ppo_networks`` is only used by the 'normal' distribution, so it has no
effect here. This module sets both halves explicitly:

    kernel(last layer) = 0
    bias(last layer)   = [0 ... 0 | softplus^-1(init_std - min_std) ...]

so that the untrained policy is deterministic-zero in mean and has a small,
non-degenerate exploration std that PPO can still adapt.
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
from flax.core import freeze, unfreeze

BRAX_TANH_NORMAL_MIN_STD = 0.001


def softplus_inverse(y: float) -> float:
  return math.log(math.expm1(y))


def _last_dense_name(params_dict: dict) -> str:
  names = [k for k in params_dict.keys() if k.startswith("hidden_")]
  if not names:
    raise KeyError(f"no 'hidden_*' layers in policy params: {list(params_dict.keys())}")
  return sorted(names, key=lambda k: int(k.split("_")[-1]))[-1]


def residual_init_policy_params(policy_params, action_size: int, init_std: float,
                                zero_mean: bool = True, min_std: float = BRAX_TANH_NORMAL_MIN_STD):
  """Return a copy of ``policy_params`` whose output layer yields mean 0 and
  std ``init_std`` for every observation (tanh_normal parametrisation)."""
  if not (min_std < init_std < 1.0):
    raise ValueError(f"init_std must be in ({min_std}, 1), got {init_std}")
  p = unfreeze(policy_params)
  inner = p["params"] if "params" in p else p
  last = _last_dense_name(inner)
  kernel = inner[last]["kernel"]
  bias = inner[last]["bias"]
  if kernel.shape[-1] != 2 * action_size or bias.shape[-1] != 2 * action_size:
    raise ValueError(f"output layer has {kernel.shape[-1]} units, expected 2*{action_size} (tanh_normal)")
  b_scale = softplus_inverse(init_std - min_std)
  new_bias = jnp.concatenate([jnp.zeros(action_size, bias.dtype),
                              jnp.full((action_size,), b_scale, bias.dtype)])
  new_kernel = jnp.zeros_like(kernel) if zero_mean else kernel.at[:, action_size:].set(0.0)
  inner[last]["kernel"] = new_kernel
  inner[last]["bias"] = new_bias
  return freeze(p) if hasattr(policy_params, "unfreeze") or isinstance(policy_params, type(freeze({}))) else p


def describe_policy_output(networks, params, obs, action_size: int, n_samples: int = 2000, seed: int = 0):
  """Measured mean/std of the untrained policy on one observation: raw loc,
  raw scale, deterministic action and empirical std of the sampled action."""
  from brax.training.agents.ppo import networks as ppo_networks
  normalizer_params, policy_params = params[0], params[1]
  raw = networks.policy_network.apply(normalizer_params, policy_params, obs)
  loc, scale_raw = jnp.split(jnp.asarray(raw), 2, axis=-1)
  std = jax.nn.softplus(scale_raw) + BRAX_TANH_NORMAL_MIN_STD
  inference_fn = ppo_networks.make_inference_fn(networks)
  det = inference_fn((normalizer_params, policy_params), deterministic=True)(obs, jax.random.PRNGKey(seed))[0]
  stoch = inference_fn((normalizer_params, policy_params), deterministic=False)
  keys = jax.random.split(jax.random.PRNGKey(seed + 1), n_samples)
  samples = jax.vmap(lambda k: stoch(obs, k)[0])(keys)
  return dict(
      loc=np.asarray(loc), pre_tanh_std=np.asarray(std),
      deterministic_action=np.asarray(det),
      sampled_mean=np.asarray(samples.mean(0)), sampled_std=np.asarray(samples.std(0)),
  )