"""Shared residual-policy initialization and training switches for PPO and SAC."""

import jax
import jax.numpy as jp
import numpy as np
from brax.training.agents.ppo import losses as ppo_losses
from brax.training.agents.sac import losses as sac_losses
from brax.training.agents.sac import networks as sac_networks
from brax.training.acme import running_statistics
from brax.training.networks import FeedForwardNetwork
from flax.core import FrozenDict, freeze, unfreeze


def make_sac_networks(
    observation_size,
    action_size,
    policy_hidden_layer_sizes,
    critic_hidden_layer_sizes,
    **network_kwargs,
):
    """Builds SAC actor and Q networks with independently configurable widths."""
    policy_bundle = sac_networks.make_sac_networks(
        observation_size=observation_size,
        action_size=action_size,
        hidden_layer_sizes=policy_hidden_layer_sizes,
        **network_kwargs,
    )
    critic_bundle = sac_networks.make_sac_networks(
        observation_size=observation_size,
        action_size=action_size,
        hidden_layer_sizes=critic_hidden_layer_sizes,
        **network_kwargs,
    )
    return policy_bundle.replace(q_network=critic_bundle.q_network)


def initialize_residual(networks, action_size, std):
    """Sets the deterministic tanh-normal action to zero and its initial std."""
    if std <= .001:
        raise ValueError("tanh-normal init std must be greater than min_std=0.001")

    old = networks.policy_network

    def init(key):
        original = old.init(key)
        params = unfreeze(original)
        head_name = max(
            (name for name in params["params"] if name.startswith("hidden_")),
            key=lambda name: int(name.split("_")[-1]),
        )
        head = params["params"][head_name]
        if head["bias"].shape != (2 * action_size,):
            raise ValueError(
                "Expected a tanh-normal output head with "
                f"{2 * action_size} outputs, got {head['bias'].shape}"
            )

        head["kernel"] = jp.zeros_like(head["kernel"])
        bias = jp.zeros_like(head["bias"])
        requested_std = jp.asarray(std, dtype=bias.dtype)
        min_std = jp.asarray(.001, dtype=bias.dtype)
        raw_std = jp.log(jp.expm1(requested_std - min_std))
        head["bias"] = bias.at[action_size:].set(raw_std)
        return freeze(params) if isinstance(original, FrozenDict) else params

    return networks.replace(
        policy_network=FeedForwardNetwork(init=init, apply=old.apply)
    )


def action_prior_terms(
    networks, normalizer_params, policy_params, observation, coefficient
):
    """Returns deterministic-action L2 and its weighted loss contribution."""
    logits = networks.policy_network.apply(
        normalizer_params, policy_params, observation
    )
    action = networks.parametric_action_distribution.mode(logits)
    mean_action_l2 = jp.mean(jp.square(action))
    return mean_action_l2, coefficient * mean_action_l2


# ─────────────────────────────────────────────────────────────────────────────
# SAC: configured loss and critic warm-up
# ─────────────────────────────────────────────────────────────────────────────

def make_configured_sac_loss(
    networks, reward_scaling, discounting, action_size
):
    """Builds SAC losses and adds the configurable deterministic-action prior."""
    alpha_loss, critic_loss, base_actor_loss = sac_losses.make_losses(
        networks, reward_scaling, discounting, action_size
    )

    def actor_loss(
        policy_params,
        normalizer_params,
        critic_params,
        alpha,
        transitions,
        rng,
        action_prior_coefficient,
    ):
        base_value = base_actor_loss(
            policy_params,
            normalizer_params,
            critic_params,
            alpha,
            transitions,
            rng,
        )
        mean_action_l2, action_prior_loss = action_prior_terms(
            networks,
            normalizer_params,
            policy_params,
            transitions.observation,
            action_prior_coefficient,
        )
        return base_value + action_prior_loss, (
            base_value,
            mean_action_l2,
            action_prior_loss,
        )

    return alpha_loss, critic_loss, actor_loss


class SacCriticWarmup:
    """Tracks convergence of the SAC critic while the actor remains frozen."""

    def __init__(self, enabled, min_blocks, max_blocks, window, rel_tol):
        if enabled and (
            window < 1
            or min_blocks < 2 * window
            or max_blocks < min_blocks
            or rel_tol <= 0
        ):
            raise ValueError(
                "Need window >= 1, min_blocks >= 2*window, "
                "max_blocks >= min_blocks, rel_tol > 0"
            )
        self.active = bool(enabled)
        self.min_blocks = min_blocks
        self.max_blocks = max_blocks
        self.window = window
        self.rel_tol = rel_tol
        self.history = []
        self.end_step = 0 if not enabled else None
        self.last_rel_change = float("nan")
        if self.active:
            print(
                "[SAC critic warmup] started: actor, alpha and their optimizer "
                "states are frozen; "
                f"min_blocks={min_blocks}, max_blocks={max_blocks}, "
                f"window={window}, rel_tol={rel_tol:g}",
                flush=True,
            )

    def update(self, block_q_pi, next_step):
        """Records one update block and releases the actor when appropriate."""
        if not self.active:
            return False
        self.history.append(block_q_pi)
        block_count = len(self.history)
        if block_count >= 2 * self.window:
            current = np.mean(self.history[-self.window:])
            previous = np.mean(
                self.history[-2 * self.window:-self.window]
            )
            self.last_rel_change = abs(current - previous) / max(
                abs(previous), 1e-6
            )
        settled = (
            block_count >= self.min_blocks
            and self.last_rel_change < self.rel_tol
        )
        if settled or block_count >= self.max_blocks:
            self.active = False
            self.end_step = next_step
            reason = "converged" if settled else "max_blocks reached"
            print(
                f"[SAC critic warmup] finished after {block_count} blocks "
                f"({reason}), relative change {self.last_rel_change:.4f}",
                flush=True,
            )
            return True
        return False

    def apply(self, new_learner, old_learner, train_actor):
        """Applies the SAC actor/alpha freeze for the current update."""
        return apply_sac_critic_warmup(
            new_learner, old_learner, train_actor
        )


def sac_policy_q_value(
    networks, critic_params, policy_params, normalizer_params, observation
):
    """Returns the mean min-Q of the deterministic SAC policy action."""
    logits = networks.policy_network.apply(
        normalizer_params, policy_params, observation
    )
    action = networks.parametric_action_distribution.mode(logits)
    q_values = networks.q_network.apply(
        normalizer_params, critic_params, observation, action
    )
    return jp.mean(jp.min(q_values, axis=-1))


def apply_sac_critic_warmup(new_learner, old_learner, train_actor):
    """Freezes SAC actor, alpha and their optimizer states during burn-in."""
    result = dict(new_learner)

    def select(new_tree, old_tree):
        return jax.tree_util.tree_map(
            lambda new, old: jp.where(train_actor, new, old),
            new_tree,
            old_tree,
        )

    for param_key, optimizer_key in (
        ("policy", "p_opt"),
        ("logalpha", "a_opt"),
    ):
        result[param_key] = select(
            result[param_key], old_learner[param_key]
        )
        result[optimizer_key] = select(
            result[optimizer_key], old_learner[optimizer_key]
        )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# PPO: configured loss and critic warm-up
# ─────────────────────────────────────────────────────────────────────────────

class PpoCriticWarmup:
    """Represents a PPO run in which only the value network is trained."""

    def __init__(self, enabled):
        self.active = bool(enabled)

    def apply(self, params):
        """Freezes PPO actor gradients when this warm-up is active."""
        return apply_ppo_critic_warmup(params, self.active)


def apply_ppo_critic_warmup(params, critic_warmup):
    """Freezes PPO actor gradients while leaving value gradients unchanged."""
    if not critic_warmup:
        return params
    frozen_policy = jax.tree_util.tree_map(
        jax.lax.stop_gradient, params.policy
    )
    return params.replace(policy=frozen_policy)


_ORIGINAL_COMPUTE_PPO_LOSS = ppo_losses.compute_ppo_loss


def make_configured_ppo_loss(
    loss_nn_action, loss_nn_action_cost, critic_warmup
):
    """Builds a PPO loss using the switches supplied by ``train_srbd.py``."""
    warmup = PpoCriticWarmup(critic_warmup)

    def compute_configured_ppo_loss(
        params, normalizer_params, data, rng, ppo_network, **kwargs
    ):
        loss_params = warmup.apply(params)

        loss, metrics = _ORIGINAL_COMPUTE_PPO_LOSS(
            loss_params, normalizer_params, data, rng, ppo_network, **kwargs
        )
        if loss_nn_action:
            mean_action_l2, action_prior_loss = action_prior_terms(
                ppo_network,
                normalizer_params,
                loss_params.policy,
                data.observation,
                loss_nn_action_cost,
            )
            loss = loss + action_prior_loss
            logits = ppo_network.policy_network.apply(
                normalizer_params, loss_params.policy, data.observation
            )
            deterministic_action = (
                ppo_network.parametric_action_distribution.mode(logits)
            )
            reduce_axes = tuple(range(deterministic_action.ndim - 1))
            action_mean = jp.mean(deterministic_action, axis=reduce_axes)
            action_rms = jp.sqrt(
                jp.mean(jp.square(deterministic_action), axis=reduce_axes)
            )

            normalized_obs = running_statistics.normalize(
                data.observation, normalizer_params, max_abs_value=10.0
            )
            policy_obs = (
                normalized_obs["state"]
                if isinstance(normalized_obs, dict)
                else normalized_obs
            )
            metrics = {
                **metrics,
                "nn_action_mean_l2": mean_action_l2,
                "nn_action_prior_loss": action_prior_loss,
                "nn_action_abs_max": jp.max(jp.abs(deterministic_action)),
                "policy_obs_norm_rms": jp.sqrt(
                    jp.mean(jp.square(policy_obs))
                ),
                "policy_obs_norm_abs_max": jp.max(jp.abs(policy_obs)),
                "policy_obs_norm_clip_fraction": jp.mean(
                    (jp.abs(policy_obs) >= 9.999).astype(jp.float32)
                ),
                **{
                    f"nn_action_mean_{i}": action_mean[i]
                    for i in range(deterministic_action.shape[-1])
                },
                **{
                    f"nn_action_rms_{i}": action_rms[i]
                    for i in range(deterministic_action.shape[-1])
                },
            }

        return loss, {
            **metrics,
            "critic_warmup_active": jp.asarray(
                warmup.active, dtype=jp.float32
            ),
        }

    return compute_configured_ppo_loss


def install_configured_ppo_loss(
    loss_nn_action, loss_nn_action_cost, critic_warmup
):
    """Installs the configured PPO loss for the duration of a training call."""
    ppo_losses.compute_ppo_loss = make_configured_ppo_loss(
        loss_nn_action, loss_nn_action_cost, critic_warmup
    )


def restore_ppo_loss():
    """Restores Brax's unmodified PPO loss."""
    ppo_losses.compute_ppo_loss = _ORIGINAL_COMPUTE_PPO_LOSS
